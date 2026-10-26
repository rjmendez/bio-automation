"""Generic benchmark harness for Topaz Photo AI models against real data.

Built once, deliberately, before any family-specific benchmark: later work
(raw_denoise, auto_classifiers, ...) calls this one shared tool with different
``--family``/``--task``/``--model-dir`` flags instead of writing bespoke
one-off scripts per model. The automation is the point, not any single number
this run produces.

Model directory layout (ground truth for this session, see topaz_bench_config.json):
    C:\\TopazModels\\<stem>\\   already holds EXTRACTED plaintext OpenVINO IR
    (*.xml + *.bin) and/or ONNX (*.onnx) -- no decryption step here. Internal
    filenames inside a cache directory are NOT predictable from the directory
    name (e.g. gendet-v1-fp32-256x256-1x-ov/ holds
    gendet-v1-fp32-256x256-1x-ov.xml, but draw_linear-v1-fp32-512x512-ov/
    holds draw_linear-v1-fp32-512x512.xml -- no "-ov" suffix); always glob,
    never guess a filename.

Model loading and the tile-and-blend inference loop are PORTED (not imported)
from D:\\apps\\bio-automation\\src\\bio_automation\\topaz.py's ModelCache,
_run_tiled and _cosine_window -- that module is production and must not be
imported from here or edited. The port is generalized beyond topaz.py's
original use (OpenVINO-only, always 3-channel RGB, scale hardcoded 1x or 2x
per call site):
  - loads OpenVINO (.xml/.bin, core.read_model + compile_model) OR, when only
    an .onnx file exists, onnxruntime (CPUExecutionProvider) -- several "v2"
    Topaz models only ship as ONNX.
  - the main image input/output and their tile size, channel count, NCHW-vs-
    NHWC layout, dtype (fp32 or fp16) and scale factor are all discovered
    from the compiled model's own static shapes, not hardcoded -- so one
    loader covers 1x denoise (dnt_beta, 512x512, 3ch), 2x upscale (ghqv2-style,
    128x128, 3ch) and 1x classify heads (noise_det, 128x128, 3ch in -> an
    [N]-vector out) without a single model-specific branch.
  - any input beyond the main image tensor (slb's t_param/t_param1, ghqv2's
    pn/pb/pc, draw_linear's two anonymous scalar inputs, ...) is treated
    generically as a "neutral" auxiliary input and fed zeros, exactly as
    topaz.py's upscale_2x() already does for pn/pb/pc.

Known gap, found while building this (not predicted by the research that fed
this harness): drw_native-v1 and drw_standard-v1 (both -ov and -ox variants)
take a 4-channel [512,512,4] input and emit a [1024,1024,3] output -- that is
almost certainly a packed raw Bayer mosaic (RAW-domain denoise+demosaic+2x),
not a 3-channel sRGB denoiser like dnt_beta/draw_linear. This harness's real
pair dataset (pair_manifest.py + precompute_denoise_patches.py) only has
already-demosaiced sRGB (rawpy.postprocess output), so task_denoise() refuses
cleanly (in_channels != 3) rather than silently feeding the wrong data. See
topaz_bench_config.json's "caveats" for those two models.

Two-gate pattern (mimicking benchmark_hat_s.py / benchmark_nafnet.py, not a
new eval protocol):
  denoise   -- real held-out NEF/Enhanced-NR-DNG crop vs plain decode:
               PSNR (validate_real.psnr) + perceptual (validate_real.
               perceptual_scores, piq LPIPS/DISTS).
  upscale   -- gate 1: synthetic-bicubic-degradation PSNR/LPIPS/DISTS vs
               bicubic, on the SAME source-dir convention validate_real.py/
               benchmark_hat_s.py use.
               gate 2: deployment sanity -- a genuinely native, undegraded
               crop (sized to the model's own LR tile, since these are
               fixed-shape IR models, unlike HAT-S's arbitrary-resolution
               transformer) fed straight through, Laplacian-variance
               (validate_real._laplacian_var) vs bicubic upscale of the same
               crop. Not a precision metric, a sanity floor, same as
               validate_real.deployment_sharpness_check's docstring.
  characterize -- for classify_route/segment families with no pixel ground
               truth: run over ~30 real images sampled from the same
               source-dir convention precompute_patches.py's --files-list
               draws from, dump raw output tensors + per-output min/max/mean/
               std to the result JSON. No pass/fail gate -- honest
               characterization only.

CLI:
    python topaz_bench.py --family raw_denoise --task denoise \
        --model-dir "C:\\TopazModels\\dnt_beta-v5-fp32-512x512-ov" \
        --held-out-csv D:\\bio-cache\\sr-train\\denoise\\held_out_pairs.csv \
        --result-json D:\\bio-cache\\sr-train\\bench_results\\dnt_beta_denoise.json

    python topaz_bench.py --family auto_classifiers --task characterize \
        --model-dir "C:\\TopazModels\\gendet-v1-fp32-256x256-1x-ox" \
        --result-json D:\\bio-cache\\sr-train\\bench_results\\gendet_characterize.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402
from train import pick_device  # noqa: E402
from validate_real import psnr, perceptual_scores, _laplacian_var  # noqa: E402
from precompute_denoise_patches import _decode  # noqa: E402

# Default real-image source dir for upscale/characterize sampling: the same
# flat JPEG tree validate_real.py / benchmark_hat_s.py already glob("*.jpg")
# over (F:\Bio-export\adobe-stock\ready, 9802 files, no subdirectories).
DEFAULT_SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")


# ---------------------------------------------------------------------------
# Generic model loader (ported + generalized from bio_automation/topaz.py)
# ---------------------------------------------------------------------------

def _find_model_files(model_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """(.xml, .bin, .onnx) found directly in *model_dir*, or None each.

    Internal filenames are not predictable from the directory name (confirmed
    across several cached dirs during this build), so this always globs.
    """
    xml = sorted(model_dir.glob("*.xml"))
    binf = sorted(model_dir.glob("*.bin"))
    onnx = sorted(model_dir.glob("*.onnx"))
    return (xml[0] if xml else None, binf[0] if binf else None, onnx[0] if onnx else None)


def _np_dtype_from_ov(ov_type: Any) -> np.dtype:
    s = str(ov_type)
    if "float16" in s:
        return np.float16
    if "float32" in s or "float" in s:
        return np.float32
    raise ValueError(f"unsupported OpenVINO element type: {s}")


def _np_dtype_from_ort(ort_type: str) -> np.dtype:
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    raise ValueError(f"unsupported ONNX tensor type: {ort_type}")


def _detect_layout(dims: tuple[int, int, int]) -> tuple[str, int, int, int]:
    """Given a 3-D shape (batch already dropped), guess (layout, C, H, W).

    Every model this harness targets has either a small leading dim (NCHW,
    e.g. (3,512,512)) or a small trailing dim (NHWC, e.g. (256,256,3)) --
    "small" meaning <=4 and strictly less than the other two dims, which is
    never true of a real spatial extent in this project's data. The one
    ambiguous case (a degenerate (1,1,N) classifier output) falls through to
    the NCHW default; characterize() only ever flattens that output, so the
    mislabelled axis order is harmless there. Pixel-output tasks (denoise,
    upscale) never hit the ambiguous case because H and W are always >4.
    """
    d0, d1, d2 = dims
    if d0 <= 4 and d0 < d1 and d0 < d2:
        return "NCHW", d0, d1, d2
    if d2 <= 4 and d2 < d0 and d2 < d1:
        return "NHWC", d2, d0, d1
    return "NCHW", d0, d1, d2


class LoadedModel:
    """A compiled OpenVINO model or ONNX Runtime session plus the generic
    metadata (main input/output name, layout, channels, tile size, dtype,
    scale) `infer_tile` needs to feed and read it without any model-specific
    code at the call site."""

    def __init__(self, kind: str, backend: Any, main_input: str, main_layout: str,
                 in_channels: int, tile_h: int, tile_w: int, in_dtype: np.dtype,
                 extra_inputs: list[tuple[str, tuple[int, ...], np.dtype]],
                 out_channels: int, out_h: int, out_w: int, out_dtype: np.dtype,
                 out_layout: str):
        self.kind = kind
        self.backend = backend
        self.main_input = main_input
        self.main_layout = main_layout
        self.in_channels = in_channels
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.in_dtype = in_dtype
        self.extra_inputs = extra_inputs
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w
        self.out_dtype = out_dtype
        self.out_layout = out_layout

    @property
    def scale(self) -> float:
        """out_h / tile_h -- 1.0 for denoise, 2.0 for a 2x upscaler."""
        return self.out_h / self.tile_h

    def infer_tile(self, tile_chw_f32: np.ndarray) -> np.ndarray:
        """*tile_chw_f32*: [in_channels, tile_h, tile_w] float32, any range
        the model expects (callers pass [0,1]). Returns a float32
        [out_channels, out_h, out_w] array -- NOT clipped, callers clip.
        """
        feed = tile_chw_f32
        if self.main_layout == "NHWC":
            feed = np.transpose(feed, (1, 2, 0))
        feed = feed.astype(self.in_dtype)[np.newaxis]
        feed_dict = {self.main_input: feed}
        for name, shape, dtype in self.extra_inputs:
            feed_dict[name] = np.zeros(shape, dtype=dtype)

        if self.kind == "openvino":
            result = self.backend(feed_dict)
            values = list(result.values())
            if len(values) != 1:
                raise RuntimeError(f"expected exactly 1 model output, got {len(values)}")
            out = np.asarray(values[0])
        else:
            outs = self.backend.run(None, feed_dict)
            if len(outs) != 1:
                raise RuntimeError(f"expected exactly 1 model output, got {len(outs)}")
            out = np.asarray(outs[0])

        out = out[0].astype(np.float32)  # drop batch dim
        if self.out_layout == "NHWC":
            out = np.transpose(out, (2, 0, 1))
        return out


def load_model(model_dir: Path, device: str = "CPU") -> LoadedModel:
    """Load *model_dir* (a C:\\TopazModels\\<stem> directory) via OpenVINO if
    .xml/.bin are present, else via onnxruntime CPUExecutionProvider if only
    .onnx is present ("CPU execution provider is fine here, these models are
    small" -- these are Topaz-size nets, not the GPU-mandated custom SR/DA-CLIP
    models bio_automation runs at scale). Raises FileNotFoundError if neither
    exists, RuntimeError if the model doesn't expose exactly one output.
    """
    model_dir = Path(model_dir)
    xml, binp, onnx = _find_model_files(model_dir)
    if xml is not None:
        import openvino as ov
        core = ov.Core()
        model = core.read_model(str(xml), str(binp)) if binp else core.read_model(str(xml))
        compiled = core.compile_model(model, device)
        inputs_info = [(inp.any_name, tuple(int(d) for d in inp.shape), _np_dtype_from_ov(inp.element_type))
                        for inp in compiled.inputs]
        outputs_info = [(out.any_name, tuple(int(d) for d in out.shape), _np_dtype_from_ov(out.element_type))
                         for out in compiled.outputs]
        kind, backend = "openvino", compiled
    elif onnx is not None:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        inputs_info = [(i.name, tuple(i.shape), _np_dtype_from_ort(i.type)) for i in sess.get_inputs()]
        outputs_info = [(o.name, tuple(o.shape), _np_dtype_from_ort(o.type)) for o in sess.get_outputs()]
        kind, backend = "onnxruntime", sess
    else:
        raise FileNotFoundError(f"no .xml or .onnx found directly in {model_dir}")

    if len(outputs_info) != 1:
        raise RuntimeError(f"{model_dir}: expected exactly 1 model output, got {len(outputs_info)}")

    def _prod(shape: tuple[int, ...]) -> int:
        p = 1
        for d in shape:
            p *= d
        return p

    main = max(inputs_info, key=lambda t: _prod(t[1]))
    extras = [t for t in inputs_info if t[0] != main[0]]

    main_name, main_shape, main_dtype = main
    main_dims = main_shape[1:]  # drop batch
    if len(main_dims) != 3:
        raise RuntimeError(f"{model_dir}: main input {main_name} has unexpected shape {main_shape}")
    main_layout, in_channels, tile_h, tile_w = _detect_layout(main_dims)

    out_name, out_shape, out_dtype = outputs_info[0]
    out_dims = out_shape[1:]
    if len(out_dims) != 3:
        raise RuntimeError(f"{model_dir}: output {out_name} has unexpected shape {out_shape}")
    out_layout, out_channels, out_h, out_w = _detect_layout(out_dims)

    return LoadedModel(kind, backend, main_name, main_layout, in_channels, tile_h, tile_w,
                        main_dtype, extras, out_channels, out_h, out_w, out_dtype, out_layout)


# ---------------------------------------------------------------------------
# Tiling engine (ported + generalized from bio_automation/topaz.py)
# ---------------------------------------------------------------------------

def _cosine_window(size: int) -> np.ndarray:
    """1-D symmetric blend window: sin^2 bump, strictly positive everywhere.
    Identical to topaz.py's _cosine_window."""
    t = np.linspace(0, np.pi, size + 2, dtype=np.float32)[1:-1]
    return np.sin(t) ** 2


def run_tiled(model: LoadedModel, img_chw_f32: np.ndarray, overlap: int) -> np.ndarray:
    """Tile *img_chw_f32* [in_channels, H, W] through *model* with cosine-
    window blending. Generalized from topaz.py's _run_tiled: tile size,
    scale and channel counts all come from *model* instead of being passed
    per call site, so this one function covers 1x denoise, 2x (or any
    integer-scale) upscale, and the degenerate single-tile case (H==tile_h,
    W==tile_w -- a plain direct inference with weight normalization that is
    a no-op). Returns a float32 [out_channels, H*scale, W*scale] array
    clipped to [0, 1].
    """
    C, H, W = img_chw_f32.shape
    if C != model.in_channels:
        raise ValueError(f"image has {C} channels, model expects {model.in_channels}")
    tile_h, tile_w = model.tile_h, model.tile_w
    scale = model.scale
    if scale != int(scale):
        raise ValueError(f"non-integer scale {scale} (out_h={model.out_h}, tile_h={tile_h}) not supported")
    scale = int(scale)

    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    out_H, out_W = H * scale, W * scale

    win_h = _cosine_window(tile_h * scale)[np.newaxis, :, np.newaxis]
    win_w = _cosine_window(tile_w * scale)[np.newaxis, np.newaxis, :]
    win2d = win_h * win_w

    pad_h = max(0, tile_h - H)
    pad_w = max(0, tile_w - W)
    img = img_chw_f32
    if pad_h or pad_w:
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, pH, pW = img.shape
    output = np.zeros((model.out_channels, pH * scale, pW * scale), dtype=np.float32)
    weight = np.zeros((1, pH * scale, pW * scale), dtype=np.float32)

    def _starts(total: int, tile: int, stride: int) -> list[int]:
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(total - tile)
        return pts

    for y in _starts(pH, tile_h, stride_h):
        for x in _starts(pW, tile_w, stride_w):
            tile = img[:, y:y + tile_h, x:x + tile_w]
            result = np.clip(model.infer_tile(tile), 0.0, 1.0)

            oy, ox = y * scale, x * scale
            output[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += result * win2d
            weight[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += win2d

    output /= np.maximum(weight, 1e-12)
    return np.clip(output[:, :out_H, :out_W], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def _to_u8_hwc(chw_f32: np.ndarray) -> np.ndarray:
    hwc = np.transpose(chw_f32, (1, 2, 0))
    return (hwc * 255.0).round().clip(0, 255).astype(np.uint8)


def task_denoise(model: LoadedModel, held_out_csv: Path, n: int, seed: int,
                  overlap: int, torch_device: torch.device) -> dict:
    """1x denoise: model(plain raw decode) vs clean (Enhanced-NR) target, two
    metrics (PSNR, LPIPS/DISTS) vs the plain-decode baseline -- same shape as
    benchmark_nafnet.py's zero-shot check, reusing its held_out_pairs.csv
    convention (plain_raw, enhanced_target columns) directly rather than
    re-deriving the split.
    """
    if model.in_channels != 3:
        return {
            "error": (f"model expects {model.in_channels}-channel input, not 3-channel sRGB -- "
                      "almost certainly a packed raw Bayer mosaic (see drw_native/drw_standard "
                      "caveat in topaz_bench_config.json); this harness's real-pair dataset only "
                      "provides already-demosaiced sRGB, so refusing rather than feeding garbage."),
        }
    rng = random.Random(seed)
    tile = model.tile_h
    with open(held_out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))[:n]

    model_psnrs, plain_psnrs = [], []
    model_preds, plains, targets = [], [], []
    skipped_shape, skipped_identical, skipped_decode = 0, 0, 0
    for row in rows:
        try:
            plain = _decode(row["plain_raw"])
            enhanced = _decode(row["enhanced_target"])
        except Exception as exc:
            print(f"skip {row['plain_raw']}: {exc}", file=sys.stderr)
            skipped_decode += 1
            continue
        if plain.shape != enhanced.shape or plain.shape[0] < tile or plain.shape[1] < tile:
            skipped_shape += 1
            continue
        h, w = plain.shape[:2]
        y0, x0 = rng.randint(0, h - tile), rng.randint(0, w - tile)
        plain_crop = plain[y0:y0 + tile, x0:x0 + tile]
        target_crop = enhanced[y0:y0 + tile, x0:x0 + tile]
        if np.array_equal(plain_crop, target_crop):
            skipped_identical += 1
            continue

        inp_chw = np.transpose(plain_crop.astype(np.float32) / 255.0, (2, 0, 1))
        pred_chw = run_tiled(model, inp_chw, overlap)
        pred_u8 = _to_u8_hwc(pred_chw)

        model_psnrs.append(psnr(pred_u8, target_crop))
        plain_psnrs.append(psnr(plain_crop, target_crop))
        model_preds.append(pred_u8)
        plains.append(plain_crop)
        targets.append(target_crop)

    if not model_psnrs:
        return {"error": "no usable pairs after decode/shape/identical filtering", "n_rows": len(rows)}

    model_avg = sum(model_psnrs) / len(model_psnrs)
    plain_avg = sum(plain_psnrs) / len(plain_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, plain_psnrs))
    model_lpips, model_dists = perceptual_scores(model_preds, targets, torch_device)
    plain_lpips, plain_dists = perceptual_scores(plains, targets, torch_device)

    passed = model_avg > plain_avg and model_lpips < plain_lpips and model_dists < plain_dists
    return {
        "n_pairs": len(model_psnrs),
        "skipped": {"decode_error": skipped_decode, "shape_mismatch": skipped_shape, "bit_identical": skipped_identical},
        "model_psnr_db": round(model_avg, 3),
        "plain_psnr_db": round(plain_avg, 3),
        "delta_db": round(model_avg - plain_avg, 3),
        "wins": wins,
        "model_lpips": round(model_lpips, 5),
        "model_dists": round(model_dists, 5),
        "plain_lpips": round(plain_lpips, 5),
        "plain_dists": round(plain_dists, 5),
        "beats_plain_decode_psnr_and_perceptual": passed,
    }


def task_upscale(model: LoadedModel, source_dir: Path, n: int, seed: int,
                  overlap: int, torch_device: torch.device) -> dict:
    """2x (or model.scale-x) upscale, two gates mirroring benchmark_hat_s.py:
    gate 1 synthetic-bicubic-degradation PSNR/LPIPS/DISTS vs bicubic; gate 2
    a genuinely native, undegraded crop (sized to the model's own LR tile --
    these are fixed-shape IR models, unlike HAT-S's arbitrary-resolution
    transformer, so the crop size is dictated by the model, not a free
    PATCH_SIZE choice) vs bicubic upscale of that same crop, Laplacian
    variance only (no ground truth exists at this scale for a real crop).
    """
    if model.in_channels != 3:
        return {"error": f"model expects {model.in_channels}-channel input, not 3-channel sRGB"}
    scale = int(model.scale)
    if scale < 2:
        return {"error": f"model.scale={model.scale} is not an upscaler"}

    rng = random.Random(seed)
    all_files = sorted(source_dir.glob("*.jpg"))
    if len(all_files) < n + 10:
        return {"error": f"only {len(all_files)} files under {source_dir}, need at least {n + 10}"}
    shuffled = all_files[:]
    random.Random(1234).shuffle(shuffled)  # same split convention as validate_real.py/benchmark_hat_s.py
    held_out_files = shuffled[:n]

    lr_tile = model.tile_h
    hr_patch = lr_tile * scale  # so degrade_hr_to_lr(hr_patch) yields exactly lr_tile

    # --- gate 1: synthetic bicubic degradation vs bicubic ---
    model_psnrs, bicubic_psnrs = [], []
    model_preds, bicubics, hrs = [], [], []
    for p in held_out_files:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width < hr_patch or im.height < hr_patch:
                continue
            hr = random_hr_crop(np.asarray(im), hr_patch, rng)
        lr = degrade_hr_to_lr(hr, rng)  # hr_patch -> hr_patch/2 == lr_tile (scale 2 only; see note below)
        if lr.shape[0] != lr_tile or lr.shape[1] != lr_tile:
            # degrade_hr_to_lr is a fixed 2x downsample; only scale==2 models reach here cleanly.
            continue
        lr_chw = np.transpose(lr.astype(np.float32) / 255.0, (2, 0, 1))
        pred_chw = run_tiled(model, lr_chw, overlap)
        pred_u8 = _to_u8_hwc(pred_chw)
        bicubic_u8 = np.asarray(Image.fromarray(lr).resize((hr_patch, hr_patch), Image.BICUBIC))

        model_psnrs.append(psnr(pred_u8, hr))
        bicubic_psnrs.append(psnr(bicubic_u8, hr))
        model_preds.append(pred_u8)
        bicubics.append(bicubic_u8)
        hrs.append(hr)

    if not model_psnrs:
        return {"error": "no usable crops for gate 1 (check model.scale against degrade_hr_to_lr's fixed 2x)"}

    model_avg = sum(model_psnrs) / len(model_psnrs)
    bicubic_avg = sum(bicubic_psnrs) / len(bicubic_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, bicubic_psnrs))
    model_lpips, model_dists = perceptual_scores(model_preds, hrs, torch_device)
    bicubic_lpips, bicubic_dists = perceptual_scores(bicubics, hrs, torch_device)

    # --- gate 2: deployment sanity, native undegraded crop at the model's own LR tile size ---
    model_vars, bicubic_vars = [], []
    for p in held_out_files[:min(40, len(held_out_files))]:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width < lr_tile or im.height < lr_tile:
                continue
            native = random_hr_crop(np.asarray(im), lr_tile, rng)
        native_chw = np.transpose(native.astype(np.float32) / 255.0, (2, 0, 1))
        pred_chw = run_tiled(model, native_chw, overlap)
        pred_u8 = _to_u8_hwc(pred_chw)
        bicubic_u8 = np.asarray(Image.fromarray(native).resize((lr_tile * scale, lr_tile * scale), Image.BICUBIC))
        model_vars.append(_laplacian_var(np.asarray(Image.fromarray(pred_u8).convert("L"))))
        bicubic_vars.append(_laplacian_var(np.asarray(Image.fromarray(bicubic_u8).convert("L"))))

    model_sharp = sum(model_vars) / len(model_vars) if model_vars else float("nan")
    bicubic_sharp = sum(bicubic_vars) / len(bicubic_vars) if bicubic_vars else float("nan")

    synthetic_pass = model_avg > bicubic_avg and model_lpips < bicubic_lpips and model_dists < bicubic_dists
    deploy_pass = model_sharp > bicubic_sharp
    return {
        "n_pairs": len(model_psnrs),
        "model_psnr_db": round(model_avg, 3),
        "bicubic_psnr_db": round(bicubic_avg, 3),
        "delta_db": round(model_avg - bicubic_avg, 3),
        "wins": wins,
        "model_lpips": round(model_lpips, 5),
        "model_dists": round(model_dists, 5),
        "bicubic_lpips": round(bicubic_lpips, 5),
        "bicubic_dists": round(bicubic_dists, 5),
        "synthetic_gate_passed": synthetic_pass,
        "n_deploy_crops": len(model_vars),
        "model_laplacian_var": round(model_sharp, 3),
        "bicubic_laplacian_var": round(bicubic_sharp, 3),
        "deployment_gate_passed": deploy_pass,
    }


def task_characterize(model: LoadedModel, source_dir: Path, n: int, seed: int) -> dict:
    """No ground truth exists for classify_route/segment outputs: run the
    model over ~30 real images, record every raw output vector, and report
    honest per-output min/max/mean/std -- no pass/fail gate.
    """
    rng = random.Random(seed)
    all_files = sorted(source_dir.glob("*.jpg"))
    if len(all_files) < n:
        return {"error": f"only {len(all_files)} files under {source_dir}, need {n}"}
    shuffled = all_files[:]
    random.Random(1234).shuffle(shuffled)
    sample_files = shuffled[:n]

    tile = model.tile_h
    raw_results = []
    for p in sample_files:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.width < tile or im.height < model.tile_w:
                continue
            crop = random_hr_crop(np.asarray(im), tile, rng)
        chw = np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))
        out = model.infer_tile(chw)  # [out_channels, out_h, out_w], usually degenerate (N,1,1)
        flat = [float(v) for v in out.astype(np.float64).reshape(-1)]
        raw_results.append({"file": p.name, "output": flat})

    if not raw_results:
        return {"error": "no files large enough to crop"}

    arr = np.array([r["output"] for r in raw_results], dtype=np.float64)
    stats = {
        "n_samples": len(raw_results),
        "output_dim": int(arr.shape[1]),
        "per_output_min": arr.min(axis=0).tolist(),
        "per_output_max": arr.max(axis=0).tolist(),
        "per_output_mean": arr.mean(axis=0).tolist(),
        "per_output_std": arr.std(axis=0).tolist(),
    }
    return {"raw_results": raw_results, "stats": stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="a key from topaz_bench_config.json, carried into the result JSON")
    ap.add_argument("--task", required=True, choices=["denoise", "upscale", "characterize"])
    ap.add_argument("--model-dir", type=Path, required=True, help=r"e.g. C:\TopazModels\dnt_beta-v5-fp32-512x512-ov")
    ap.add_argument("--result-json", type=Path, required=True)
    ap.add_argument("--device", default="CPU", help="OpenVINO compile device; ignored for ONNX models (CPU EP always)")
    ap.add_argument("--held-out-csv", type=Path, default=None,
                     help="denoise only: precompute_denoise_patches.py's held_out_pairs.csv (plain_raw,enhanced_target)")
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                     help="upscale/characterize only: flat dir of source JPEGs")
    ap.add_argument("--n", type=int, default=30, help="characterize: sample count; denoise/upscale: held-out count")
    ap.add_argument("--overlap", type=int, default=None, help="tile blend overlap in input px; default tile//8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda-name-substr", type=str, default=None,
                     help="torch device substring for LPIPS/DISTS (perceptual_scores); unrelated to the OpenVINO device")
    args = ap.parse_args()

    model = load_model(args.model_dir, device=args.device)
    print(f"loaded {args.model_dir.name}: backend={model.kind} main_input={model.main_input!r} "
          f"in={model.in_channels}x{model.tile_h}x{model.tile_w} ({model.main_layout}, {model.in_dtype.__name__}) "
          f"out={model.out_channels}x{model.out_h}x{model.out_w} scale={model.scale}")

    overlap = args.overlap if args.overlap is not None else max(4, model.tile_h // 8)
    torch_device = pick_device(args.cuda_name_substr)

    t0 = time.time()
    if args.task == "denoise":
        if args.held_out_csv is None:
            raise SystemExit("--held-out-csv is required for --task denoise")
        result = task_denoise(model, args.held_out_csv, args.n, args.seed, overlap, torch_device)
    elif args.task == "upscale":
        result = task_upscale(model, args.source_dir, args.n, args.seed, overlap, torch_device)
    else:
        result = task_characterize(model, args.source_dir, args.n, args.seed)
    elapsed = time.time() - t0

    payload = {
        "family": args.family,
        "task": args.task,
        "model_dir": str(args.model_dir),
        "backend": model.kind,
        "elapsed_sec": round(elapsed, 2),
        **result,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.result_json}  ({elapsed:.1f}s)")
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        for k, v in result.items():
            if k not in ("raw_results",):
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
