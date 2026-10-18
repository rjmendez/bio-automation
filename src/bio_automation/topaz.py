"""Topaz Photo AI standalone inference module.

Decrypts .tz (ZipCrypto-encrypted ZIPs) model archives, caches extracted
OpenVINO IR models on disk, and runs tiled inference for lens-sharpening and
2× upscaling.

All I/O uses Pillow (no cv2 dependency).
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

_PASSWORD = b"a0331add513e4e6fae035cb12e4b11e5"
MODELS_DIR = Path(r"C:\ProgramData\Topaz Labs LLC\Topaz Photo AI\models")
CACHE_DIR = Path(r"C:\TopazModels")


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _is_complete_zip(path: Path) -> bool:
    """Return True iff *path* contains an End-of-Central-Directory record.

    Topaz streams .tz files as they download; an absent EOCD (``PK\\x05\\x06``)
    means the file is still being written and cannot be opened as a ZIP.
    """
    EOCD = b"PK\x05\x06"
    size = path.stat().st_size
    read_size = min(65536, size)
    with open(path, "rb") as f:
        f.seek(size - read_size)
        tail = f.read(read_size)
    return EOCD in tail


def extract_tz(tz_path: Path, dest_dir: Path) -> dict[str, Path]:
    """Decrypt a .tz archive and extract all members flat into *dest_dir*.

    Raises ``RuntimeError`` if *tz_path* is a streaming/truncated ZIP that
    has not yet finished downloading.

    Returns a mapping from the original internal filename (basename only) to
    the extracted Path on disk.
    """
    if not _is_complete_zip(tz_path):
        raise RuntimeError(
            f"{tz_path.name} is not yet fully downloaded (truncated ZIP)"
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(tz_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            out_path = dest_dir / name
            data = zf.read(member.filename, pwd=_PASSWORD)
            out_path.write_bytes(data)
            extracted[name] = out_path
    return extracted


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """LRU in-memory cache of compiled OpenVINO models with persistent disk cache.

    The on-disk extracted files are never deleted; only the in-memory compiled
    model objects are evicted when *max_models* is exceeded.
    """

    def __init__(self, max_models: int = 3, device: str = "CPU",
                 cache_dir: Path = CACHE_DIR):
        self._max = max_models
        self._device = device
        self._cache_dir = Path(cache_dir)
        self._lru: OrderedDict[str, object] = OrderedDict()   # stem -> compiled

    # ------------------------------------------------------------------
    def _xml_for(self, tz_path: Path) -> Path | None:
        """Return existing .xml path from the persistent cache, or None."""
        model_dir = self._cache_dir / tz_path.stem
        if model_dir.is_dir():
            xmls = list(model_dir.glob("*.xml"))
            if xmls:
                return xmls[0]
        return None

    def _ensure_extracted(self, tz_path: Path) -> Path:
        """Extract *tz_path* to the persistent cache dir (or a temp dir) and
        return the directory containing the .xml file.

        RuntimeError from a truncated/incomplete ZIP propagates immediately.
        OSError (cache dir not writable) triggers a temp-dir fallback.
        """
        xml = self._xml_for(tz_path)
        if xml is not None:
            return xml.parent

        try:
            dest = self._cache_dir / tz_path.stem
            dest.mkdir(parents=True, exist_ok=True)
        except OSError:
            dest = Path(tempfile.mkdtemp())

        # RuntimeError (truncated ZIP) propagates to the caller
        extract_tz(tz_path, dest)

        xmls = list(dest.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"No .xml found after extracting {tz_path.name}")
        return xmls[0].parent

    def load(self, tz_path: Path):
        """Return a compiled OpenVINO model for *tz_path* (LRU cached)."""
        import openvino as ov  # imported lazily so tests can mock it

        key = tz_path.stem
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        model_dir = self._ensure_extracted(tz_path)
        xmls = list(model_dir.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"No .xml found in {model_dir}")
        xml_path = xmls[0]
        bin_path = xml_path.with_suffix(".bin")

        core = ov.Core()
        # Pass both xml and bin explicitly; OpenVINO 2026 requires this for
        # models whose .bin stem differs from the .xml stem (e.g. ghqv2).
        if bin_path.exists():
            model = core.read_model(str(xml_path), str(bin_path))
        else:
            model = core.read_model(str(xml_path))
        compiled = core.compile_model(model, self._device)

        # Evict oldest if over capacity
        while len(self._lru) >= self._max:
            self._lru.popitem(last=False)

        self._lru[key] = compiled
        return compiled

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._lru.clear()


# ---------------------------------------------------------------------------
# Tiling engine
# ---------------------------------------------------------------------------

def _cosine_window(size: int) -> np.ndarray:
    """1-D cosine blend window (peaks at 1.0 in the centre, tapers to 0)."""
    t = np.linspace(0, np.pi, size, dtype=np.float32)
    return (1.0 - np.cos(t)) / 2.0


def _run_tiled(
    compiled,
    img_chw_f32: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int,
    scale: int,
    feed_fn: Callable[[np.ndarray, object], np.ndarray],
    out_key: str,
) -> np.ndarray:
    """Tile *img_chw_f32* [C,H,W] through *compiled* model with cosine blending.

    *feed_fn(tile_chw_f32, compiled)* must return a float32 array
    [C, tile_h*scale, tile_w*scale].

    Returns a float32 [C, H*scale, W*scale] array (values clamped to [0,1]).
    """
    C, H, W = img_chw_f32.shape
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    out_H = H * scale
    out_W = W * scale
    output = np.zeros((C, out_H, out_W), dtype=np.float32)
    weight = np.zeros((1, out_H, out_W), dtype=np.float32)

    win_h = _cosine_window(tile_h * scale)[np.newaxis, :, np.newaxis]
    win_w = _cosine_window(tile_w * scale)[np.newaxis, np.newaxis, :]
    win2d = win_h * win_w  # [1, tile_h*scale, tile_w*scale]

    # Pad input so every tile fits without going out of bounds
    pad_h = max(0, tile_h - H)
    pad_w = max(0, tile_w - W)
    if pad_h or pad_w:
        img_chw_f32 = np.pad(img_chw_f32, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, pH, pW = img_chw_f32.shape

    # Build start positions: regular stride + one extra snap-to-end tile
    def _starts(total: int, tile: int, stride: int) -> list[int]:
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(total - tile)
        return pts

    for y in _starts(pH, tile_h, stride_h):
        for x in _starts(pW, tile_w, stride_w):
            tile = img_chw_f32[:, y:y + tile_h, x:x + tile_w]
            result = feed_fn(tile, compiled)  # [C, th*s, tw*s]

            oy = y * scale
            ox = x * scale
            oy_end = min(oy + tile_h * scale, out_H)
            ox_end = min(ox + tile_w * scale, out_W)
            rh = oy_end - oy
            rw = ox_end - ox

            output[:, oy:oy_end, ox:ox_end] += result[:, :rh, :rw] * win2d[:, :rh, :rw]
            weight[:, oy:oy_end, ox:ox_end] += win2d[:, :rh, :rw]

    weight = np.maximum(weight, 1e-6)
    output /= weight
    return np.clip(output, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def _input_names_of(compiled) -> frozenset[str]:
    """Return the set of input tensor ``any_name`` values for a compiled model.

    Falls back to an empty frozenset if the model doesn't expose ``.inputs``
    (e.g. a test mock).
    """
    try:
        return frozenset(inp.any_name for inp in compiled.inputs)
    except Exception:
        return frozenset()


def sharpen(img_bgr: np.ndarray, cache: ModelCache,
            blur: float = 0.5, noise: float = 0.0) -> np.ndarray:
    """Lens-blur sharpening via slb-v1-fp32-512x512-ov.tz, 512×512 tiles."""
    tz = MODELS_DIR / "slb-v1-fp32-512x512-ov.tz"
    compiled = cache.load(tz)

    # BGR uint8 -> RGB float32 CHW
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    blur_val = np.array([blur], dtype=np.float32)
    noise_val = np.array([noise], dtype=np.float32)

    def feed(tile: np.ndarray, mdl) -> np.ndarray:
        results = mdl({"netInput": tile[np.newaxis], "t_param": blur_val, "t_param1": noise_val})
        return next(iter(results.values()))[0]                   # [3,512,512]

    result_chw = _run_tiled(compiled, img_chw, 512, 512, 64, 1, feed, "tl_unet/netOutput")

    # CHW float32 -> HWC uint8 BGR
    result_rgb = np.transpose(result_chw, (1, 2, 0))
    result_bgr = result_rgb[:, :, ::-1]
    return (result_bgr * 255.0).clip(0, 255).astype(np.uint8)


def upscale_2x(img_bgr: np.ndarray, cache: ModelCache) -> np.ndarray:
    """2× Gigapixel HQ upscale via ghqv2-v1-fp16-128x128-2x-ov.tz, 128×128 tiles."""
    tz = MODELS_DIR / "ghqv2-v1-fp16-128x128-2x-ov.tz"
    compiled = cache.load(tz)

    # Probe actual input names at runtime; the model may have optional colour-
    # classifier inputs (infer_input / infer_output) that we zero-fill.
    known = _input_names_of(compiled)

    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    # Pre-allocate zero placeholders for the gclc colour-classifier bypass
    _z128 = np.zeros((1, 3, 128, 128), dtype=np.float32)
    _z256 = np.zeros((1, 3, 256, 256), dtype=np.float32)
    _z1 = np.zeros((1,), dtype=np.float32)

    def feed(tile: np.ndarray, mdl) -> np.ndarray:
        feed_dict: dict = {"netInput": tile[np.newaxis]}        # [1,3,128,128]
        # Optional scalar params (zero = neutral)
        for pname in ("pn", "pb", "pc"):
            if not known or pname in known:
                feed_dict[pname] = _z1
        # gclc colour-classifier placeholders (zero-fill skips the imcc step)
        if not known or "infer_input" in known:
            feed_dict["infer_input"] = _z128                    # [1,3,128,128]
        if not known or "infer_output" in known:
            feed_dict["infer_output"] = _z256                   # [1,3,256,256]
        results = mdl(feed_dict)
        return next(iter(results.values()))[0]                  # [3,256,256]

    result_chw = _run_tiled(compiled, img_chw, 128, 128, 16, 2, feed, "netOutput")

    result_rgb = np.transpose(result_chw, (1, 2, 0))
    result_bgr = result_rgb[:, :, ::-1]
    return (result_bgr * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def enhance(src: Path, dst: Path, mode: str = "stock",
            cache: ModelCache | None = None,
            min_free_gb: float = 5.0) -> None:
    """Enhance *src* JPEG and write the result to *dst* JPEG.

    Modes
    -----
    stock   : sharpen only (full-resolution output)
    inat    : sharpen only (same, tuned label for smaller 2048 px renders)
    upscale : sharpen then 2× upscale
    """
    src, dst = Path(src), Path(dst)

    # Disk guard
    check_dir = dst.parent if dst.parent.exists() else src.parent
    free_gb = shutil.disk_usage(check_dir).free / 1e9
    if free_gb < min_free_gb:
        raise OSError(
            f"Insufficient disk space: {free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
        )

    own_cache = cache is None
    if own_cache:
        cache = ModelCache()

    try:
        pil_img = Image.open(src).convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        img_bgr = img_rgb[:, :, ::-1].copy()

        if mode in ("stock", "inat"):
            result_bgr = sharpen(img_bgr, cache)
        elif mode == "upscale":
            sharpened = sharpen(img_bgr, cache)
            result_bgr = upscale_2x(sharpened, cache)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        result_rgb = result_bgr[:, :, ::-1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb).save(str(dst), "JPEG", quality=97, subsampling=0)
    finally:
        if own_cache:
            cache.__exit__(None, None, None)
