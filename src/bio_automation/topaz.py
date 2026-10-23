"""Topaz Photo AI standalone inference module.

Decrypts .tz (ZipCrypto-encrypted ZIPs) model archives and .tz2
(HKDF-SHA256 + ChaCha20-Poly1305) archives, caches extracted OpenVINO IR
models on disk, and runs tiled inference for refocus sharpening (the slb
model, Topaz "Sharpen Refocus") and 2× upscaling.

**Private use only** — requires a licensed Topaz Photo AI installation.
The extracted model weights are Topaz Labs IP; do not redistribute.

All I/O uses Pillow (no cv2 dependency).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import settings

_PASSWORD = b"a0331add513e4e6fae035cb12e4b11e5"
_TZ2_IKM   = b"a0331add513e4e6f"   # first 16 bytes of password; hardcoded in aiengine.dll .rdata
_TZ2_INFO  = b"payload"            # separate 7-byte constant in .rdata
_TZ2_MAGIC = b"com.topazlabs.tz2\x00"
_TZ2_BLOCK = 64 * 1024 * 1024      # 64 MiB per ChaCha20-Poly1305 block
_TZ2_TAG   = 16                    # Poly1305 tag bytes

MODELS_DIR = settings.TOPAZ_MODELS_DIR
CACHE_DIR = settings.MODEL_CACHE_DIR

DEFAULT_DEVICE: str = os.environ.get("TOPAZ_DEVICE", "CPU")
# NOTE: topaz.py models are all OpenVINO IR (.xml/.bin). CPU is the default.
# UNVERIFIED: an earlier note claimed the GPU plugin (OpenCL) is slower than CPU
# (slb 820 ms vs 660 ms); a re-measurement found slb at ~0.95-1.15 s per tile on
# CPU and GPU was not re-measured. Set TOPAZ_DEVICE to try another device.


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


def extract_tz2(tz2_path: Path, dest_dir: Path) -> dict[str, Path]:
    """Decrypt a .tz2 file (HKDF-SHA256 + ChaCha20-Poly1305) and extract members.

    .tz2 container (after decryption):
        uint64 count | N×(char[112] name | uint64 data_offset | uint64 data_size) | blobs

    IKM and INFO are hardcoded in aiengine.dll .rdata (recovered via Frida, 2026-10-18).
    Returns a mapping from internal filename (basename) to extracted Path.
    """
    try:
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as exc:
        raise ImportError("pip install cryptography  # required for .tz2 decryption") from exc

    raw = tz2_path.read_bytes()
    if not raw.startswith(_TZ2_MAGIC):
        raise ValueError(f"{tz2_path.name}: bad magic {raw[:18]!r}")

    salt = raw[18:34]
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=_TZ2_INFO).derive(_TZ2_IKM)
    cipher = ChaCha20Poly1305(key)

    # Decrypt all blocks
    ct = raw[34:]
    total = len(ct)
    n_full, remainder = divmod(total, _TZ2_BLOCK + _TZ2_TAG)
    n_blocks = n_full + (1 if remainder > _TZ2_TAG else 0)

    plaintext_parts: list[bytes] = []
    off = 0
    for i in range(n_blocks):
        is_last = i == n_blocks - 1
        nonce = b"\x00" * 10 + bytes([i & 0xFF, 0x01 if is_last else 0x00])
        chunk = ct[off: off + _TZ2_BLOCK + _TZ2_TAG]
        plaintext_parts.append(cipher.decrypt(nonce, chunk, None))
        off += len(chunk)
    plain = b"".join(plaintext_parts)

    # Parse container: uint64 count | N×(name[112] | offset[8] | size[8]) | blobs
    import struct
    count = struct.unpack_from("<Q", plain, 0)[0]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for i in range(count):
        base = 8 + i * 128
        name = plain[base: base + 112].rstrip(b"\x00").decode("utf-8")
        d_off = struct.unpack_from("<Q", plain, base + 112)[0]
        d_sz  = struct.unpack_from("<Q", plain, base + 120)[0]
        out = dest_dir / Path(name).name
        out.write_bytes(plain[d_off: d_off + d_sz])
        result[Path(name).name] = out
    return result




class ModelCache:
    """LRU in-memory cache of compiled OpenVINO models with persistent disk cache.

    The on-disk extracted files are never deleted; only the in-memory compiled
    model objects are evicted when *max_models* is exceeded.
    """

    def __init__(self, max_models: int = 3, device: str = DEFAULT_DEVICE,
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

        dest = self._cache_dir / tz_path.stem
        created = not dest.exists()
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError:
            dest = Path(tempfile.mkdtemp(dir=settings.scratch_dir()))
            created = False

        try:
            # RuntimeError (truncated ZIP) propagates to the caller
            if tz_path.suffix == ".tz2":
                extract_tz2(tz_path, dest)
            else:
                extract_tz(tz_path, dest)

            xmls = list(dest.glob("*.xml"))
            if not xmls:
                raise FileNotFoundError(f"No .xml found after extracting {tz_path.name}")
        except BaseException:
            # Do not leave an empty/partial cache dir behind on failure.
            if created:
                shutil.rmtree(dest, ignore_errors=True)
            raise
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
    """1-D symmetric blend window: sin^2 bump, strictly positive everywhere.

    The interior samples of ``linspace(0, pi, size + 2)`` are used, so the
    first and last sample are small but non-zero and no output pixel can end
    up with zero total weight. (The previous (1-cos)/2 ramp over [0, pi] was
    0 at index 0 and 1 at the last index: it zeroed the first output row and
    column and favoured each tile's worst-context trailing edge.)
    """
    t = np.linspace(0, np.pi, size + 2, dtype=np.float32)[1:-1]
    return np.sin(t) ** 2


def _run_tiled(
    compiled,
    img_chw_f32: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int,
    scale: int,
    feed_fn: Callable[[np.ndarray, object], np.ndarray],
    out_key: str | None = None,
) -> np.ndarray:
    """Tile *img_chw_f32* [C,H,W] through *compiled* model with window blending.

    *feed_fn(tile_chw_f32, compiled)* must return a float32 array
    [C, tile_h*scale, tile_w*scale]. Each tile result is clipped to [0,1]
    before blending so one hot tile cannot pull its neighbours.

    *out_key* is accepted for call-site compatibility and ignored: the output
    is whatever *feed_fn* returns.

    Returns a float32 [C, H*scale, W*scale] array in [0,1].
    """
    C, H, W = img_chw_f32.shape
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap

    out_H = H * scale
    out_W = W * scale

    win_h = _cosine_window(tile_h * scale)[np.newaxis, :, np.newaxis]
    win_w = _cosine_window(tile_w * scale)[np.newaxis, np.newaxis, :]
    win2d = win_h * win_w  # [1, tile_h*scale, tile_w*scale]

    # Pad input so every tile fits without going out of bounds
    pad_h = max(0, tile_h - H)
    pad_w = max(0, tile_w - W)
    if pad_h or pad_w:
        img_chw_f32 = np.pad(img_chw_f32, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

    _, pH, pW = img_chw_f32.shape
    # Accumulate over the padded extent, crop at the end.
    output = np.zeros((C, pH * scale, pW * scale), dtype=np.float32)
    weight = np.zeros((1, pH * scale, pW * scale), dtype=np.float32)

    # Build start positions: regular stride + one extra snap-to-end tile
    def _starts(total: int, tile: int, stride: int) -> list[int]:
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(total - tile)
        return pts

    for y in _starts(pH, tile_h, stride_h):
        for x in _starts(pW, tile_w, stride_w):
            tile = img_chw_f32[:, y:y + tile_h, x:x + tile_w]
            result = np.clip(feed_fn(tile, compiled), 0.0, 1.0)  # [C, th*s, tw*s]

            oy = y * scale
            ox = x * scale
            output[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += result * win2d
            weight[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += win2d

    output /= np.maximum(weight, 1e-12)
    return np.clip(output[:, :out_H, :out_W], 0.0, 1.0)


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


def _single_output(results) -> np.ndarray:
    """Return the only output of an inference result, failing loudly otherwise.

    slb has one output (``tl_unet/add_4``), ghqv2 one (``netOutput``) and
    slsp_ap one (``Identity:0``). Anything else means the wrong model file.
    """
    values = list(results.values())
    if len(values) != 1:
        raise RuntimeError(f"expected exactly 1 model output, got {len(values)}")
    return values[0]


# Defaults for sharpen(). The slb model is Topaz "Sharpen Refocus"
# (inputs param1 -> t_param, param2 -> t_param1), NOT Lens Blur. Measured on
# 9 real frames: t_param 0.25 raises subject Laplacian variance ~1.2x with
# negligible halos; t_param >= 0.5 gives 9-13x and visible halos / invented
# texture. t_param1 > 0 amplifies flat-region noise.
SHARPEN_T_PARAM = 0.25
SHARPEN_T_PARAM1 = 0.0
SHARPEN_ALPHA = 0.5          # blend: out = in + alpha * (model - in)

# Gate thresholds (centre 512 crop, 0-255 luminance units).
MIN_SUBJECT_LAP_VAR = 580.0  # below this the input is too soft; model invents texture
MOTION_SKIP = 0.3            # autoparam output p[2]; high = blurred input
MAX_LAP_RATIO = 2.0          # reject output whose Laplacian variance grows more
MAX_OVERSHOOT_GROWTH = 0.005  # reject output with more extra halo pixels than this
_OVERSHOOT_LEVELS = 8.0
_CENTRE = 512


def _centre_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Luminance (0-255 float32) of the centre 512x512 crop of a BGR image."""
    h, w = img_bgr.shape[:2]
    y0 = max(0, h // 2 - _CENTRE // 2)
    x0 = max(0, w // 2 - _CENTRE // 2)
    crop = img_bgr[y0:y0 + _CENTRE, x0:x0 + _CENTRE].astype(np.float32)
    return crop[..., 0] * 0.114 + crop[..., 1] * 0.587 + crop[..., 2] * 0.299


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior of *gray*."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    c = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * c
    return float(lap.var())


def _overshoot_fraction(src_gray: np.ndarray, out_gray: np.ndarray) -> float:
    """Fraction of pixels where *out* exceeds the local 7x7 min/max of *src*
    by more than 8 levels (a halo / ringing proxy)."""
    from numpy.lib.stride_tricks import sliding_window_view

    if src_gray.shape[0] < 7 or src_gray.shape[1] < 7:
        return 0.0
    win = sliding_window_view(src_gray, (7, 7))
    lo = win.min(axis=(-1, -2))
    hi = win.max(axis=(-1, -2))
    o = out_gray[3:-3, 3:-3]
    over = (o > hi + _OVERSHOOT_LEVELS) | (o < lo - _OVERSHOOT_LEVELS)
    return float(over.mean())


def _gate_input(img_bgr: np.ndarray, motion: float,
                min_lap_var: float = MIN_SUBJECT_LAP_VAR) -> str | None:
    """Return a skip reason if sharpening *img_bgr* is unsafe, else None."""
    if motion >= MOTION_SKIP:
        return f"autoparam motion score {motion:.2f} >= {MOTION_SKIP}"
    lap = _laplacian_var(_centre_gray(img_bgr))
    if lap < min_lap_var:
        return f"input too soft (Laplacian variance {lap:.1f} < {min_lap_var})"
    return None


def _gate_output(src_bgr: np.ndarray, out_bgr: np.ndarray) -> str | None:
    """Return a rejection reason if *out_bgr* looks hallucinated, else None."""
    sg, og = _centre_gray(src_bgr), _centre_gray(out_bgr)
    base = _laplacian_var(sg)
    if base > 0 and _laplacian_var(og) / base > MAX_LAP_RATIO:
        return "Laplacian variance ratio above limit"
    if _overshoot_fraction(sg, og) > MAX_OVERSHOOT_GROWTH:
        return "halo overshoot above limit"
    return None


def _estimate_sharpen_params(img_chw_f32: np.ndarray, cache: ModelCache) -> tuple[float, float, float, float]:
    """Run slsp_ap on a centre crop; return its 4 outputs clipped to [0,1].

    The outputs are Topaz's ``parameterNames`` order (noise, lensblur,
    motionblur, compression) but measured behaviour is looser: p[0] falls as
    the image blurs and rises with noise; p[1] is near 0 below gaussian
    sigma 2; p[2] rises with ANY blur; p[3] is flat (~0.44-0.55) even for
    JPEG q8. Raw outputs can be negative; they are clipped here. Do not drive
    a sharpening strength from these values.

    Falls back to (0.0, 0.5, 0.0, 0.0) if the .tz2 model is unavailable.
    """
    ap_tz2 = MODELS_DIR / "slsp_ap-v3-fp32-512x512-ov.tz2"
    if not ap_tz2.exists():
        return (0.0, 0.5, 0.0, 0.0)
    try:
        compiled = cache.load(ap_tz2)
    except Exception:
        return (0.0, 0.5, 0.0, 0.0)

    _, H, W = img_chw_f32.shape
    cy, cx = H // 2, W // 2
    h2, w2 = min(256, H // 2), min(256, W // 2)
    crop = img_chw_f32[:, cy - h2:cy + h2, cx - w2:cx + w2]
    # Pad to 512x512
    ph = max(0, 512 - crop.shape[1])
    pw = max(0, 512 - crop.shape[2])
    if ph or pw:
        crop = np.pad(crop, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    crop = crop[:, :512, :512]

    results = compiled({"netInput": crop[np.newaxis]})
    params = _single_output(results)[0]  # [4]
    return tuple(float(np.clip(p, 0.0, 1.0)) for p in params)


def sharpen(img_bgr: np.ndarray, cache: ModelCache,
            blur: float | None = None, noise: float = SHARPEN_T_PARAM1,
            auto_params: bool = True, alpha: float = SHARPEN_ALPHA,
            gate: bool = True) -> np.ndarray:
    """Refocus sharpening via slb-v1-fp32-512x512-ov.tz (Topaz Sharpen
    Refocus, NOT Lens Blur), 512x512 tiles, blended with the input.

    *blur* / *noise* are slb's ``t_param`` / ``t_param1`` (defaults 0.25 / 0).
    The result is ``in + alpha * (model - in)``. With *gate* True the call
    returns *img_bgr itself* (same object, nothing re-quantised) when the
    input is too soft or motion-blurred, or when the model output shows
    excess Laplacian gain or halos; callers can detect that with ``is``.
    *auto_params* only feeds the motion-blur skip rule, never the strength.
    """
    tz = MODELS_DIR / "slb-v1-fp32-512x512-ov.tz"
    compiled = cache.load(tz)

    # BGR uint8 -> RGB float32 CHW
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    if gate:
        motion = 0.0
        if auto_params:
            _, _, motion, _ = _estimate_sharpen_params(img_chw, cache)
        if _gate_input(img_bgr, motion) is not None:
            return img_bgr

    t_param = SHARPEN_T_PARAM if blur is None else blur
    blur_val  = np.array([t_param], dtype=np.float32)
    noise_val = np.array([noise], dtype=np.float32)

    def feed(tile: np.ndarray, mdl) -> np.ndarray:
        results = mdl({"netInput": tile[np.newaxis], "t_param": blur_val, "t_param1": noise_val})
        return _single_output(results)[0]                        # [3,512,512]

    result_chw = _run_tiled(compiled, img_chw, 512, 512, 128, 1, feed)
    blended = img_chw + alpha * (result_chw - img_chw)

    # CHW float32 -> HWC uint8 BGR
    result_rgb = np.transpose(blended, (1, 2, 0))
    result_bgr = result_rgb[:, :, ::-1]
    out = (result_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    if gate and _gate_output(img_bgr, out) is not None:
        return img_bgr
    return out


def upscale_2x(img_bgr: np.ndarray, cache: ModelCache) -> np.ndarray:
    """2x Gigapixel HQ upscale via ghqv2-v1-fp16-128x128-2x-ov.tz, 128x128 tiles."""
    tz = MODELS_DIR / "ghqv2-v1-fp16-128x128-2x-ov.tz"
    compiled = cache.load(tz)

    # The compiled model's inputs are netInput, pn, pb, pc. If the model does
    # not expose .inputs (a test mock) all four are fed.
    known = _input_names_of(compiled)

    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    _z1 = np.zeros((1,), dtype=np.float32)

    def feed(tile: np.ndarray, mdl) -> np.ndarray:
        feed_dict: dict = {"netInput": tile[np.newaxis]}        # [1,3,128,128]
        # Optional scalar params (zero = neutral)
        for pname in ("pn", "pb", "pc"):
            if not known or pname in known:
                feed_dict[pname] = _z1
        return _single_output(mdl(feed_dict))[0]                # [3,256,256]

    result_chw = _run_tiled(compiled, img_chw, 128, 128, 16, 2, feed)

    result_rgb = np.transpose(result_chw, (1, 2, 0))
    result_bgr = result_rgb[:, :, ::-1]
    return (result_bgr * 255.0).round().clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

INAT_MAX_EDGE = 2048
STOCK_MAX_MP = 100.0
JPEG_QUALITY = 95


def enhance(src: Path, dst: Path, mode: str = "stock",
            cache: ModelCache | None = None,
            min_free_gb: float = 5.0) -> None:
    """Enhance *src* JPEG and write the result to *dst* JPEG (q95).

    Modes
    -----
    stock   : gated sharpen (full-resolution output). When the gate skips,
              the source bytes are copied unchanged (no re-encode).
    inat    : no enhancement. iNat uploads (2048 px) are not sharpened; the
              source is copied unchanged.
    upscale : gated sharpen then 2x upscale. Refused for iNat-sized inputs
              (long edge 1025..2048) and when the result would exceed 100 MP.
    """
    src, dst = Path(src), Path(dst)

    if mode not in ("stock", "inat", "upscale"):
        raise ValueError(f"Unknown mode: {mode!r}")

    # Disk guard
    check_dir = dst.parent if dst.parent.exists() else src.parent
    free_gb = shutil.disk_usage(check_dir).free / 1e9
    if free_gb < min_free_gb:
        raise OSError(
            f"Insufficient disk space: {free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
        )

    def _copy_unchanged() -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)

    if mode == "inat":
        _copy_unchanged()
        return

    own_cache = cache is None
    if own_cache:
        cache = ModelCache()

    try:
        pil_img = Image.open(src)
        # Pillow drops EXIF and the ICC profile on save unless handed back;
        # the camera EXIF and the sRGB profile must survive an enhance pass.
        keep = {k: pil_img.info[k] for k in ("exif", "icc_profile") if pil_img.info.get(k)}
        pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img, dtype=np.uint8)
        img_bgr = img_rgb[:, :, ::-1].copy()

        h, w = img_bgr.shape[:2]
        if mode == "upscale":
            if INAT_MAX_EDGE // 2 < max(h, w) <= INAT_MAX_EDGE:
                raise ValueError(
                    f"upscale refused: {w}x{h} is iNat-sized (long edge <= {INAT_MAX_EDGE})")
            if (h * 2) * (w * 2) / 1e6 > STOCK_MAX_MP:
                raise ValueError(
                    f"upscale refused: result {w * 2}x{h * 2} exceeds {STOCK_MAX_MP:.0f} MP")

        sharpened = sharpen(img_bgr, cache)
        if mode == "stock":
            if sharpened is img_bgr:
                _copy_unchanged()
                return
            result_bgr = sharpened
        else:
            result_bgr = upscale_2x(sharpened, cache)

        result_rgb = result_bgr[:, :, ::-1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result_rgb).save(
            str(dst), "JPEG", quality=JPEG_QUALITY, subsampling=0, **keep)
    finally:
        if own_cache:
            cache.__exit__(None, None, None)
