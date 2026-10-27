"""Upscale the Stock backlog's actual crops (sub-region selections smaller
than full sensor resolution) via lce's HAT models, scaled per image so the
result never exceeds Stock's 100MP ceiling -- a flat scale=4 (16x the pixel
count) overflows it for 84% of the crops actually in this library.

Full-frame exports (matching a known native sensor size) are skipped
entirely -- they are already at or above Stock's required resolution; see
FULL_FRAME_SIZES. Measured 2026-10-27 via a full scan of ready/ (9,802
files): (8256,5504) x5109, (5504,8256) x1964, (7360,4912) x23, no
(4912,7360) seen -- kept anyway as the expected portrait counterpart.

Never writes in place: ready/ is the only copy of these originals, no
backup. Every result goes to *out_dir* under the same filename; resumable
(a file already in out_dir is skipped).

scale=4's default upscale model (lce.models.upscale) is hat-sentinel-x4, a
Sentinel-2 satellite-imagery fine-tune with no real-photo validation --
variant="stock" is forced instead, the zero-shot HAT-S checkpoint actually
measured (vs bicubic) on real photos. scale=2 already defaults to the
real-photo-tuned local checkpoint, so no override is needed there.

CUDA DLL path setup and the megapixel-ceiling-aware scale choice both live
in lce itself (lce.cuda_dlls, lce.models.upscale.recommended_scale) -- they
are generic to any onnxruntime-CUDA/HAT-upscale consumer on this machine,
not specific to this pipeline, so they are not duplicated here.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from lce import cuda_dlls
from PIL import Image

from . import disk_guard, settings
from .topaz import STOCK_MAX_MP

READY_DIR = settings.STOCK_DIR / "ready"
DEFAULT_OUT_DIR = settings.PHOTOS_CACHE / "crop-upscaled"

FULL_FRAME_SIZES = {(8256, 5504), (5504, 8256), (7360, 4912), (4912, 7360)}

_cuda_ready = False


def _ensure_cuda_dlls() -> None:
    global _cuda_ready
    if _cuda_ready:
        return
    missing = cuda_dlls.add_cuda_dll_dirs(settings.CUDA_DLL_DIRS)
    if missing:
        raise RuntimeError(f"crop_upscale: CUDA DLL folders not found: {[str(d) for d in missing]} "
                            f"(set BIO_CUDA_DLL_DIRS)")
    _cuda_ready = True


def pick_scale(w: int, h: int) -> int | None:
    """4, 2, or None (skip -- already close enough to the ceiling that even
    2x would exceed it): lce.models.upscale.recommended_scale against
    STOCK_MAX_MP."""
    from lce.models.upscale import recommended_scale
    return recommended_scale(w, h, max_mp=STOCK_MAX_MP)


def upscale_one(src: Path, dst: Path, scale: int, cache) -> None:
    """Upscale *src* by *scale* (2 or 4, variant="stock" at either) and write
    to *dst*, preserving EXIF/ICC."""
    from lce.models import upscale as lce_upscale

    pil_img = Image.open(src)
    keep = {k: pil_img.info[k] for k in ("exif", "icc_profile") if pil_img.info.get(k)}
    pil_img = pil_img.convert("RGB")
    rgb = np.array(pil_img, dtype=np.uint8)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

    out_chw = lce_upscale.run(chw, cache, scale=scale, variant="stock")
    out_chw = np.clip(out_chw, 0.0, 1.0)
    out_hwc = np.transpose(out_chw, (1, 2, 0))
    out_u8 = (out_hwc * 255.0 + 0.5).astype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_u8).save(str(dst), "JPEG", quality=95, subsampling=0, **keep)


def run(
    src_dir: Path = READY_DIR,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_free_gb: float = 10.0,
    limit: int | None = None,
    cache=None,
    log: Callable[[str], None] = print,
) -> dict:
    """Upscale every not-yet-processed crop in *src_dir* into *out_dir*.

    Returns {"done": int, "skipped_existing": int, "skipped_full_frame": int,
    "skipped_too_large": int, "failed": int, "total": int}.
    """
    out_dir = Path(out_dir)
    files = sorted(Path(src_dir).glob("*.jpg"))
    if limit is not None:
        files = files[:limit]

    done = skipped_existing = skipped_full_frame = skipped_too_large = failed = 0

    def _process(active_cache) -> None:
        nonlocal done, skipped_existing, skipped_full_frame, skipped_too_large, failed
        for jpg in files:
            dst = out_dir / jpg.name
            if dst.exists():
                skipped_existing += 1
                continue

            with Image.open(jpg) as im:
                w, h = im.size
            if (w, h) in FULL_FRAME_SIZES:
                skipped_full_frame += 1
                continue

            scale = pick_scale(w, h)
            if scale is None:
                skipped_too_large += 1
                continue

            if not disk_guard.wait_for_output(out_dir, min_free_gb, who="crop_upscale", say=log):
                free = shutil.disk_usage(out_dir.parent if not out_dir.exists() else out_dir).free / 1e9
                log(f"STOP: only {free:.0f} GB free (limit {min_free_gb:.0f} GB)")
                return

            try:
                _ensure_cuda_dlls()
                upscale_one(jpg, dst, scale, active_cache)
                done += 1
            except Exception as e:
                failed += 1
                log(f"ERROR {jpg.name} (scale={scale}): {type(e).__name__} {str(e)[:120]}")
                if dst.exists():
                    dst.unlink()

    if cache is None:
        from lce.engine import ModelCache as LceCache
        with LceCache() as owned_cache:
            _process(owned_cache)
    else:
        _process(cache)

    return {"done": done, "skipped_existing": skipped_existing, "skipped_full_frame": skipped_full_frame,
            "skipped_too_large": skipped_too_large, "failed": failed, "total": len(files)}
