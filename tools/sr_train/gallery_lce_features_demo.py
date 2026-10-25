"""Real before/after examples of the lce model-stack audit's new features
(rjmendez/lce#3), run through lce's actual shipped code -- not a reimplementation
-- so the gallery shows what the plugin really does, not what the eval harness
predicts. Covers:

  upscale  -- scale=2 default (now the stock LoRA fine-tune) vs the explicit
              "sentinel" variant it replaced as default; scale=4 default
              (unchanged, still sentinel) vs the new explicit "stock"
              zero-shot variant.
  segment  -- the existing default 3x3 grid prompt vs a single caller-supplied
              point near the subject (the new `points` argument).

Unlike the other gallery_*.py scripts in this file, this one does NOT need
the GPU-only bio-sr-train venv or CUDA: lce's upscale/segment code is plain
ONNX Runtime (CPU execution provider here), by design CPU/GPU-agnostic --
run with any Python that has lce installed/importable plus its "dev" extras.

Run: PYTHONPATH=<lce repo>/src python gallery_lce_features_demo.py
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")
GALLERY_OUT = Path(r"D:\bio-results\gallery\_examples\lce-model-stack-demo")
UPSCALE_CROP = 256
SEGMENT_CROP = 512
N_UPSCALE_EXAMPLES = 4
N_SEGMENT_EXAMPLES = 3


def _held_out_files():
    """Same held-out split as the other gallery scripts and train_lora_hat.py's
    own --held-out 80 -- these were never trained on, so examples are a fair
    demo, not cherry-picked train data."""
    all_files = sorted(SOURCE_DIR.glob("*.jpg"))
    rng = random.Random(1234)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    return shuffled[:80]


def _center_crop(path: Path, size: int) -> np.ndarray | None:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w < size or h < size:
            return None
        x0, y0 = (w - size) // 2, (h - size) // 2
        return np.asarray(im.crop((x0, y0, x0 + size, y0 + size)))


def _to_chw_f32(hwc_u8: np.ndarray) -> np.ndarray:
    return hwc_u8.astype(np.float32).transpose(2, 0, 1) / 255.0


def _to_hwc_u8(chw_f32: np.ndarray) -> np.ndarray:
    return (chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)


def _sidebyside(a: np.ndarray, b: np.ndarray) -> Image.Image:
    strip = Image.new("RGB", (a.shape[1] + b.shape[1], max(a.shape[0], b.shape[0])))
    strip.paste(Image.fromarray(a), (0, 0))
    strip.paste(Image.fromarray(b), (a.shape[1], 0))
    return strip


def upscale_demo(cache):
    from lce.models import upscale

    held_out = _held_out_files()
    crops = []
    for p in held_out:
        if len(crops) >= N_UPSCALE_EXAMPLES:
            break
        crop = _center_crop(p, UPSCALE_CROP)
        if crop is not None:
            crops.append((p.stem, crop))
    print(f"[upscale] {len(crops)} held-out {UPSCALE_CROP}px crops (center crop, not cherry-picked)")

    for name, crop_hwc in crops:
        out_dir = GALLERY_OUT / "upscale" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        chw = _to_chw_f32(crop_hwc)

        bicubic_2x = np.asarray(Image.fromarray(crop_hwc).resize(
            (UPSCALE_CROP * 2, UPSCALE_CROP * 2), Image.BICUBIC))
        default_2x = _to_hwc_u8(upscale.run(chw, cache, scale=2))                       # stock (new default)
        sentinel_2x = _to_hwc_u8(upscale.run(chw, cache, scale=2, variant="sentinel"))  # old default, now explicit

        bicubic_4x = np.asarray(Image.fromarray(crop_hwc).resize(
            (UPSCALE_CROP * 4, UPSCALE_CROP * 4), Image.BICUBIC))
        default_4x = _to_hwc_u8(upscale.run(chw, cache, scale=4))                       # sentinel (unchanged default)
        stock_4x = _to_hwc_u8(upscale.run(chw, cache, scale=4, variant="stock"))        # new zero-shot variant

        Image.fromarray(crop_hwc).save(out_dir / f"0_original_{UPSCALE_CROP}px_crop.jpg", quality=95)
        Image.fromarray(bicubic_2x).save(out_dir / "1_bicubic_2x.jpg", quality=95)
        Image.fromarray(default_2x).save(out_dir / "2_upscale_2x_default_stock_lora.jpg", quality=95)
        Image.fromarray(sentinel_2x).save(out_dir / "3_upscale_2x_sentinel_variant.jpg", quality=95)
        Image.fromarray(bicubic_4x).save(out_dir / "4_bicubic_4x.jpg", quality=95)
        Image.fromarray(default_4x).save(out_dir / "5_upscale_4x_default_sentinel.jpg", quality=95)
        Image.fromarray(stock_4x).save(out_dir / "6_upscale_4x_stock_zeroshot_variant.jpg", quality=95)
        _sidebyside(sentinel_2x, default_2x).save(
            out_dir / "7_sentinel_vs_default_2x_sidebyside.jpg", quality=92)
        _sidebyside(default_4x, stock_4x).save(
            out_dir / "8_default_vs_stock_4x_sidebyside.jpg", quality=92)

        print(f"  {name} -> {out_dir}")


def _mask_overlay(hwc_u8: np.ndarray, mask_1hw: np.ndarray) -> np.ndarray:
    """Red tint over the selected region, 50% alpha, for a quick visual check."""
    overlay = hwc_u8.copy().astype(np.float32)
    sel = mask_1hw[0] > 0.5
    overlay[sel] = overlay[sel] * 0.5 + np.array([255, 40, 40], dtype=np.float32) * 0.5
    return overlay.round().clip(0, 255).astype(np.uint8)


def segment_demo(cache):
    from lce.models import segment

    held_out = _held_out_files()[N_UPSCALE_EXAMPLES:]  # don't reuse the upscale crops
    crops = []
    for p in held_out:
        if len(crops) >= N_SEGMENT_EXAMPLES:
            break
        crop = _center_crop(p, SEGMENT_CROP)
        if crop is not None:
            crops.append((p.stem, crop))
    print(f"[segment] {len(crops)} held-out {SEGMENT_CROP}px crops")

    for name, crop_hwc in crops:
        out_dir = GALLERY_OUT / "segment" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        chw = _to_chw_f32(crop_hwc)

        mask_grid = segment.run(chw, cache)                          # old/default: fixed 3x3 grid
        mask_point = segment.run(chw, cache, points=[(0.5, 0.5)])    # new: single point near the subject

        Image.fromarray(crop_hwc).save(out_dir / f"0_original_{SEGMENT_CROP}px_crop.jpg", quality=95)
        Image.fromarray((mask_grid[0] * 255).astype(np.uint8)).save(out_dir / "1_mask_default_3x3_grid.png")
        Image.fromarray((mask_point[0] * 255).astype(np.uint8)).save(out_dir / "2_mask_custom_center_point.png")
        overlay_grid = _mask_overlay(crop_hwc, mask_grid)
        overlay_point = _mask_overlay(crop_hwc, mask_point)
        Image.fromarray(overlay_grid).save(out_dir / "3_overlay_default_3x3_grid.jpg", quality=92)
        Image.fromarray(overlay_point).save(out_dir / "4_overlay_custom_center_point.jpg", quality=92)
        _sidebyside(overlay_grid, overlay_point).save(
            out_dir / "5_default_vs_custom_point_sidebyside.jpg", quality=92)

        grid_frac = float(mask_grid.mean())
        point_frac = float(mask_point.mean())
        print(f"  {name} -> {out_dir}  (grid selects {grid_frac:.1%} of crop, point selects {point_frac:.1%})")


def main():
    from lce.engine import ModelCache

    cache = ModelCache(max_models=4, device="CPU")
    GALLERY_OUT.mkdir(parents=True, exist_ok=True)

    upscale_demo(cache)
    segment_demo(cache)

    print(f"\nWrote examples to {GALLERY_OUT}")


if __name__ == "__main__":
    main()
