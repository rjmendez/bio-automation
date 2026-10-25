"""Full-size (not cropped) before/after examples of lce's new upscale/segment
features (rjmendez/lce#3), run through lce's actual ONNX inference code, same
convention as gallery_fullsize_examples.py and the crop-level
gallery_lce_features_demo.py.

Upscale is deliberately limited to the "stock" models (hat-stock-lora-x2,
hat-stock-x4 -- dynamic input shape, tiled at lce's default 256px) and skips
the "sentinel" models at full size: hat-sentinel-x2/x4 take a FIXED 64x64
input (see lce.engine.static_hw), so a real multi-megapixel photo needs
thousands of tiles at that size -- impractical on CPU for a demo. The
sentinel-vs-stock comparison is already covered at crop scale by
gallery_lce_features_demo.py; this script's job is just to prove the new
stock defaults/variants hold up on a REAL full-resolution photo, not a
synthetic crop. Even at 256px tiles this is slow (measured: 2x took 505s,
4x took 448s, on a 2375x2288 input -- ~80 tiles), so N=1.

Each upscale call runs in its own subprocess (_upscale_one.py) -- running
hat-stock-lora-x2 then hat-stock-x4 back-to-back in ONE process (one
ModelCache) OOM'd (905MB allocation failure, ONNX Runtime arena memory
accumulating across two large dynamic-shape sessions); each alone, in a
fresh process, succeeded. This is a real, measured constraint, not a
hypothetical one -- confirmed by first reproducing the OOM, then confirming
the isolated retry worked.

segment is cheap regardless of input size (MobileSAM always resizes to a
1024px long side internally), so it runs at full size on more examples.

Run: PYTHONPATH=<lce repo>/src python gallery_lce_features_fullsize_demo.py
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")
GALLERY_OUT = Path(r"D:\bio-results\gallery\_examples\lce-model-stack-fullsize-demo")
N_UPSCALE_EXAMPLES = 2
N_SEGMENT_EXAMPLES = 2


def _held_out_files():
    all_files = sorted(SOURCE_DIR.glob("*.jpg"))
    rng = random.Random(1234)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    return shuffled[:80]


def _to_chw_f32(hwc_u8: np.ndarray) -> np.ndarray:
    return hwc_u8.astype(np.float32).transpose(2, 0, 1) / 255.0


def _to_hwc_u8(chw_f32: np.ndarray) -> np.ndarray:
    return (chw_f32.transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)


_WORKER = Path(__file__).resolve().parent / "_upscale_one.py"


def _upscale_in_subprocess(in_path: Path, out_path: Path, scale: int, variant: str | None) -> None:
    """Each call is a fresh process -- see _upscale_one.py's docstring for why:
    two large dynamic-shape ONNX models back-to-back in one process OOM'd,
    each alone did not."""
    import subprocess

    args = [sys.executable, str(_WORKER), str(in_path), str(out_path), str(scale)]
    if variant:
        args.append(variant)
    subprocess.run(args, check=True)


def upscale_demo():
    held_out = _held_out_files()
    picked = held_out[:N_UPSCALE_EXAMPLES]

    for p in picked:
        out_dir = GALLERY_OUT / "upscale" / p.stem
        if (out_dir / "4_bicubic_4x.jpg").exists():
            print(f"[upscale] {p.stem} already done, skipping")
            continue

        with Image.open(p) as im:
            hwc = np.asarray(im.convert("RGB"))
        h, w = hwc.shape[:2]
        print(f"[upscale] {p.stem} ({w}x{h})")
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(hwc).save(out_dir / f"0_original_{w}x{h}.jpg", quality=95)

        _upscale_in_subprocess(p, out_dir / "1_upscale_2x_default_stock_lora.jpg", 2, None)
        bicubic_2x = np.asarray(Image.fromarray(hwc).resize((w * 2, h * 2), Image.BICUBIC))
        Image.fromarray(bicubic_2x).save(out_dir / "2_bicubic_2x.jpg", quality=95)

        _upscale_in_subprocess(p, out_dir / "3_upscale_4x_stock_zeroshot_variant.jpg", 4, "stock")
        bicubic_4x = np.asarray(Image.fromarray(hwc).resize((w * 4, h * 4), Image.BICUBIC))
        Image.fromarray(bicubic_4x).save(out_dir / "4_bicubic_4x.jpg", quality=95)

        print(f"  {p.stem} -> {out_dir}")


def _mask_overlay(hwc_u8: np.ndarray, mask_1hw: np.ndarray) -> np.ndarray:
    overlay = hwc_u8.copy().astype(np.float32)
    sel = mask_1hw[0] > 0.5
    overlay[sel] = overlay[sel] * 0.5 + np.array([255, 40, 40], dtype=np.float32) * 0.5
    return overlay.round().clip(0, 255).astype(np.uint8)


_UPSCALE_RESERVED_SLOTS = 4  # fixed, independent of N_UPSCALE_EXAMPLES, so growing the
# upscale example count doesn't shift which held-out files segment_demo picks and
# orphan/rename its already-generated output.


def segment_demo(cache):
    from lce.models import segment

    held_out = _held_out_files()[_UPSCALE_RESERVED_SLOTS:]
    picked = held_out[:N_SEGMENT_EXAMPLES]

    for p in picked:
        out_dir = GALLERY_OUT / "segment" / p.stem
        if (out_dir / "2_overlay_custom_center_point.jpg").exists():
            print(f"[segment] {p.stem} already done, skipping")
            continue

        with Image.open(p) as im:
            hwc = np.asarray(im.convert("RGB"))
        h, w = hwc.shape[:2]
        print(f"[segment] {p.stem} ({w}x{h})")
        out_dir.mkdir(parents=True, exist_ok=True)
        chw = _to_chw_f32(hwc)

        mask_grid = segment.run(chw, cache)
        mask_point = segment.run(chw, cache, points=[(0.5, 0.5)])

        Image.fromarray(hwc).save(out_dir / f"0_original_{w}x{h}.jpg", quality=95)
        overlay_grid = _mask_overlay(hwc, mask_grid)
        overlay_point = _mask_overlay(hwc, mask_point)
        Image.fromarray(overlay_grid).save(out_dir / "1_overlay_default_3x3_grid.jpg", quality=92)
        Image.fromarray(overlay_point).save(out_dir / "2_overlay_custom_center_point.jpg", quality=92)

        grid_frac = float(mask_grid.mean())
        point_frac = float(mask_point.mean())
        print(f"  {p.stem} -> {out_dir}  (grid selects {grid_frac:.1%}, point selects {point_frac:.1%})")


def main():
    from lce.engine import ModelCache

    GALLERY_OUT.mkdir(parents=True, exist_ok=True)

    upscale_demo()  # each call isolated in its own subprocess, see _upscale_one.py
    segment_demo(ModelCache(max_models=4, device="CPU"))  # cheap, one process is fine

    print(f"\nWrote full-size examples to {GALLERY_OUT}")


if __name__ == "__main__":
    main()
