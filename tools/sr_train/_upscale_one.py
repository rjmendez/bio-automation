"""Worker: run exactly one lce upscale call in a fresh process and save the
result. Isolation per call avoids accumulating ONNX Runtime arena/VRAM
across multiple large dynamic-shape models in one process (measured on CPU:
running hat-stock-lora-x2 then hat-stock-x4 back-to-back in one ModelCache
OOM'd on a 905MB allocation; each alone succeeds).

GPU: measured unreliable/crash-prone for lce's ONNX models on this machine
across 3 real runs (see _lce_gpu's docstring) -- _lce_gpu.force_cpu() makes
this worker take lce's own CPU-only path deliberately, not by accident.

Usage: _upscale_one.py <in_image> <out_image> <scale> [variant]
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lce_gpu  # noqa: E402
_lce_gpu.force_cpu()

sys.path.insert(0, str(Path(r"C:\Users\rjmendez-admin\development\lce\src")))
from lce.engine import ModelCache  # noqa: E402
from lce.models import upscale  # noqa: E402


def main():
    in_path, out_path, scale = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else None

    with Image.open(in_path) as im:
        hwc = np.asarray(im.convert("RGB"))
    chw = hwc.astype(np.float32).transpose(2, 0, 1) / 255.0

    cache = ModelCache(max_models=1, device="CPU")
    t0 = time.time()
    out_chw = upscale.run(chw, cache, scale=scale, variant=variant)
    elapsed = time.time() - t0
    providers = next(iter(cache.providers.values()), ["?"])
    out_hwc = (out_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
    Image.fromarray(out_hwc).save(out_path, quality=95)
    print(f"scale={scale} variant={variant} providers={providers}: {elapsed:.1f}s -> "
          f"{out_hwc.shape[1]}x{out_hwc.shape[0]} -> {out_path}")


if __name__ == "__main__":
    main()
