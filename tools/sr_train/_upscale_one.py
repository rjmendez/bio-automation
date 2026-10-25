"""Worker: run exactly one lce upscale call in a fresh process and save the
result. Isolation per call avoids accumulating ONNX Runtime arena memory
across multiple large dynamic-shape models in one process (measured: running
hat-stock-lora-x2 then hat-stock-x4 back-to-back in one ModelCache OOM'd on a
905MB allocation; each alone succeeds in ~450-500s on a ~2300x2300 input).

Usage: _upscale_one.py <in_image> <out_image> <scale> [variant]
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

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
    out_hwc = (out_chw.transpose(1, 2, 0) * 255.0).clip(0, 255).round().astype(np.uint8)
    Image.fromarray(out_hwc).save(out_path, quality=95)
    print(f"scale={scale} variant={variant}: {time.time() - t0:.1f}s -> "
          f"{out_hwc.shape[1]}x{out_hwc.shape[0]} -> {out_path}")


if __name__ == "__main__":
    main()
