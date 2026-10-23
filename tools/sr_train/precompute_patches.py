"""Pre-extract small HR patches once, locally (32 cores here vs 4 vCPU on the
training instance), instead of fully decoding a 24-100 MP Stock JPEG on every
single training sample draw -- which is what stalled the first AWS run: 3
workers pegged at ~98% CPU for 30 minutes produced zero completed steps because
each sample threw away 99.9%+ of a full decode to keep one 256x256 crop.

Output is one memory-mappable .npy array [N, 256, 256, 3] uint8 that a Dataset
can index with zero decode cost. Random crop diversity per source image comes
from extracting several crops per image; LR diversity per patch still comes
from degrade.py's randomized degradation applied fresh every time a patch is
drawn during training, not from this step.
"""
from __future__ import annotations

import argparse
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import PATCH_SIZE  # noqa: E402
from degrade import random_hr_crop  # noqa: E402


def _extract(args: tuple[Path, int, int]) -> np.ndarray | None:
    path, patches_per_image, seed = args
    rng = random.Random(seed)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                return None
            arr = np.asarray(im)
            return np.stack([random_hr_crop(arr, PATCH_SIZE, rng) for _ in range(patches_per_image)])
    except Exception as exc:  # a handful of corrupt/truncated source files must not kill the whole run
        print(f"skip {path.name}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-list", type=Path, required=True,
                     help="text file, one source image path per line (the train-pool split)")
    ap.add_argument("--patches-per-image", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [Path(line.strip()) for line in args.files_list.read_text().splitlines() if line.strip()]
    print(f"{len(paths)} source images, {args.patches_per_image} patches each, {args.processes} processes")

    tasks = [(p, args.patches_per_image, args.seed + i) for i, p in enumerate(paths)]
    chunks: list[np.ndarray] = []
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap_unordered(_extract, tasks, chunksize=4)):
            if result is not None:
                chunks.append(result)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)} images processed, {sum(len(c) for c in chunks)} patches so far")

    patches = np.concatenate(chunks, axis=0)
    print(f"total patches: {patches.shape}  ({patches.nbytes / 1e9:.2f} GB)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, patches)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
