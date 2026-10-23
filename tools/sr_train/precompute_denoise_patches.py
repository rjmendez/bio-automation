"""Pre-extract matching HR patch pairs from the owner's own raw pairs: a plain
NEF decode (input) and its Lightroom "Enhanced-NR" DNG (AI denoise, target).
These are REAL pairs, not synthetic degradation -- unlike the crop-upscale
model's first (buggy) version, there is no train/deploy domain mismatch here:
production always starts from a plain decode too.

Alignment is pixel-exact (verified by hand: zero-shift MSE is already minimal,
no shift scores lower -- both decodes come from the identical Bayer sensor
data, just different demosaic/denoise post-processing), so the same crop
offset on both images gives a genuinely matched pair.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degrade import random_hr_crop  # noqa: E402

PATCH_SIZE = 256


def _decode(path: str) -> np.ndarray:
    with rawpy.imread(path) as r:
        return r.postprocess(output_color=rawpy.ColorSpace.sRGB)


def _extract_pair(args: tuple[str, str, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    plain_path, enhanced_path, patches_per_pair, seed = args
    rng = random.Random(seed)
    try:
        plain = _decode(plain_path)
        enhanced = _decode(enhanced_path)
        if plain.shape != enhanced.shape:
            print(f"skip (shape mismatch {plain.shape} vs {enhanced.shape}): {plain_path}", file=sys.stderr)
            return None
        h, w = plain.shape[:2]
        if h < PATCH_SIZE or w < PATCH_SIZE:
            return None
        plain_crops, enh_crops = [], []
        for _ in range(patches_per_pair):
            y0 = rng.randint(0, h - PATCH_SIZE)
            x0 = rng.randint(0, w - PATCH_SIZE)
            plain_crops.append(plain[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
            enh_crops.append(enhanced[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE])
        return np.stack(plain_crops), np.stack(enh_crops)
    except Exception as exc:
        print(f"skip {plain_path}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="pair_manifest.py output (kind,plain_raw,enhanced_target)")
    ap.add_argument("--held-out", type=int, default=40)
    ap.add_argument("--patches-per-pair", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--processes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.manifest, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] == "raw_raw"]
    print(f"{len(rows)} raw/raw pairs in manifest")

    shuffled = rows[:]
    random.Random(1234).shuffle(shuffled)  # same fixed split seed convention as the crop-upscale track
    held_out_rows = shuffled[: args.held_out]
    train_rows = shuffled[args.held_out:]
    print(f"train pool: {len(train_rows)}  held-out: {len(held_out_rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "held_out_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["plain_raw", "enhanced_target"])
        for r in held_out_rows:
            w.writerow([r["plain_raw"], r["enhanced_target"]])

    tasks = [(r["plain_raw"], r["enhanced_target"], args.patches_per_pair, args.seed + i)
              for i, r in enumerate(train_rows)]
    plain_chunks, enh_chunks = [], []
    with Pool(args.processes) as pool:
        for i, result in enumerate(pool.imap_unordered(_extract_pair, tasks, chunksize=2)):
            if result is not None:
                plain_chunks.append(result[0])
                enh_chunks.append(result[1])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} pairs processed, {sum(len(c) for c in plain_chunks)} patch-pairs so far")

    plain_patches = np.concatenate(plain_chunks, axis=0)
    enh_patches = np.concatenate(enh_chunks, axis=0)
    print(f"total patch-pairs: {plain_patches.shape}  ({plain_patches.nbytes / 1e9:.2f} GB each array)")
    np.save(args.out_dir / "patches_plain.npy", plain_patches)
    np.save(args.out_dir / "patches_enhanced.npy", enh_patches)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
