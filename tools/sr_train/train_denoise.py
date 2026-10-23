"""Local real-data gate for the denoise/detail track, same pattern as
validate_real.py for the crop-upscale track: train briefly on the real raw
pairs, then measure on held-out pairs never touched in training.

The baseline here is different from the crop-upscale track's bicubic: there IS
no "do nothing" baseline that makes sense except literally doing nothing --
PSNR(plain, enhanced_target) measures how far the plain decode already is
from the owner's own Lightroom result. The gate is: does the model get closer
to that result than the plain decode already is? If not, it isn't learning
anything useful about the owner's own denoise/detail preference.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinyRestore1x, count_params  # noqa: E402
from train import pick_device  # noqa: E402
from patch_pair_dataset import make_pair_loader  # noqa: E402
from validate_real import psnr  # noqa: E402
from precompute_denoise_patches import PATCH_SIZE, _decode  # noqa: E402
from degrade import random_hr_crop  # noqa: E402


def held_out_eval(model, held_out_csv: Path, device, rng, n: int = 40):
    """model(plain decode) vs plain decode itself, scored against the real
    Lightroom-enhanced target -- on raw pairs never used in training."""
    with open(held_out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))[:n]

    model_psnrs, plain_psnrs = [], []
    with torch.no_grad():
        for row in rows:
            try:
                plain = _decode(row["plain_raw"])
                enhanced = _decode(row["enhanced_target"])
            except Exception as exc:
                print(f"skip held-out {row['plain_raw']}: {exc}", file=sys.stderr)
                continue
            if plain.shape != enhanced.shape or plain.shape[0] < PATCH_SIZE or plain.shape[1] < PATCH_SIZE:
                continue
            h, w = plain.shape[:2]
            y0 = rng.randint(0, h - PATCH_SIZE)
            x0 = rng.randint(0, w - PATCH_SIZE)
            plain_crop = plain[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            target_crop = enhanced[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]

            inp = torch.from_numpy(plain_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)

            model_psnrs.append(psnr(pred_u8, target_crop))
            plain_psnrs.append(psnr(plain_crop, target_crop))
    return model_psnrs, plain_psnrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", type=Path, required=True, help="precompute_denoise_patches.py --out-dir")
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cuda_name_substr)

    model = TinyRestore1x().to(device)
    print(f"TinyRestore1x params: {count_params(model):,}  device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    loader = make_pair_loader(
        args.patch_dir / "patches_plain.npy", args.patch_dir / "patches_enhanced.npy",
        args.batch_size, args.workers, seed=args.seed, nominal_len=(args.train_steps + 2) * args.batch_size)

    t0 = time.time()
    losses = []
    it = iter(loader)
    for step in range(args.train_steps):
        plain_t, enh_t = next(it)
        plain_t, enh_t = plain_t.to(device), enh_t.to(device)
        pred = model(plain_t)
        loss = F.l1_loss(pred, enh_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps - 1:
            print(f"step {step:5d}  l1 {loss.item():.4f}  elapsed {time.time() - t0:.1f}s")

    first_avg = sum(losses[:10]) / min(10, len(losses))
    last_avg = sum(losses[-10:]) / min(10, len(losses))
    print(f"train loss: first-10 avg {first_avg:.4f} -> last-10 avg {last_avg:.4f}")

    model.eval()
    model_psnrs, plain_psnrs = held_out_eval(model, args.patch_dir / "held_out_pairs.csv", device, rng)
    model_avg = sum(model_psnrs) / len(model_psnrs)
    plain_avg = sum(plain_psnrs) / len(plain_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, plain_psnrs))
    print(f"\nheld-out ({len(model_psnrs)} raw pairs, never seen in training):")
    print(f"  model PSNR (vs Lightroom target):  {model_avg:.2f} dB")
    print(f"  plain decode PSNR (vs same target): {plain_avg:.2f} dB")
    print(f"  delta: {model_avg - plain_avg:+.2f} dB")
    print(f"  model closer to target on {wins}/{len(model_psnrs)} pairs")

    if model_avg <= plain_avg:
        print("\nGATE FAILED: model does not get closer to the owner's own Lightroom result "
              "than the plain decode already is. Do not proceed to AWS.")
        raise SystemExit(1)
    print("\nGATE PASSED.")

    if args.out:
        torch.save(model.state_dict(), args.out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
