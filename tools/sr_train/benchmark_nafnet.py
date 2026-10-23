"""M3a (denoise track): zero-shot benchmark the REAL official NAFNet-SIDD-width64
(megvii-research/NAFNet, MIT license, confirmed) against the plain-decode baseline
on the SAME held-out real raw pairs train_denoise.py uses -- the parallel check to
benchmark_hat_s.py, run because TinyRestore1x (from-scratch, pure L1) failed its
own gate decisively (-4.86 dB vs the plain decode) on this data.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nafnet_vendor"))
from train import pick_device  # noqa: E402
from validate_real import psnr, perceptual_scores  # noqa: E402
from precompute_denoise_patches import PATCH_SIZE, _decode  # noqa: E402
from degrade import random_hr_crop  # noqa: E402
from NAFNet_arch import NAFNet  # noqa: E402


def load_nafnet_sidd_width64(weights_path: Path, device) -> NAFNet:
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("params", ckpt.get("params_ema", ckpt))
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out-csv", type=Path, required=True, help="precompute_denoise_patches.py's held_out_pairs.csv")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--held-out", type=int, default=40)
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = pick_device(args.cuda_name_substr)

    model = load_nafnet_sidd_width64(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet-SIDD-width64 params: {n_params:,}  device: {device}")

    with open(args.held_out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))[: args.held_out]

    model_psnrs, plain_psnrs = [], []
    model_preds, plains, targets = [], [], []
    zero_mse_skipped = 0
    with torch.no_grad():
        for row in rows:
            try:
                plain = _decode(row["plain_raw"])
                enhanced = _decode(row["enhanced_target"])
            except Exception as exc:
                print(f"skip {row['plain_raw']}: {exc}", file=sys.stderr)
                continue
            if plain.shape != enhanced.shape or plain.shape[0] < PATCH_SIZE or plain.shape[1] < PATCH_SIZE:
                continue
            h, w = plain.shape[:2]
            y0 = rng.randint(0, h - PATCH_SIZE)
            x0 = rng.randint(0, w - PATCH_SIZE)
            plain_crop = plain[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            target_crop = enhanced[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]

            # Same pre-existing bug train_denoise.py found and documented: some
            # crops are bit-identical between plain and target (no PSNR signal,
            # would report +inf and poison the average) -- skip them here too.
            if np.array_equal(plain_crop, target_crop):
                zero_mse_skipped += 1
                continue

            inp = torch.from_numpy(plain_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)

            model_psnrs.append(psnr(pred_u8, target_crop))
            plain_psnrs.append(psnr(plain_crop, target_crop))
            model_preds.append(pred_u8)
            plains.append(plain_crop)
            targets.append(target_crop)

    if zero_mse_skipped:
        print(f"skipped {zero_mse_skipped}/{len(rows)} bit-identical pairs")

    model_avg, plain_avg = sum(model_psnrs) / len(model_psnrs), sum(plain_psnrs) / len(plain_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, plain_psnrs))
    model_lpips, model_dists = perceptual_scores(model_preds, targets, device)
    plain_lpips, plain_dists = perceptual_scores(plains, targets, device)

    print(f"\n=== NAFNet zero-shot vs plain decode ({len(model_psnrs)} held-out real pairs) ===")
    print(f"  NAFNet PSNR:  {model_avg:.2f} dB")
    print(f"  plain  PSNR:  {plain_avg:.2f} dB")
    print(f"  delta:        {model_avg - plain_avg:+.2f} dB")
    print(f"  NAFNet wins {wins}/{len(model_psnrs)}")
    print(f"  NAFNet LPIPS {model_lpips:.4f}  DISTS {model_dists:.4f}")
    print(f"  plain  LPIPS {plain_lpips:.4f}  DISTS {plain_dists:.4f}")

    print("\n=== M3a VERDICT (denoise track) ===")
    passed = model_avg > plain_avg and model_lpips < plain_lpips and model_dists < plain_dists
    print(f"  zero-shot NAFNet beats plain decode on PSNR+LPIPS+DISTS together: {passed}")
    if passed:
        print("  PROCEED to a LoRA/fine-tune of NAFNet -- zero-shot already clears the baseline.")
    else:
        print("  DO NOT PROCEED yet -- zero-shot NAFNet does not clear the plain-decode baseline on this data.")


if __name__ == "__main__":
    main()
