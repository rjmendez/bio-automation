"""M3a: zero-shot benchmark the REAL official HAT-S (XPixelGroup/HAT, Apache-2.0,
HAT-S_SRx2.pth) against plain bicubic on the SAME held-out real crops
validate_real.py uses -- not lce's unrelated satellite-tuned derivative.

Two comparisons:
  1. Synthetic-bicubic-degradation PSNR/LPIPS/DISTS (apples-to-apples with the
     from-scratch TinySR2x v2 test, since HAT-S's own pretraining degradation
     IS plain bicubic -- this is literally the benchmark it was trained for).
  2. Deployment sanity check: genuinely native, undegraded crops fed directly
     (no ground truth exists at 2x for these, so Laplacian-variance only, same
     proxy as validate_real.py's deployment_sharpness_check).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "hat_vendor"))
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402
from train import pick_device, PATCH_SIZE  # noqa: E402
from validate_real import psnr, perceptual_scores, _laplacian_var  # noqa: E402
from hat_arch import HAT  # noqa: E402


def load_hat_s_x2(weights_path: Path, device) -> HAT:
    model = HAT(
        upscale=2, in_chans=3, img_size=64, window_size=16,
        compress_ratio=24, squeeze_factor=24, conv_scale=0.01, overlap_ratio=0.5,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    )
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["params_ema"] if "params_ema" in ckpt else ckpt["params"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--held-out", type=int, default=40)
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = pick_device(args.cuda_name_substr)

    all_files = sorted(args.source_dir.glob("*.jpg"))
    shuffled = all_files[:]
    random.Random(1234).shuffle(shuffled)  # same split convention as validate_real.py
    held_out_files = shuffled[: args.held_out]
    print(f"held-out: {len(held_out_files)} files  device: {device}")

    model = load_hat_s_x2(args.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HAT-S params: {n_params:,}")

    # --- 1. synthetic-bicubic-degradation benchmark (apples-to-apples with TinySR2x v2) ---
    model_psnrs, bicubic_psnrs = [], []
    model_preds, bicubics, hrs = [], [], []
    with torch.no_grad():
        for p in held_out_files:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                hr = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
            lr = degrade_hr_to_lr(hr, rng)  # current degrade.py: fixed mild blur + bicubic downsample
            lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(lr_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(Image.fromarray(lr).resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC))
            model_psnrs.append(psnr(pred_u8, hr))
            bicubic_psnrs.append(psnr(bicubic, hr))
            model_preds.append(pred_u8)
            bicubics.append(bicubic)
            hrs.append(hr)

    model_avg, bicubic_avg = sum(model_psnrs) / len(model_psnrs), sum(bicubic_psnrs) / len(bicubic_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, bicubic_psnrs))
    model_lpips, model_dists = perceptual_scores(model_preds, hrs, device)
    bicubic_lpips, bicubic_dists = perceptual_scores(bicubics, hrs, device)

    print(f"\n=== synthetic-degradation benchmark ({len(model_psnrs)} images) ===")
    print(f"  HAT-S PSNR:    {model_avg:.2f} dB")
    print(f"  bicubic PSNR:  {bicubic_avg:.2f} dB")
    print(f"  delta:         {model_avg - bicubic_avg:+.2f} dB")
    print(f"  HAT-S wins {wins}/{len(model_psnrs)}")
    print(f"  HAT-S  LPIPS {model_lpips:.4f}  DISTS {model_dists:.4f}")
    print(f"  bicubic LPIPS {bicubic_lpips:.4f}  DISTS {bicubic_dists:.4f}")

    # --- 2. deployment sanity check: genuinely native, undegraded crops ---
    model_vars, bicubic_vars = [], []
    with torch.no_grad():
        for p in held_out_files[:40]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                native = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
            inp_t = torch.from_numpy(native.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(Image.fromarray(native).resize((PATCH_SIZE * 2, PATCH_SIZE * 2), Image.BICUBIC))
            model_vars.append(_laplacian_var(np.asarray(Image.fromarray(pred_u8).convert("L"))))
            bicubic_vars.append(_laplacian_var(np.asarray(Image.fromarray(bicubic).convert("L"))))

    model_sharp, bicubic_sharp = sum(model_vars) / len(model_vars), sum(bicubic_vars) / len(bicubic_vars)
    print(f"\n=== deployment sanity check (real undegraded crops, {len(model_vars)} images) ===")
    print(f"  HAT-S Laplacian var:   {model_sharp:.2f}")
    print(f"  bicubic Laplacian var: {bicubic_sharp:.2f}")

    print("\n=== M3a VERDICT ===")
    synthetic_pass = model_avg > bicubic_avg and model_lpips < bicubic_lpips and model_dists < bicubic_dists
    deploy_pass = model_sharp > bicubic_sharp
    print(f"  synthetic-benchmark pass: {synthetic_pass}")
    print(f"  deployment-sanity pass:   {deploy_pass}")
    if synthetic_pass and deploy_pass:
        print("  PROCEED to M3b (LoRA fine-tune) -- HAT-S zero-shot beats bicubic on both checks.")
    else:
        print("  DO NOT PROCEED to M3b yet -- HAT-S zero-shot does not clear bicubic on this dataset.")


if __name__ == "__main__":
    main()
