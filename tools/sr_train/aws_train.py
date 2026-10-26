"""The real AWS training run: parallel-loaded real data, a held-out PSNR gate
(same bar validate_real.py already cleared locally: beat plain bicubic), then
save the checkpoint. Designed to run once on a g4dn.xlarge (T4, 4 vCPU) under
systemd/user-data and exit -- the instance terminates itself right after,
success or failure, so nothing can run away and keep billing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinySR2x, count_params  # noqa: E402
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402
from train import pick_device, PATCH_SIZE  # noqa: E402
from patch_dataset import make_patch_loader  # noqa: E402
from validate_real import psnr, deployment_sharpness_check, perceptual_scores  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-array", type=Path, required=True,
                     help="precompute_patches.py output: pre-extracted HR patches, zero decode "
                          "cost per sample. Training must never fully decode a Stock-resolution "
                          "JPEG per step again -- that stalled the first AWS run (0 steps in 30 "
                          "min, 0% GPU util, 3 workers pegged at 98% CPU on throwaway full decodes).")
    ap.add_argument("--held-out-dir", type=Path, required=True,
                     help="original full-res JPEGs for the held-out PSNR gate, never in --patch-array")
    ap.add_argument("--steps", type=int, default=200_000,
                     help="hard upper cap on step count; --time-budget-s is the real stopping condition")
    ap.add_argument("--time-budget-s", type=float, default=5400,
                     help="stop training after this many seconds regardless of step count, "
                          "leaving headroom under the external process timeout for staging/eval/upload")
    ap.add_argument("--checkpoint-every", type=int, default=250,
                     help="write the checkpoint to disk every N steps so a kill from the external "
                          "timeout still leaves a usable model, not just whatever was in memory")
    ap.add_argument("--lr-schedule-steps", type=int, default=3000,
                     help="cosine LR decay horizon; independent of --steps since actual achievable "
                          "steps under --time-budget-s depend on throughput we can't know in advance")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)  # g4dn.xlarge has 4 vCPU; leave 1 for the main process
    ap.add_argument("--held-out", type=int, default=80)
    ap.add_argument("--cuda-name-substr", type=str, default=None)  # single-GPU instance: no index ambiguity, device 0 is fine
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0") if args.cuda_name_substr is None else pick_device(args.cuda_name_substr)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- refusing to train on CPU for the real run")

    held_out_files = sorted(args.held_out_dir.glob("*.jpg"))[: args.held_out]
    print(f"held-out: {len(held_out_files)}  patch array: {args.patch_array}  device: {device}")

    model = TinySR2x().to(device)
    print(f"TinySR2x params: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.lr_schedule_steps)

    loader = make_patch_loader(args.patch_array, args.batch_size, args.workers, seed=args.seed,
                                nominal_len=(args.steps + 2) * args.batch_size)

    def save_checkpoint(step: int, elapsed: float) -> None:
        # Write to a temp path then rename: an external kill mid-write can never leave a
        # corrupt checkpoint, since the rename is atomic and only happens after a full write.
        tmp = args.out_dir / "tinysr2x.pt.tmp"
        torch.save(model.state_dict(), tmp)
        tmp.replace(args.out_dir / "tinysr2x.pt")
        with open(args.out_dir / "progress.json", "w") as fh:
            json.dump({"step": step, "elapsed_s": elapsed}, fh)

    t0 = time.time()
    losses = []
    step = -1
    for step, (lr_t, hr_t) in enumerate(loader):
        elapsed = time.time() - t0
        if step >= args.steps or elapsed >= args.time_budget_s:
            break
        lr_t, hr_t = lr_t.to(device, non_blocking=True), hr_t.to(device, non_blocking=True)
        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:6d}  l1 {loss.item():.4f}  "
                  f"{elapsed:.0f}s/{args.time_budget_s:.0f}s budget  {elapsed / (step + 1) * 1000:.0f} ms/step")
        if step > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(step, elapsed)

    train_elapsed = time.time() - t0
    actual_steps = step + 1
    print(f"stopped after {actual_steps} steps, {train_elapsed:.0f}s "
          f"({'time budget reached' if train_elapsed >= args.time_budget_s else 'step cap reached'})")
    first_avg = sum(losses[:20]) / min(20, len(losses))
    last_avg = sum(losses[-20:]) / min(20, len(losses))

    model.eval()
    model_psnrs, bicubic_psnrs = [], []
    model_preds, bicubics, hrs = [], [], []
    with torch.no_grad():
        for p in held_out_files:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                hr = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
            lr = degrade_hr_to_lr(hr, rng)
            lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(lr_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_uint8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(Image.fromarray(lr).resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC))
            model_psnrs.append(psnr(pred_uint8, hr))
            bicubic_psnrs.append(psnr(bicubic, hr))
            model_preds.append(pred_uint8)
            bicubics.append(bicubic)
            hrs.append(hr)

    model_avg = sum(model_psnrs) / len(model_psnrs)
    bicubic_avg = sum(bicubic_psnrs) / len(bicubic_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, bicubic_psnrs))

    # Second, independent gate: real UNDEGRADED crops, the actual deployment input.
    # The first crop model scored +5 dB on the gate above yet shipped softer than
    # bicubic in production (1.4 vs 4.9 Laplacian var) -- this is the check that
    # would have caught it before any AWS money was spent on the bad run.
    model_sharp, bicubic_sharp = deployment_sharpness_check(model, held_out_files, device, rng)

    # Third, independent signal: PSNR/SSIM are documented to anti-correlate with
    # perceived quality in places (PIRM 2018) -- LPIPS/DISTS report alongside PSNR,
    # never instead of it, so a checkpoint can't pass on a metric blind spot.
    model_lpips, model_dists = perceptual_scores(model_preds, hrs, device)
    bicubic_lpips, bicubic_dists = perceptual_scores(bicubics, hrs, device)

    result = {
        "steps": actual_steps, "batch_size": args.batch_size, "train_elapsed_s": train_elapsed,
        "train_loss_first20_avg": first_avg, "train_loss_last20_avg": last_avg,
        "held_out_n": len(model_psnrs), "model_psnr_db": model_avg, "bicubic_psnr_db": bicubic_avg,
        "delta_db": model_avg - bicubic_avg, "wins": wins,
        "deployment_model_sharpness": model_sharp, "deployment_bicubic_sharpness": bicubic_sharp,
        "deployment_gate_passed": model_sharp > bicubic_sharp,
        "model_lpips": model_lpips, "bicubic_lpips": bicubic_lpips,
        "model_dists": model_dists, "bicubic_dists": bicubic_dists,
        "perceptual_gate_passed": model_lpips < bicubic_lpips and model_dists < bicubic_dists,
        "device_name": torch.cuda.get_device_name(0),
    }
    print(json.dumps(result, indent=2))

    torch.save(model.state_dict(), args.out_dir / "tinysr2x.pt")
    with open(args.out_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"saved checkpoint + result.json to {args.out_dir}")


if __name__ == "__main__":
    main()
