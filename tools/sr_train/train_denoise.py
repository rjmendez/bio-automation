"""Local real-data gate for the denoise/detail track, same pattern as
validate_real.py for the crop-upscale track: train briefly on the real raw
pairs, then measure on held-out pairs never touched in training.

The baseline here is different from the crop-upscale track's bicubic: there IS
no "do nothing" baseline that makes sense except literally doing nothing --
PSNR(plain, enhanced_target) measures how far the plain decode already is
from the owner's own Lightroom result. The gate is: does the model get closer
to that result than the plain decode already is? If not, it isn't learning
anything useful about the owner's own denoise/detail preference.

M2 recipe upgrade (training-loop only, no architecture change):
  --loss {l1,charbonnier}   Charbonnier (sqrt((pred-target)^2+eps^2)) instead
                            of plain L1 -- smoother gradient near zero error,
                            still pure pixel-distortion (no perceptual/GAN
                            term: this model must never be rewarded for
                            inventing plausible-but-fake detail).
  --fft-weight FLOAT        adds fft_weight * L1(|rfft2(pred)|, |rfft2(target)|)
                            on top of the pixel loss -- still pixel-distortion
                            family (matches a magnitude spectrum, not a learned
                            feature space), 0 disables it.
  --ema-decay {"auto",F,0}  exponential moving average of the weights (see
                            model.EMA); eval/export use the EMA weights, not
                            the raw optimizer weights. "auto" scales the decay
                            ceiling to --train-steps (model._scaled_ema_decay);
                            0 disables EMA entirely (raw weights used instead).
  --aug {legacy,extended}   "extended" adds RGB-channel permutation and
                            CutBlur to the augmentation pool (patch_pair_dataset).
Each is independently switchable so the recipe can be A/B'd against the
unmodified behavior (--loss l1 --fft-weight 0 --ema-decay 0 --aug legacy
reproduces the pre-M2 script exactly, same data/seed/steps, for the gate's
"old" side).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinyRestore1x, count_params, EMA  # noqa: E402
from train import pick_device  # noqa: E402
from patch_pair_dataset import make_pair_loader  # noqa: E402
from validate_real import psnr, perceptual_scores  # noqa: E402
from precompute_denoise_patches import PATCH_SIZE, _decode  # noqa: E402
from degrade import random_hr_crop  # noqa: E402


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt((pred-target)^2 + eps^2), mean-reduced. Approaches L1 for large
    errors and L2 (smooth, non-zero gradient at 0) for small ones -- a common
    swap-in for L1 in restoration tasks since it avoids L1's non-differentiable
    kink exactly where most residuals end up once training has converged."""
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def fft_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between the magnitude spectra of pred and target (real 2D
    FFT over the last two [H, W] dims; batch and channel dims are untouched,
    so this is still a pure per-pixel-frequency-bin distortion metric, not a
    learned/perceptual feature space). Meant as a small-weighted add-on term
    that nudges reconstructions toward matching the target's frequency content
    (e.g. not systematically softer) without ever being able to reward
    hallucinated spatial detail the way a perceptual/adversarial loss could."""
    pred_mag = torch.fft.rfft2(pred, norm="ortho").abs()
    target_mag = torch.fft.rfft2(target, norm="ortho").abs()
    return F.l1_loss(pred_mag, target_mag)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_kind: str, fft_weight: float) -> torch.Tensor:
    base = F.l1_loss(pred, target) if loss_kind == "l1" else charbonnier_loss(pred, target)
    if fft_weight > 0:
        base = base + fft_weight * fft_l1_loss(pred, target)
    return base


def resolve_ema_decay(ema_decay_arg: str, train_steps: int) -> float:
    """'auto' scales the EMA ceiling to --train-steps: 0.995 at 1,000 steps up
    to 0.9995 at 10,000+ steps (clamped to that documented band), so a short
    smoke run doesn't ask for a decay so high the shadow copy barely moves off
    its random initial deep-copy. '0' disables EMA. Anything else is parsed as
    an explicit float and used as-is (still validated against the (0,1) range
    by model.EMA)."""
    if ema_decay_arg == "auto":
        if train_steps <= 0:
            return 0.995
        return min(0.9995, max(0.995, 1.0 - 5.0 / train_steps))
    value = float(ema_decay_arg)
    return value


def held_out_eval(model, held_out_csv: Path, device, rng, n: int = 40):
    """model(plain decode) vs plain decode itself, scored against the real
    Lightroom-enhanced target -- on raw pairs never used in training. Returns
    per-pair PSNR lists (for the pre-existing PSNR gate) plus the raw uint8
    pred/target crops (for the perceptual-score gate, which needs the actual
    images, not just a scalar)."""
    with open(held_out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))[:n]

    model_psnrs, plain_psnrs = [], []
    pred_u8_list, target_u8_list = [], []
    zero_mse_skipped = 0
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

            # A handful of these pairs land, at this particular crop, on a region
            # where the plain decode and the Lightroom-enhanced decode are
            # bit-identical (observed: 6/40 on this held-out set, e.g. the
            # denoiser was a no-op there) -- psnr() returns +inf for a bit-
            # identical pair, and a single inf poisons sum()/len() for the whole
            # list, making every PSNR-gate comparison vacuously "fail" (a
            # pre-existing bug, not introduced by the M2 loss/EMA/aug changes
            # below -- confirmed by reproducing it against the unmodified recipe
            # too). There is also no real restoration signal to score in a patch
            # with zero difference from its own target, so skip it from the
            # average rather than let one inf decide every run's gate.
            if np.array_equal(plain_crop, target_crop):
                zero_mse_skipped += 1
                continue

            inp = torch.from_numpy(plain_crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)

            model_psnrs.append(psnr(pred_u8, target_crop))
            plain_psnrs.append(psnr(plain_crop, target_crop))
            pred_u8_list.append(pred_u8)
            target_u8_list.append(target_crop)
    if zero_mse_skipped:
        print(f"held-out: skipped {zero_mse_skipped}/{len(rows)} pairs where the sampled crop was "
              f"bit-identical between plain and enhanced (no PSNR signal, would report +inf)")
    return model_psnrs, plain_psnrs, pred_u8_list, target_u8_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-dir", type=Path, required=True, help="precompute_denoise_patches.py --out-dir")
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    # M2 recipe flags -- each independently switchable for A/B (see module docstring).
    ap.add_argument("--loss", choices=["l1", "charbonnier"], default="charbonnier")
    ap.add_argument("--fft-weight", type=float, default=0.1, help="0 disables the FFT-magnitude L1 term")
    ap.add_argument("--ema-decay", type=str, default="auto", help="'auto', '0' (disabled), or an explicit float")
    ap.add_argument("--aug", choices=["legacy", "extended"], default="extended")
    ap.add_argument("--metrics-out", type=Path, default=None, help="dump a JSON of all reported numbers, for A/B diffing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cuda_name_substr)

    ema_decay = resolve_ema_decay(args.ema_decay, args.train_steps)

    model = TinyRestore1x().to(device)
    print(f"TinyRestore1x params: {count_params(model):,}  device: {device}")
    print(f"recipe: loss={args.loss} fft_weight={args.fft_weight} ema_decay={ema_decay} aug={args.aug}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    loader = make_pair_loader(
        args.patch_dir / "patches_plain.npy", args.patch_dir / "patches_enhanced.npy",
        args.batch_size, args.workers, seed=args.seed, nominal_len=(args.train_steps + 2) * args.batch_size,
        aug_mode=args.aug)

    t0 = time.time()
    losses = []
    it = iter(loader)
    for step in range(args.train_steps):
        plain_t, enh_t = next(it)
        plain_t, enh_t = plain_t.to(device), enh_t.to(device)
        pred = model(plain_t)
        loss = compute_loss(pred, enh_t, args.loss, args.fft_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ema is not None:
            ema.update(model)
        losses.append(loss.item())
        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  elapsed {time.time() - t0:.1f}s")

    first_avg = sum(losses[:10]) / min(10, len(losses))
    last_avg = sum(losses[-10:]) / min(10, len(losses))
    print(f"train loss: first-10 avg {first_avg:.4f} -> last-10 avg {last_avg:.4f}")

    eval_model = ema.module if ema is not None else model
    eval_model.eval()
    model_psnrs, plain_psnrs, pred_u8_list, target_u8_list = held_out_eval(
        eval_model, args.patch_dir / "held_out_pairs.csv", device, rng)
    model_avg = sum(model_psnrs) / len(model_psnrs)
    plain_avg = sum(plain_psnrs) / len(plain_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, plain_psnrs))
    print(f"\nheld-out ({len(model_psnrs)} raw pairs, never seen in training, "
          f"eval weights: {'EMA' if ema is not None else 'raw'}):")
    print(f"  model PSNR (vs Lightroom target):  {model_avg:.2f} dB")
    print(f"  plain decode PSNR (vs same target): {plain_avg:.2f} dB")
    print(f"  delta: {model_avg - plain_avg:+.2f} dB")
    print(f"  model closer to target on {wins}/{len(model_psnrs)} pairs")

    model_lpips, model_dists = perceptual_scores(pred_u8_list, target_u8_list, device)
    print(f"  model  LPIPS {model_lpips:.4f}  DISTS {model_dists:.4f}  (lower = closer to target)")

    metrics = {
        "loss": args.loss, "fft_weight": args.fft_weight, "ema_decay": ema_decay, "aug": args.aug,
        "train_steps": args.train_steps, "seed": args.seed,
        "held_out_pairs": len(model_psnrs),
        "psnr_model": model_avg, "psnr_plain": plain_avg, "psnr_wins": wins,
        "lpips_model": model_lpips, "dists_model": model_dists,
    }
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"metrics written: {args.metrics_out}")

    if model_avg <= plain_avg:
        print("\nGATE FAILED: model does not get closer to the owner's own Lightroom result "
              "than the plain decode already is. Do not proceed to AWS.")
        raise SystemExit(1)
    print("\nGATE PASSED.")

    if args.out:
        torch.save(eval_model.state_dict(), args.out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
