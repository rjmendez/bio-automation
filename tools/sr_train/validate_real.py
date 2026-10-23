"""Real-data gate before any paid AWS training: train briefly on real Stock JPEGs,
then measure PSNR on a held-out split against the one baseline that matters --
plain bicubic upsampling. The existing Topaz ghqv2 2x model was measured 5 dB
WORSE than bicubic/Lanczos on real frames; this script exists so the same
mistake can't reach AWS undetected. If the trained model doesn't beat bicubic
here, nothing proceeds to a billed instance.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import piq
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinySR2x, count_params  # noqa: E402
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402
from train import pick_device, PATCH_SIZE  # noqa: E402


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """pred/target: uint8 arrays, same shape. Always a native Python float --
    np.log10 on a numpy scalar returns numpy.float64, which json.dumps cannot
    serialize (the bug that lost the first AWS run's final result.json: the
    held-out eval completed fine, but the json.dumps(result) print crashed on
    'Object of type int64 is not JSON serializable' before anything was saved)."""
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def _laplacian_var(gray: np.ndarray) -> float:
    gray = gray.astype(np.float64)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def deployment_sharpness_check(model, held_out_files: list[Path], device, rng: random.Random, n: int = 40):
    """Feed real, UNDEGRADED native crops directly through the model (no
    degrade_hr_to_lr) and compare a Laplacian-variance detail proxy against plain
    bicubic upscaling of the same crops. This is the check that would have caught
    the first crop model's regression: it scored +5 dB on the synthetic-degradation
    PSNR gate below, yet came out softer than bicubic (1.4 vs 4.9) once fed a real,
    already-sharp small crop in production -- because training always fed it a
    synthetically blurred+noised LR, never a genuinely native-resolution one.

    Not a precision metric (there's no ground-truth "true" 2x detail for a real
    crop to compare against) -- a sanity floor: the model must not make a real
    crop objectively softer than doing nothing. model_avg <= bicubic_avg means
    the model learned to expect degraded input and will disappoint on real crops
    even if its PSNR gate above looks great.
    """
    model_vars, bicubic_vars = [], []
    with torch.no_grad():
        for p in held_out_files[:n]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                native = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)  # genuinely native, UNDEGRADED
            inp_t = torch.from_numpy(native.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(
                Image.fromarray(native).resize((PATCH_SIZE * 2, PATCH_SIZE * 2), Image.BICUBIC))
            model_vars.append(_laplacian_var(np.asarray(Image.fromarray(pred_u8).convert("L"))))
            bicubic_vars.append(_laplacian_var(np.asarray(Image.fromarray(bicubic).convert("L"))))
    return sum(model_vars) / len(model_vars), sum(bicubic_vars) / len(bicubic_vars)


def perceptual_scores(pred_u8_list: list[np.ndarray], target_u8_list: list[np.ndarray], device) -> tuple[float, float]:
    """Mean LPIPS and DISTS (both: lower = more similar) over a list of uint8
    HWC image pairs. PSNR/SSIM are documented to anti-correlate with human
    judgment in places (PIRM 2018) -- this exists so a checkpoint can't pass
    the PSNR gate while actually looking worse, the way v1 would not have been
    caught by PSNR alone if its failure mode had been "looks wrong" rather than
    "numerically softer than bicubic". Report alongside PSNR, never instead of it.
    """
    def to_tensor(arr_list: list[np.ndarray]) -> torch.Tensor:
        stacked = np.stack(arr_list).astype(np.float32) / 255.0
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).to(device)

    pred_t, target_t = to_tensor(pred_u8_list), to_tensor(target_u8_list)
    with torch.no_grad():
        lpips = piq.LPIPS(reduction="mean")(pred_t, target_t).item()
        dists = piq.DISTS(reduction="mean")(pred_t, target_t).item()
    return float(lpips), float(dists)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--train-steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--held-out", type=int, default=40, help="files reserved for eval, never used in training")
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cuda_name_substr)

    all_files = sorted(args.source_dir.glob("*.jpg"))
    if len(all_files) < args.held_out + 50:
        raise SystemExit(f"only {len(all_files)} files under {args.source_dir}, need held-out + a real train pool")

    # Deterministic split so training and eval never touch the same source image.
    shuffled = all_files[:]
    random.Random(1234).shuffle(shuffled)  # fixed seed independent of --seed: the split itself must not vary run to run
    held_out_files = shuffled[: args.held_out]
    train_files = shuffled[args.held_out:]
    print(f"train pool: {len(train_files)} files   held-out: {len(held_out_files)} files")

    model = TinySR2x().to(device)
    print(f"TinySR2x params: {count_params(model):,}  device: {device}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    def sample_hr_batch(files: list[Path], n: int) -> np.ndarray:
        out = np.empty((n, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        for i in range(n):
            for _ in range(8):
                p = rng.choice(files)
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if im.width >= PATCH_SIZE and im.height >= PATCH_SIZE:
                        out[i] = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
                        break
            else:
                raise RuntimeError(f"no file >= {PATCH_SIZE}px found near {p}")
        return out

    t0 = time.time()
    losses = []
    for step in range(args.train_steps):
        hr_uint8 = sample_hr_batch(train_files, args.batch_size)
        lr_list = [degrade_hr_to_lr(hr_uint8[i], rng) for i in range(hr_uint8.shape[0])]
        lr = np.stack(lr_list).astype(np.float32) / 255.0
        hr = hr_uint8.astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr).permute(0, 3, 1, 2).to(device)
        hr_t = torch.from_numpy(hr).permute(0, 3, 1, 2).to(device)

        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps - 1:
            print(f"step {step:5d}  l1 {loss.item():.4f}  elapsed {time.time() - t0:.1f}s")

    first_avg = sum(losses[:10]) / min(10, len(losses))
    last_avg = sum(losses[-10:]) / min(10, len(losses))
    print(f"train loss: first-10 avg {first_avg:.4f} -> last-10 avg {last_avg:.4f}")

    # --- held-out PSNR: model vs plain bicubic, same LR input, never-trained-on images ---
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

            bicubic = np.asarray(
                Image.fromarray(lr).resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC))

            model_psnrs.append(psnr(pred_uint8, hr))
            bicubic_psnrs.append(psnr(bicubic, hr))
            model_preds.append(pred_uint8)
            bicubics.append(bicubic)
            hrs.append(hr)

    model_avg = sum(model_psnrs) / len(model_psnrs)
    bicubic_avg = sum(bicubic_psnrs) / len(bicubic_psnrs)
    wins = sum(m > b for m, b in zip(model_psnrs, bicubic_psnrs))
    print(f"\nheld-out ({len(model_psnrs)} images, never seen in training):")
    print(f"  model PSNR:   {model_avg:.2f} dB")
    print(f"  bicubic PSNR: {bicubic_avg:.2f} dB")
    print(f"  delta:        {model_avg - bicubic_avg:+.2f} dB")
    print(f"  model beats bicubic on {wins}/{len(model_psnrs)} images")

    model_lpips, model_dists = perceptual_scores(model_preds, hrs, device)
    bicubic_lpips, bicubic_dists = perceptual_scores(bicubics, hrs, device)
    print(f"  model  LPIPS {model_lpips:.4f}  DISTS {model_dists:.4f}  (lower = closer to target)")
    print(f"  bicubic LPIPS {bicubic_lpips:.4f}  DISTS {bicubic_dists:.4f}")
    if model_avg > bicubic_avg and (model_lpips >= bicubic_lpips or model_dists >= bicubic_dists):
        print("  WARNING: model wins on PSNR but not on LPIPS/DISTS -- PSNR/SSIM are documented "
              "to anti-correlate with perceived quality (PIRM 2018); do not accept this checkpoint "
              "on PSNR alone, review real 100% crops before proceeding.")

    if model_avg <= bicubic_avg:
        print("\nGATE FAILED: trained model does not beat plain bicubic on held-out data. "
              "Do not proceed to AWS -- this is the exact failure mode measured for ghqv2 vs Lanczos.")
        raise SystemExit(1)
    print("\nGATE 1 PASSED: trained model beats bicubic on synthetically-degraded held-out data.")

    # --- second, independent gate: real undegraded crops, the actual deployment input ---
    model_sharp, bicubic_sharp = deployment_sharpness_check(model, held_out_files, device, rng)
    print(f"\ndeployment sanity check (real undegraded crops, {min(40, len(held_out_files))} images):")
    print(f"  model Laplacian var:   {model_sharp:.2f}")
    print(f"  bicubic Laplacian var: {bicubic_sharp:.2f}")
    if model_sharp <= bicubic_sharp:
        print("\nGATE 2 FAILED: model is softer than plain bicubic on real undegraded crops -- "
              "it will disappoint in production even though gate 1 passed. Do not proceed to AWS.")
        raise SystemExit(1)
    print("\nGATE 2 PASSED: model beats bicubic on real undegraded crops too.")


if __name__ == "__main__":
    main()
