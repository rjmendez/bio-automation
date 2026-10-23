"""Train TinySR2x on real source images (full-res Stock-ready JPEGs or raw decodes).

Two modes:
  --smoke-test   synthetic random patches, a handful of steps, CPU is fine.
                 Proves the model/loss/optimizer plumbing is correct before
                 anything touches a billed GPU. Not a quality signal.
  (default)      real source images from --source-dir, random crop + degrade
                 per step. This is the real training run; it expects a CUDA
                 device (fails loudly otherwise, matching AGENTS.md "No CPU
                 processing" for anything that is not an explicit dev check).

GPU selection never trusts a bare index: this machine has shown the CUDA
device order flip between runtimes (ONNXRuntime enumerates PCI order, PyTorch
enumerates FASTEST_FIRST), and the 4070 Ti is the display card that TDR-resets
under sustained load. --cuda-name-substr picks a device by
torch.cuda.get_device_name() instead of a positional index.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinySR2x, count_params  # noqa: E402
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402

PATCH_SIZE = 256  # HR patch edge; LR input is 128


def pick_device(cuda_name_substr: str | None) -> torch.device:
    if cuda_name_substr is None:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-name-substr given but torch.cuda.is_available() is False")
    matches = [i for i in range(torch.cuda.device_count())
               if cuda_name_substr.lower() in torch.cuda.get_device_name(i).lower()]
    if len(matches) != 1:
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        raise RuntimeError(
            f"--cuda-name-substr {cuda_name_substr!r} matched {len(matches)} devices "
            f"(want exactly 1); visible devices: {names}")
    return torch.device(f"cuda:{matches[0]}")


def hr_batch_from_source(paths: list[Path], batch_size: int, rng: random.Random) -> np.ndarray:
    """[batch_size, PATCH_SIZE, PATCH_SIZE, 3] uint8, cropped from random source images."""
    from PIL import Image

    out = np.empty((batch_size, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    for i in range(batch_size):
        for _ in range(8):  # a handful of source images may be too small; skip and retry
            p = rng.choice(paths)
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width >= PATCH_SIZE and im.height >= PATCH_SIZE:
                    out[i] = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
                    break
        else:
            raise RuntimeError(f"no source image >= {PATCH_SIZE}px found in 8 tries")
    return out


def hr_batch_synthetic(batch_size: int, rng: random.Random) -> np.ndarray:
    """Structured-but-arbitrary synthetic patches (not pure noise: pure noise has
    no learnable low-frequency structure for a 2x upscaler to recover, so a
    smoke test on it can't tell a working model from a broken one)."""
    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
    out = np.empty((batch_size, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE]
    for i in range(batch_size):
        fx, fy = rng_np.uniform(0.01, 0.05, 2)
        phase = rng_np.uniform(0, 2 * np.pi)
        base = 128 + 100 * np.sin(fx * xx + fy * yy + phase)
        patch = np.stack([base] * 3, axis=-1)
        patch += rng_np.normal(0, 5, patch.shape)
        out[i] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


def to_tensor_pairs(hr_uint8: np.ndarray, rng: random.Random, device: torch.device):
    """hr_uint8 [B,H,W,3] uint8 -> (lr_tensor [B,3,H/2,W/2], hr_tensor [B,3,H,W]) float32 in [0,1]."""
    lr_list = [degrade_hr_to_lr(hr_uint8[i], rng) for i in range(hr_uint8.shape[0])]
    lr = np.stack(lr_list).astype(np.float32) / 255.0
    hr = hr_uint8.astype(np.float32) / 255.0
    lr_t = torch.from_numpy(lr).permute(0, 3, 1, 2).to(device)
    hr_t = torch.from_numpy(hr).permute(0, 3, 1, 2).to(device)
    return lr_t, hr_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--source-dir", type=Path, default=None,
                     help="directory of source JPEGs (real run; required unless --smoke-test)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--cuda-name-substr", type=str, default=None,
                     help='e.g. "2080" to pick the 2080 Ti; omit for CPU')
    ap.add_argument("--out", type=Path, default=Path("tinysr2x.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.cuda_name_substr)

    if args.smoke_test:
        steps = args.steps or 20
        batch_size = args.batch_size or 4
        source_paths: list[Path] | None = None
    else:
        if args.source_dir is None:
            raise SystemExit("--source-dir is required unless --smoke-test")
        source_paths = sorted(args.source_dir.glob("*.jpg")) + sorted(args.source_dir.glob("*.jpeg"))
        if not source_paths:
            raise SystemExit(f"no .jpg/.jpeg files under {args.source_dir}")
        steps = args.steps or 2000
        batch_size = args.batch_size or 8

    model = TinySR2x().to(device)
    print(f"TinySR2x params: {count_params(model):,}  device: {device}  steps: {steps}")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    losses = []
    t0 = time.time()
    for step in range(steps):
        hr_uint8 = (hr_batch_synthetic(batch_size, rng) if source_paths is None
                    else hr_batch_from_source(source_paths, batch_size, rng))
        lr_t, hr_t = to_tensor_pairs(hr_uint8, rng, device)

        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"step {step:5d}  l1 {loss.item():.4f}  elapsed {time.time() - t0:.1f}s")

    first_avg = sum(losses[:5]) / min(5, len(losses))
    last_avg = sum(losses[-5:]) / min(5, len(losses))
    print(f"loss: first-5 avg {first_avg:.4f} -> last-5 avg {last_avg:.4f}")
    if args.smoke_test and not (last_avg < first_avg):
        raise SystemExit("smoke test FAILED: loss did not decrease -- plumbing is broken, do not train for real")

    torch.save(model.state_dict(), args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
