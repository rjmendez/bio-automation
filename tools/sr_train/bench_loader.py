"""Quick timed comparison: parallel DataLoader vs the single-process path already
measured at ~2.4 s/step on real Stock JPEGs (validate_real.py's 800-step run)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinySR2x  # noqa: E402
from train import pick_device  # noqa: E402
from dataset import make_loader  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cuda-name-substr", type=str, default="4070")
    args = ap.parse_args()

    device = pick_device(args.cuda_name_substr)
    files = sorted(args.source_dir.glob("*.jpg"))
    loader = make_loader(files, args.batch_size, args.workers,
                          nominal_len=(args.steps + 2) * args.batch_size)

    model = TinySR2x().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    it = iter(loader)
    next(it)  # exclude worker warm-up/startup from the timed loop
    t0 = time.time()
    for step in range(args.steps):
        lr_t, hr_t = next(it)
        lr_t, hr_t = lr_t.to(device), hr_t.to(device)
        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    elapsed = time.time() - t0
    print(f"{args.steps} steps, {args.workers} workers: {elapsed:.1f}s total, "
          f"{elapsed / args.steps * 1000:.0f} ms/step  (single-process baseline: ~2400 ms/step)")
