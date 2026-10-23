"""Parallel-decode dataset wrapper so GPU compute isn't stalled behind single-threaded
JPEG decode of full-res (24-100 MP) source images. The 800-step real-data validation
run spent ~32 min wall-clock on a model that needs under 10 ms/step of GPU time --
almost all of it was main-process PIL decode. A DataLoader with worker processes
overlaps decode of the NEXT batch with GPU compute on the CURRENT one instead of
reducing decode work (full-res decode is unavoidable: we need a genuine full-res
crop, not a thumbnail -- PIL's JPEG draft() downscale mode would defeat the point).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from degrade import degrade_hr_to_lr, random_hr_crop
from train import PATCH_SIZE


class SRPatchDataset(Dataset):
    """Infinite random-sample dataset: __len__ is a nominal epoch size (controls how
    often the DataLoader's workers get recycled / progress gets logged), __getitem__
    ignores *idx* and draws a fresh random (file, crop, degradation) every call so
    training never exhausts real variety. Each worker gets its own RNG (seeded from
    base_seed + worker id by the DataLoader's worker_init_fn, set by the caller) so
    parallel workers don't draw identical patches.
    """

    def __init__(self, files: list[Path], nominal_len: int = 10_000, seed: int = 0) -> None:
        self.files = files
        self.nominal_len = nominal_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.nominal_len

    def __getitem__(self, idx: int):
        for _ in range(8):
            p = self.rng.choice(self.files)
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width >= PATCH_SIZE and im.height >= PATCH_SIZE:
                    hr = random_hr_crop(np.asarray(im), PATCH_SIZE, self.rng)
                    break
        else:
            raise RuntimeError(f"no file >= {PATCH_SIZE}px found near {p}")

        lr = degrade_hr_to_lr(hr, self.rng)
        lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return lr_t, hr_t


def _worker_init(worker_id: int) -> None:
    import torch.utils.data as tud

    info = tud.get_worker_info()
    # Reseed this worker's dataset copy so sibling workers don't sample identical patches.
    info.dataset.rng = random.Random(info.seed)


def make_loader(files: list[Path], batch_size: int, num_workers: int, seed: int = 0,
                 nominal_len: int = 10_000):
    from torch.utils.data import DataLoader

    ds = SRPatchDataset(files, nominal_len=nominal_len, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
