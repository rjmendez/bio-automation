"""Dataset over a pre-extracted HR patch array (see precompute_patches.py):
zero decode cost per sample, just an mmap read + the cheap degrade() pass.
Replaces SRPatchDataset (dataset.py) as the training-time source; held-out
evaluation still reads the original full-res JPEGs directly (unchanged --
that only happens once per run, not once per step, so it was never the
bottleneck)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from degrade import degrade_hr_to_lr


class PatchArrayDataset(Dataset):
    def __init__(self, npy_path: Path, nominal_len: int = 10_000, seed: int = 0) -> None:
        self.npy_path = npy_path
        self.nominal_len = nominal_len
        self.rng = random.Random(seed)
        self._patches = None  # lazy per-process mmap: opened after fork/spawn, never pickled

    def _ensure_loaded(self) -> np.ndarray:
        if self._patches is None:
            self._patches = np.load(self.npy_path, mmap_mode="r")
        return self._patches

    def __len__(self) -> int:
        return self.nominal_len

    def __getitem__(self, idx: int):
        patches = self._ensure_loaded()
        hr = np.array(patches[self.rng.randrange(len(patches))])  # copy out of the mmap
        lr = degrade_hr_to_lr(hr, self.rng)
        lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1)
        hr_t = torch.from_numpy(hr.astype(np.float32) / 255.0).permute(2, 0, 1)
        return lr_t, hr_t


def _worker_init(worker_id: int) -> None:
    import torch.utils.data as tud

    info = tud.get_worker_info()
    info.dataset.rng = random.Random(info.seed)


def make_patch_loader(npy_path: Path, batch_size: int, num_workers: int, seed: int = 0,
                       nominal_len: int = 10_000):
    from torch.utils.data import DataLoader

    ds = PatchArrayDataset(npy_path, nominal_len=nominal_len, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
