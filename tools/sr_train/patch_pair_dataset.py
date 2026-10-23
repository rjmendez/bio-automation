"""Dataset over the precomputed real (plain, enhanced) patch pairs -- see
precompute_denoise_patches.py. No synthetic degradation: these are genuine
paired samples, so unlike the crop-upscale dataset there's nothing to apply
on the fly except light augmentation (flip/rotate), which is safe here
because it's applied identically to both halves of the pair.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchPairDataset(Dataset):
    def __init__(self, plain_path: Path, enhanced_path: Path, nominal_len: int = 10_000, seed: int = 0) -> None:
        self.plain_path = plain_path
        self.enhanced_path = enhanced_path
        self.nominal_len = nominal_len
        self.rng = random.Random(seed)
        self._plain = None
        self._enhanced = None

    def _ensure_loaded(self) -> None:
        if self._plain is None:
            self._plain = np.load(self.plain_path, mmap_mode="r")
            self._enhanced = np.load(self.enhanced_path, mmap_mode="r")
            assert len(self._plain) == len(self._enhanced), "plain/enhanced patch count mismatch"

    def __len__(self) -> int:
        return self.nominal_len

    def __getitem__(self, idx: int):
        self._ensure_loaded()
        i = self.rng.randrange(len(self._plain))
        plain = np.array(self._plain[i])
        enhanced = np.array(self._enhanced[i])

        # Flip/rotate augmentation applied identically to both halves of the pair.
        if self.rng.random() < 0.5:
            plain, enhanced = plain[:, ::-1].copy(), enhanced[:, ::-1].copy()
        if self.rng.random() < 0.5:
            plain, enhanced = plain[::-1, :].copy(), enhanced[::-1, :].copy()
        k = self.rng.randrange(4)
        if k:
            plain, enhanced = np.rot90(plain, k).copy(), np.rot90(enhanced, k).copy()

        plain_t = torch.from_numpy(plain.astype(np.float32) / 255.0).permute(2, 0, 1)
        enh_t = torch.from_numpy(enhanced.astype(np.float32) / 255.0).permute(2, 0, 1)
        return plain_t, enh_t


def _worker_init(worker_id: int) -> None:
    import torch.utils.data as tud

    info = tud.get_worker_info()
    info.dataset.rng = random.Random(info.seed)


def make_pair_loader(plain_path: Path, enhanced_path: Path, batch_size: int, num_workers: int,
                      seed: int = 0, nominal_len: int = 10_000):
    from torch.utils.data import DataLoader

    ds = PatchPairDataset(plain_path, enhanced_path, nominal_len=nominal_len, seed=seed)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
