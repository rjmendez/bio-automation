"""Dataset over the precomputed real (plain, enhanced) patch pairs -- see
precompute_denoise_patches.py. No synthetic degradation: these are genuine
paired samples, so unlike the crop-upscale dataset there's nothing to apply
on the fly except augmentation, which is safe here because every augmentation
below is applied identically to both halves of the pair (so the pairing stays
pixel-exact) -- except CutBlur, which by design patches the two halves
*differently* at one shared location (see `_cut_blur`).

Two augmentation modes, selected by `aug_mode`:
  "legacy":   the original scheme -- independent flip-h / flip-v / rot90 draws
              (each may or may not fire), unconditionally.
  "extended": M2 upgrade -- exactly ONE augmentation is drawn per sample from a
              weighted pool (CutBlur dominant, the rest roughly even), instead
              of stacking several at once. Added members: RGB-channel
              permutation and CutBlur. Deliberately NOT added: Mixup/CutMix/
              Cutout -- those blend or blank whole regions and would distort
              the local spatial structure this restoration task depends on.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# "extended" mode: draw exactly one entry per sample, weighted. CutBlur
# dominant (~40%), the rest ~10-15% each -- see module docstring.
_EXTENDED_AUG_POOL = (
    ("cutblur", 0.40),
    ("flip_h", 0.15),
    ("flip_v", 0.15),
    ("rot90", 0.15),
    ("channel_perm", 0.15),
)


def _flip_h(plain: np.ndarray, enhanced: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    return plain[:, ::-1].copy(), enhanced[:, ::-1].copy()


def _flip_v(plain: np.ndarray, enhanced: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    return plain[::-1, :].copy(), enhanced[::-1, :].copy()


def _rot90(plain: np.ndarray, enhanced: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randrange(1, 4)  # 1..3: k=0 would be a no-op draw, wasted
    return np.rot90(plain, k).copy(), np.rot90(enhanced, k).copy()


def _channel_perm(plain: np.ndarray, enhanced: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Same random permutation of the 3 RGB channels applied to both halves --
    stays a valid pair (channel identity carries no absolute meaning the model
    is meant to rely on), and is a cheap way to triple the effective color
    diversity of a patch pool drawn from only 871 raw pairs."""
    idx = list(range(3))
    rng.shuffle(idx)
    return plain[:, :, idx].copy(), enhanced[:, :, idx].copy()


def _cut_blur(plain: np.ndarray, enhanced: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """CutBlur (Yoo et al. 2020), adapted to a same-resolution real-pair task:
    picks one small random region and, with equal probability, either
      (a) pastes the LR/plain patch into the HR/target region -- the target
          now equals the input there, so the model must learn to leave an
          already-fine-looking local region alone (restraint, not just
          restoration everywhere), or
      (b) pastes the HR/enhanced patch into the LR/plain input region -- the
          input now already looks "clean" in that patch while its surroundings
          don't, teaching the model not to over-process (or halo) a patch that
          needs nothing done to it.
    This is the one augmentation here that intentionally makes the two halves
    differ beyond a shared geometric/color transform -- that asymmetry is the
    point, not a bug in "applied identically to both halves".
    """
    h, w = plain.shape[:2]
    cut_h = rng.randint(max(1, h // 8), max(2, h // 2))
    cut_w = rng.randint(max(1, w // 8), max(2, w // 2))
    y0 = rng.randint(0, h - cut_h)
    x0 = rng.randint(0, w - cut_w)
    plain_out, enhanced_out = plain.copy(), enhanced.copy()
    region = (slice(y0, y0 + cut_h), slice(x0, x0 + cut_w))
    if rng.random() < 0.5:
        enhanced_out[region] = plain[region]
    else:
        plain_out[region] = enhanced[region]
    return plain_out, enhanced_out


_EXTENDED_AUG_FN = {
    "cutblur": _cut_blur,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "rot90": _rot90,
    "channel_perm": _channel_perm,
}


class PatchPairDataset(Dataset):
    def __init__(self, plain_path: Path, enhanced_path: Path, nominal_len: int = 10_000, seed: int = 0,
                 aug_mode: str = "legacy") -> None:
        if aug_mode not in ("legacy", "extended"):
            raise ValueError(f"aug_mode must be 'legacy' or 'extended', got {aug_mode!r}")
        self.plain_path = plain_path
        self.enhanced_path = enhanced_path
        self.nominal_len = nominal_len
        self.aug_mode = aug_mode
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

        if self.aug_mode == "legacy":
            # Flip/rotate augmentation applied identically to both halves of the pair.
            if self.rng.random() < 0.5:
                plain, enhanced = plain[:, ::-1].copy(), enhanced[:, ::-1].copy()
            if self.rng.random() < 0.5:
                plain, enhanced = plain[::-1, :].copy(), enhanced[::-1, :].copy()
            k = self.rng.randrange(4)
            if k:
                plain, enhanced = np.rot90(plain, k).copy(), np.rot90(enhanced, k).copy()
        else:
            names, weights = zip(*_EXTENDED_AUG_POOL)
            choice = self.rng.choices(names, weights=weights, k=1)[0]
            plain, enhanced = _EXTENDED_AUG_FN[choice](plain, enhanced, self.rng)

        plain_t = torch.from_numpy(plain.astype(np.float32) / 255.0).permute(2, 0, 1)
        enh_t = torch.from_numpy(enhanced.astype(np.float32) / 255.0).permute(2, 0, 1)
        return plain_t, enh_t


def _worker_init(worker_id: int) -> None:
    import torch.utils.data as tud

    info = tud.get_worker_info()
    info.dataset.rng = random.Random(info.seed)


def make_pair_loader(plain_path: Path, enhanced_path: Path, batch_size: int, num_workers: int,
                      seed: int = 0, nominal_len: int = 10_000, aug_mode: str = "legacy"):
    from torch.utils.data import DataLoader

    ds = PatchPairDataset(plain_path, enhanced_path, nominal_len=nominal_len, seed=seed, aug_mode=aug_mode)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
