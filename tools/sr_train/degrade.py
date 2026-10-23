"""Self-supervised LR/HR pair synthesis for the crop-upscale track.

LR = plain bicubic downsample of HR, nothing else. This was NOT the first version:
an earlier revision added synthetic blur + heavy gaussian noise + random re-JPEG,
modeled on real-world blind-SR degradation (Real-ESRGAN/BSRGAN style, meant for
restoring already-degraded photos). That model scored +5 dB vs bicubic on its own
held-out *synthetically degraded* eval -- but when wired into topaz.enhance() and
fed a genuine small crop (native resolution, not synthetically degraded), it came
out SOFTER than plain bicubic (Laplacian variance 1.4 vs bicubic's 4.9): trained to
expect "input is always blurry and noisy, sharpen while upscaling", it treated a
real sharp crop as noise to suppress.

The deployment input (a real, already-sharp, already-native-resolution small crop)
is not a degraded photo -- it is just smaller. That is the classic bicubic-SR
setup (SRCNN/EDSR-style), not the blind real-world-restoration setup. Matching the
degradation to the deployment distribution matters more than making it "realistic"
in the abstract.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image


def degrade_hr_to_lr(hr: np.ndarray, rng: random.Random) -> np.ndarray:
    """*hr* is [H, W, 3] uint8 RGB, H and W even. Returns [H/2, W/2, 3] uint8 RGB:
    a plain bicubic 2x downsample, matching what a real native-resolution small
    crop actually looks like (sharp at its own scale, not artificially degraded).
    *rng* is accepted for call-site compatibility (dataset/training code seeds and
    passes it) even though this function is currently deterministic given *hr*.
    """
    img = Image.fromarray(hr)
    lr = img.resize((hr.shape[1] // 2, hr.shape[0] // 2), Image.BICUBIC)
    return np.asarray(lr, dtype=np.uint8)


def random_hr_crop(img: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    """A random [size, size, 3] uint8 crop of *img* ([H, W, 3] uint8); *img* must
    be at least *size* on each side (callers filter source images first)."""
    h, w = img.shape[:2]
    y0 = rng.randint(0, h - size)
    x0 = rng.randint(0, w - size)
    return img[y0:y0 + size, x0:x0 + size]
