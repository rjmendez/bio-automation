"""Self-supervised LR/HR pair synthesis for the crop-upscale track.

History of two failures, in opposite directions:
  v1: blur(random, up to sigma 2.2) + heavy gaussian noise + random re-JPEG on the
      LR side. Modeled on real-world blind-SR degradation (Real-ESRGAN/BSRGAN
      style, meant for restoring already-degraded photos). Scored +5 dB vs bicubic
      on its own held-out *synthetically degraded* eval, but wired into
      topaz.enhance() and fed a genuine small crop (native resolution, not
      synthetically degraded), it came out SOFTER than plain bicubic (Laplacian
      variance 1.4 vs bicubic's 4.9): trained to expect "input is always blurry
      and noisy", it treated a real sharp crop as noise to suppress.
  v2: plain bicubic downsample, nothing else (the obvious fix for v1). FAILED
      gate 1 outright: model PSNR 50.35 dB vs bicubic's 50.51 dB, a near-tie
      (wins 18/40). Bicubic-downsample-then-bicubic-upsample is itself nearly
      lossless for natural images -- there was almost no real information gap
      left for a tiny model to close, so it learned nothing worth shipping.

v3 (current): a single FIXED mild blur (sigma 0.5, not v1's randomized 0-2.2
range), no noise, no JPEG. This models realistic optical softness (lens
diffraction, a slightly-off focus plane) -- something a real small crop
genuinely has a little of -- without inventing the sensor noise Topaz's own
noise-detector already measured as negligible on these frames (see
reference-topaz-measurements), and without the wide randomization that taught
v1 to expect much heavier degradation than deployment ever provides. This gives
the model a small, real, consistent gap to close -- the middle ground between
v1 (too much, wrong kind) and v2 (none at all).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageFilter

BLUR_SIGMA = 0.5  # fixed, not randomized -- see module docstring


def degrade_hr_to_lr(hr: np.ndarray, rng: random.Random) -> np.ndarray:
    """*hr* is [H, W, 3] uint8 RGB, H and W even. Returns [H/2, W/2, 3] uint8 RGB:
    a fixed-sigma gaussian blur (realistic optical softness) then a bicubic 2x
    downsample. *rng* is accepted for call-site compatibility (dataset/training
    code seeds and passes it) even though this function is currently
    deterministic given *hr* -- BLUR_SIGMA is fixed, not sampled.
    """
    img = Image.fromarray(hr).filter(ImageFilter.GaussianBlur(BLUR_SIGMA))
    lr = img.resize((hr.shape[1] // 2, hr.shape[0] // 2), Image.BICUBIC)
    return np.asarray(lr, dtype=np.uint8)


def random_hr_crop(img: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    """A random [size, size, 3] uint8 crop of *img* ([H, W, 3] uint8); *img* must
    be at least *size* on each side (callers filter source images first)."""
    h, w = img.shape[:2]
    y0 = rng.randint(0, h - size)
    x0 = rng.randint(0, w - size)
    return img[y0:y0 + size, x0:x0 + size]
