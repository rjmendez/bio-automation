"""Compact residual 2x super-resolution network (RRDB-lite).

Small on purpose: this trains on a few hundred to a few thousand source images
(see pair_manifest.py / prepare_patches.py), not a web-scale corpus, so a heavy
ESRGAN-sized backbone would overfit long before it generalizes. No GAN/adversarial
loss anywhere in this module: the iNat-track model must not hallucinate detail
into a photo used for species identification (AGENTS.md: "never assume a model
helps", and the existing slb sharpen gate already rejects invented texture).

Tiling contract matches bio_automation.topaz._run_tiled: a [3, H, W] float32
RGB tensor in [0, 1] in, a [3, H*2, W*2] float32 RGB tensor in [0, 1] out, so a
trained checkpoint can be fed through the same tiler the Topaz models use.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return x + y


class TinySR2x(nn.Module):
    """2x upscaler: stem conv -> N residual blocks -> pixel-shuffle upsample -> output conv.

    *channels*/*num_blocks* default to a size that trains in minutes on a CPU for
    the smoke test and in well under an hour per epoch on a single T4 for the
    real run (871-ish source images, patch-augmented).
    """

    def __init__(self, channels: int = 48, num_blocks: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.post = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat + self.post(self.blocks(feat))
        up = self.upsample(feat)
        residual = self.to_rgb(up)
        # Bilinear-upsampled input as the base, network only has to learn the
        # residual detail on top of it -- keeps early training stable and
        # guarantees the output can never be wildly unlike the input.
        base = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return torch.clamp(base + residual, 0.0, 1.0)


class TinyRestore1x(nn.Module):
    """Same-resolution denoise/detail model: stem -> N residual blocks -> output
    conv, no pixel-shuffle upsample (no scale change at all). For the raw-pair
    track: input = a plain rawpy decode, target = the owner's own Lightroom
    "Enhanced-NR" (AI denoise) export of the SAME raw frame -- a real pair, not a
    synthetic degradation, so (unlike the first crop-upscale model) there is no
    train/deploy domain-mismatch risk here: the model always sees a genuine plain
    decode as input, both in training and in production.
    """

    def __init__(self, channels: int = 48, num_blocks: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.post = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = feat + self.post(self.blocks(feat))
        residual = self.to_rgb(feat)
        # Identity base + residual: same stability rationale as TinySR2x, except
        # the base here is the input itself (no resize needed at 1x).
        return torch.clamp(x + residual, 0.0, 1.0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class EMA:
    """Exponential moving average of a model's weights -- a training-loop
    utility, not a new architecture: it wraps whatever model it is given
    (TinySR2x or TinyRestore1x unchanged) and keeps a shadow copy that moves
    more slowly than the raw optimizer weights, which is usually a better
    checkpoint to evaluate/export than the last noisy SGD step.

    Decay ramps up from 0 toward *decay* over the first ~9 updates (the
    `updates / (updates + 9)` term below, the standard EMA warmup so the
    shadow copy does not start out 99.9%-weighted toward an almost-random
    initialization) and should itself be chosen scaled to the planned total
    step count by the caller (a short run should use a lower ceiling so the
    shadow actually has time to move; see train_denoise.py's --ema-decay).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, self.updates / (self.updates + 9))
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)  # buffers (none in these two models today) copied as-is, never EMA'd
