"""Minimal standalone replacement for the two basicsr symbols hat_arch.py needs,
so loading the official HAT architecture for inference doesn't require installing
the full basicsr framework (old torch pins, registry/training machinery we don't
use -- we only load a pretrained state_dict and run eval())."""
from __future__ import annotations

from torch.nn.init import trunc_normal_  # noqa: F401  (torch has this natively now)


def to_2tuple(x):
    return x if isinstance(x, (tuple, list)) else (x, x)


class _NoOpRegistry:
    def register(self):
        def _decorator(cls):
            return cls
        return _decorator


ARCH_REGISTRY = _NoOpRegistry()
