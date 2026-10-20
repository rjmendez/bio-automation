"""Real-model tests for bio_automation.topaz.

These run by default. They skip LOUDLY (the reason names the missing file)
only when the licensed Topaz model files are genuinely absent or still
0-byte download stubs, which is the case on CI hosts without Topaz Photo AI.
Models and their extracted cache stay on C: and are read, never modified.
"""
from __future__ import annotations

import numpy as np
import pytest

from bio_automation import topaz
from bio_automation.topaz import MODELS_DIR, ModelCache, sharpen

SLB = MODELS_DIR / "slb-v1-fp32-512x512-ov.tz"


def _require(path):
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"SKIPPED LOUDLY: Topaz model {path} is missing or a 0-byte stub")
    pytest.importorskip("openvino", reason="SKIPPED LOUDLY: openvino is not installed")


def _textured_bgr(h=560, w=560, seed=7):
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 215, size=(h // 4, w // 4, 3)).astype(np.float32)
    big = np.kron(base, np.ones((4, 4, 1), dtype=np.float32))
    noise = rng.randn(h, w, 3).astype(np.float32) * 6.0
    return np.clip(big + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def slb_cache():
    _require(SLB)
    cache = ModelCache()
    yield cache
    cache.__exit__(None, None, None)


def test_slb_io_signature(slb_cache):
    compiled = slb_cache.load(SLB)
    names = {i.any_name: tuple(i.shape) for i in compiled.inputs}
    assert names == {"netInput": (1, 3, 512, 512), "t_param": (1,), "t_param1": (1,)}
    assert [tuple(o.shape) for o in compiled.outputs] == [(1, 3, 512, 512)]


def test_real_slb_output_has_no_black_first_row_or_column(slb_cache):
    img = _textured_bgr()
    out = sharpen(img, slb_cache, gate=False, alpha=1.0, blur=0.25)
    assert out.shape == img.shape
    # regression: the old ramp window produced mean 0.0 on row 0 / col 0
    assert float(out[0].mean()) == pytest.approx(float(img[0].mean()), abs=25.0)
    assert float(out[:, 0].mean()) == pytest.approx(float(img[:, 0].mean()), abs=25.0)
    assert float(out[0].mean()) > 20.0 and float(out[:, 0].mean()) > 20.0


def test_real_slb_alpha_blend_is_the_midpoint_between_input_and_model(slb_cache):
    img = _textured_bgr()
    full = sharpen(img, slb_cache, gate=False, alpha=1.0)
    half = sharpen(img, slb_cache, gate=False, alpha=0.5)
    midpoint = (img.astype(np.float32) + full.astype(np.float32)) / 2.0
    assert float(np.abs(half.astype(np.float32) - midpoint).max()) <= 1.0
    # positive twin: the model really changes this image, so the check is not vacuous
    assert float(np.abs(full.astype(np.float32) - img.astype(np.float32)).mean()) > 2.0
