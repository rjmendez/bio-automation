"""Live, unmocked proof that bio_automation.preclean's raw-decode and
OpenVINO-model code paths actually work against real data -- not hermetic,
and deliberately NOT gated behind an opt-in env var or marker: every
default `pytest` run exercises the real rawpy call against a real raw
file (tests/fixtures/sample_raw.NEF) and the real slb OpenVINO model.

Why this file exists and is not skippable: test_preclean.py mocks both
rawpy.imread and _sharpen_f32, which proves the module's own logic is
internally consistent but nothing about whether rawpy's actual kwargs or
the real model's actual input/output tensor names match what the code
assumes. A prior tool (smart_focus_stack.py, this same project) was
recorded as "built and verified" in Loci's memory but was never actually
proven against anything that persisted -- the verification evaporated
with whatever ephemeral environment ran it, and the file itself no longer
exists anywhere on disk. An opt-in-gated live test has the same failure
mode in miniature: a check that doesn't run by default doesn't run.

Cropped to a 768x768 window post-decode (not the full ~48MP frame) so
this stays fast (~3-10s) while still forcing _run_tiled through multiple
overlapping tiles with real blending, not a single trivial tile.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation.preclean import (
    MODELS_DIR,
    SUBJECT_MASK_MODEL,
    _sharpen_f32,
    _to_chw_rgb_f32,
    apply_levels,
    apply_levels_masked,
    compute_fixed_params,
    decode_raw_to_bgr16,
    refine_mask_by_color,
    subject_mask,
)
from bio_automation.topaz import ModelCache

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.NEF"


def _crop_chw(bgr16: np.ndarray, size: int = 768) -> np.ndarray:
    h, w = bgr16.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    crop = bgr16[cy - half:cy + half, cx - half:cx + half]
    return _to_chw_rgb_f32(crop)


@pytest.fixture(scope="module")
def cache():
    with ModelCache() as c:
        yield c


class TestDecodeRawToBgr16Live:
    def test_real_nef_decodes_to_sane_sensor_data(self):
        assert FIXTURE.exists(), f"missing committed fixture: {FIXTURE}"
        bgr16 = decode_raw_to_bgr16(FIXTURE)

        assert bgr16.dtype == np.uint16
        assert bgr16.ndim == 3
        assert bgr16.shape[2] == 3
        # a real daylight macro exposure: not blank, not saturated, real variance
        assert bgr16.min() < bgr16.max()
        assert 0 < int(bgr16.mean()) < 65535
        assert float(bgr16.std()) > 100.0  # real sensor texture, not a flat/corrupt frame


class TestSharpenF32Live:
    def test_output_shape_and_range_on_real_tile_boundary_crossing_crop(self, cache):
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _crop_chw(bgr16, size=768)  # forces multi-tile blending, see module docstring

        out = _sharpen_f32(chw, cache, blur=0.5, noise=0.5)

        assert out.shape == chw.shape
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_model_actually_transforms_real_data_not_identity(self, cache):
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _crop_chw(bgr16, size=768)

        out = _sharpen_f32(chw, cache, blur=0.5, noise=0.5)

        assert not np.array_equal(out, chw)

    def test_different_noise_params_produce_different_real_output(self, cache):
        """Proves noise actually reaches the real model call, not silently ignored."""
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _crop_chw(bgr16, size=768)

        low = _sharpen_f32(chw, cache, blur=0.5, noise=0.0)
        high = _sharpen_f32(chw, cache, blur=0.5, noise=1.0)

        assert not np.allclose(low, high)

    def test_same_input_and_params_are_reproducible(self, cache):
        """Real-model determinism check: same input/params -> same output, not
        just 'ran without crashing'."""
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _crop_chw(bgr16, size=768)

        first = _sharpen_f32(chw, cache, blur=0.4, noise=0.6)
        second = _sharpen_f32(chw, cache, blur=0.4, noise=0.6)

        assert np.array_equal(first, second)


class TestComputeFixedParamsLive:
    def test_real_autoparam_model_returns_in_range_blur(self, cache):
        blur, noise = compute_fixed_params([FIXTURE], cache, noise=0.7)

        assert 0.0 <= blur <= 1.0
        assert noise == 0.7

    def test_real_autoparam_model_is_reproducible(self, cache):
        blur1, _ = compute_fixed_params([FIXTURE], cache, noise=0.5)
        blur2, _ = compute_fixed_params([FIXTURE], cache, noise=0.5)
        assert blur1 == blur2


class TestEndToEndLive:
    def test_full_denoise_then_levels_pipeline_on_real_crop(self, cache):
        """The exact sequence denoise_stack_frames runs per frame, unmocked:
        decode -> real denoise model -> real levels -- end to end on real data."""
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _crop_chw(bgr16, size=768)

        blur, noise = compute_fixed_params([FIXTURE], cache, noise=0.8)
        denoised = _sharpen_f32(chw, cache, blur=blur, noise=noise)
        leveled = apply_levels(denoised, exposure_ev=-0.5)

        assert leveled.shape == chw.shape
        assert float(leveled.max()) <= 1.0
        assert float(leveled.min()) >= 0.0
        # levels is a gain < 1.0 here (negative EV): leveled must be <= denoised everywhere
        assert bool(np.all(leveled <= denoised + 1e-6))


class TestSubjectMaskLive:
    """Real s_mask_l_flexible (Topaz Photo AI's own masked-sharpen model,
    not Meta's SAM/samenc/samdec) against the real spider fixture.
    Grounds the values measured live 2026-10-19 before this feature was
    built: max confidence 0.943, mean 0.0123 (a small subject against a
    mostly-empty wall)."""

    def test_real_mask_confidently_isolates_a_small_subject(self, cache):
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _to_chw_rgb_f32(bgr16)

        mask = subject_mask(chw, cache)

        assert mask.shape == chw.shape[1:]
        assert mask.dtype == np.float32
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0
        # a confident peak on the spider...
        assert float(mask.max()) > 0.8
        # ...against a mostly-empty wall, so the mean stays low
        assert float(mask.mean()) < 0.05

    def test_real_mask_is_reproducible(self, cache):
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _to_chw_rgb_f32(bgr16)

        first = subject_mask(chw, cache)
        second = subject_mask(chw, cache)
        assert np.array_equal(first, second)

    def test_real_masked_levels_lifts_subject_without_lifting_background(self, cache):
        """End-to-end through the real public function on real data: a
        positive subject_boost_ev must brighten the (small, dark) subject
        region far more than the mean of the rest of the frame."""
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _to_chw_rgb_f32(bgr16)
        mask = subject_mask(chw, cache)

        baseline = apply_levels(chw, exposure_ev=0.0)
        boosted = apply_levels_masked(chw, mask, exposure_ev=0.0, subject_boost_ev=1.0)

        subject_region = mask > 0.5
        background_region = mask < 0.1

        subject_gain = float(boosted[:, subject_region].mean() / (baseline[:, subject_region].mean() + 1e-6))
        background_gain = float(
            boosted[:, background_region].mean() / (baseline[:, background_region].mean() + 1e-6)
        )

        # Real measured value 2026-10-19: 1.695x (mask>0.5 includes partial-
        # blend pixels near the 0.5 boundary, and some near-peak subject
        # pixels clip at 1.0, both pulling the mean below the idealized
        # 2.0x boost) -- well above the near-unchanged background, which is
        # the actual property being proven, not a specific ideal ratio.
        assert subject_gain > 1.5
        assert background_gain < 1.1  # background stays close to unchanged
        assert subject_gain > background_gain * 1.5  # the real differential effect


class TestRefineMaskByColorLive:
    """Real spider fixture, real s_mask_l_flexible model: proves
    refine_mask_by_color's shadow-removal actually does something on real
    model output, not just on the synthetic fixtures in test_preclean.py."""

    @staticmethod
    def _raw_sigmoid_mask(chw: np.ndarray, cache: ModelCache) -> np.ndarray:
        """Reproduces subject_mask()'s own pre-refinement steps (resize to
        320, real model inference, sigmoid, resize back up) to get the real,
        *unrefined* mask as input for refine_mask_by_color -- this is how
        subject_mask() itself builds the value it hands to
        refine_mask_by_color, not a reimplementation of the function under
        test here."""
        _, h, w = chw.shape
        hwc_u8 = (np.transpose(chw, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        small = Image.fromarray(hwc_u8).resize((320, 320), Image.BILINEAR)
        in320 = np.transpose(np.array(small, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis]

        compiled = cache.load(MODELS_DIR / SUBJECT_MASK_MODEL)
        out = compiled({"netInput": in320})
        logits = next(iter(out.values()))[0, 0]
        sig = 1.0 / (1.0 + np.exp(-logits))

        mask_img = Image.fromarray((sig * 255.0).clip(0, 255).astype(np.uint8), mode="L")
        mask_full = mask_img.resize((w, h), Image.BILINEAR)
        return np.array(mask_full, dtype=np.float32) / 255.0

    def test_real_refinement_attenuates_the_raw_mask_without_changing_shape_or_dtype(self, cache):
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _to_chw_rgb_f32(bgr16)
        raw_prob = self._raw_sigmoid_mask(chw, cache)

        refined = refine_mask_by_color(chw, raw_prob)

        assert refined.shape == raw_prob.shape
        assert refined.dtype == np.float32
        assert float(refined.min()) >= 0.0
        assert float(refined.max()) <= 1.0
        # Real measured relationship (module docstring + the subject_mask
        # live test above): the shadow's near-background hue gets knocked
        # down while the confident subject peak (hue far from the wall) is
        # essentially untouched, so the refined mask's mean must drop below
        # the raw, unrefined mask's mean.
        assert float(refined.mean()) < float(raw_prob.mean())
        # the confident peak sits on the real subject, whose hue is far
        # enough from the wall's (module docstring: ~16deg vs the 20deg
        # ramp ceiling) to be ungated -- it stays confident, even though the
        # blur alone (applied everywhere, gated or not) pulls every local
        # peak down a bit by mixing in its immediate neighbours.
        assert float(refined.max()) > 0.7

    def test_consistent_with_subject_mask_which_calls_it_internally(self, cache):
        """subject_mask() is documented as calling refine_mask_by_color() on
        its own raw sigmoid output as its last step; calling it explicitly
        here must reproduce subject_mask()'s own real result exactly."""
        bgr16 = decode_raw_to_bgr16(FIXTURE)
        chw = _to_chw_rgb_f32(bgr16)
        raw_prob = self._raw_sigmoid_mask(chw, cache)

        via_subject_mask = subject_mask(chw, cache)
        via_direct_call = refine_mask_by_color(chw, raw_prob)

        assert np.array_equal(via_subject_mask, via_direct_call)
