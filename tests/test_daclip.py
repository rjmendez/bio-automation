"""Unit tests for bio_automation.daclip (hermetic – no real model loaded)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bio_automation import daclip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bgr_gray(value: int = 128, size: int = 300) -> np.ndarray:
    """Return a solid-gray BGR uint8 image of *size*×*size*."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _mock_session(output: np.ndarray) -> MagicMock:
    """Return a mock ort.InferenceSession whose run() returns *output*."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="image")]
    sess.run.return_value = [output]
    return sess


# ---------------------------------------------------------------------------
# preprocess()
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shape(self):
        img = _make_bgr_gray(128)
        out = daclip.preprocess(img)
        assert out.shape == (1, 224, 224, 3)

    def test_output_dtype(self):
        img = _make_bgr_gray(128)
        out = daclip.preprocess(img)
        assert out.dtype == np.float32

    def test_values_in_normalised_range(self):
        """After CLIP normalisation, pixel values should fall roughly in [-3, 3]."""
        img = _make_bgr_gray(0)   # darkest image → most negative after norm
        out = daclip.preprocess(img)
        assert out.min() >= -3.0 and out.max() <= 3.0

    def test_all_ones_not_normalised(self):
        """An all-white image should NOT produce zero-mean output (positive shift expected)."""
        img = _make_bgr_gray(255)
        out = daclip.preprocess(img)
        # After normalisation white maps to (1 - mean)/std which is clearly > 0
        assert out.mean() > 0.0

    def test_gray128_mean_near_zero(self):
        """A mid-gray image is close to the CLIP mean → per-channel residual ≈ 0."""
        # CLIP mean ≈ [0.481, 0.458, 0.408] in [0,1]; 128/255 ≈ 0.502
        # residuals are small but not exactly zero; test they are in (−1, 1)
        img = _make_bgr_gray(128)
        out = daclip.preprocess(img)
        assert abs(float(out.mean())) < 1.0

    def test_different_gray_values_differ(self):
        """Positive twin: different gray intensities must produce different outputs."""
        dark = daclip.preprocess(_make_bgr_gray(50))
        light = daclip.preprocess(_make_bgr_gray(200))
        assert not np.allclose(dark, light)


# ---------------------------------------------------------------------------
# embed()
# ---------------------------------------------------------------------------

class TestEmbed:
    def _fake_vecs(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((2, 512), dtype=np.float64).astype(np.float32)

    def test_returns_two_vectors(self):
        sess = _mock_session(self._fake_vecs())
        content, degrad = daclip.embed(_make_bgr_gray(), sess)
        assert content.shape == (512,)
        assert degrad.shape == (512,)

    def test_content_and_degrad_are_distinct(self):
        vecs = self._fake_vecs()
        sess = _mock_session(vecs)
        content, degrad = daclip.embed(_make_bgr_gray(), sess)
        assert not np.allclose(content, degrad)

    def test_content_matches_row0(self):
        vecs = self._fake_vecs()
        sess = _mock_session(vecs)
        content, _ = daclip.embed(_make_bgr_gray(), sess)
        np.testing.assert_array_equal(content, vecs[0])

    def test_degrad_matches_row1(self):
        vecs = self._fake_vecs()
        sess = _mock_session(vecs)
        _, degrad = daclip.embed(_make_bgr_gray(), sess)
        np.testing.assert_array_equal(degrad, vecs[1])

    def test_session_called_once(self):
        sess = _mock_session(self._fake_vecs())
        daclip.embed(_make_bgr_gray(), sess)
        assert sess.run.call_count == 1

    def test_wrong_session_not_called_when_not_invoked(self):
        """Negative twin: a separate session that must NOT be called."""
        other_sess = _mock_session(self._fake_vecs())
        real_sess = _mock_session(self._fake_vecs())
        daclip.embed(_make_bgr_gray(), real_sess)
        assert other_sess.run.call_count == 0


# ---------------------------------------------------------------------------
# Routing logic (norm → route label)
# ---------------------------------------------------------------------------

from pipeline.stages.daclip_triage import _route_from_norm  # noqa: E402


class TestRouteFromNorm:
    # --- skip band ---
    def test_very_low_norm_skips(self):
        assert _route_from_norm(0.1) == "skip"

    def test_zero_norm_skips(self):
        assert _route_from_norm(0.0) == "skip"

    def test_just_below_skip_boundary_skips(self):
        assert _route_from_norm(0.299) == "skip"

    # --- sharpen band ---
    def test_mid_norm_sharpens(self):
        assert _route_from_norm(0.5) == "sharpen"

    def test_at_lower_sharpen_boundary_sharpens(self):
        assert _route_from_norm(0.3) == "sharpen"

    def test_at_upper_sharpen_boundary_sharpens(self):
        assert _route_from_norm(0.6) == "sharpen"

    # --- sharpen_strong band ---
    def test_high_norm_sharpen_strong(self):
        assert _route_from_norm(0.8) == "sharpen_strong"

    def test_just_above_sharpen_boundary_sharpen_strong(self):
        assert _route_from_norm(0.601) == "sharpen_strong"

    def test_very_high_norm_sharpen_strong(self):
        assert _route_from_norm(10.0) == "sharpen_strong"

    # --- positive/negative twin pairs ---
    def test_skip_is_not_sharpen(self):
        assert _route_from_norm(0.1) != "sharpen"

    def test_sharpen_is_not_skip(self):
        assert _route_from_norm(0.5) != "skip"

    def test_sharpen_strong_is_not_skip(self):
        assert _route_from_norm(0.8) != "skip"
