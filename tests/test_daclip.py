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


class TestEmbedPath:
    """embed_path() loads a real file from disk (draft-decoded when it's a big JPEG); these use a real Pillow image and a mocked session,
    checking what actually reached the model, not just that nothing raised."""

    def _solid_jpeg(self, tmp_path, size, color, name="a.jpg"):
        from PIL import Image as PILImage

        p = tmp_path / name
        PILImage.new("RGB", size, color).save(p, "JPEG", quality=95)
        return p

    def _preprocessed_mean_rgb(self, session) -> np.ndarray:
        """Undo preprocess()'s CLIP normalisation on the array the mock session actually received, back to ~[0,1] RGB. embed()'s input dict
        is keyed by session.get_inputs()[0].name, which on a MagicMock is itself a mock object, not a readable string -- so take the one
        value the dict holds rather than assume a key name."""
        [arr] = list(session.run.call_args[0][1].values())
        rgb01 = arr[0] * daclip._STD + daclip._MEAN
        return rgb01.mean(axis=(0, 1))

    def test_a_large_jpeg_above_the_draft_threshold_still_embeds_the_right_color(self, tmp_path):
        p = self._solid_jpeg(tmp_path, (2000, 2000), (200, 40, 10))                 # well above _DRAFT_SIZE: draft() actually kicks in
        sess = _mock_session(self._fake_vecs_for_route())
        daclip.embed_path(p, sess)
        mean_rgb = self._preprocessed_mean_rgb(sess)
        np.testing.assert_allclose(mean_rgb, np.array([200, 40, 10]) / 255.0, atol=0.03)   # JPEG + draft-scale decode: allow real compression slack

    def test_a_small_jpeg_under_the_threshold_is_unaffected_by_draft(self, tmp_path):
        p = self._solid_jpeg(tmp_path, (64, 64), (10, 200, 40))                     # below _DRAFT_SIZE: draft() is a no-op here
        sess = _mock_session(self._fake_vecs_for_route())
        daclip.embed_path(p, sess)
        mean_rgb = self._preprocessed_mean_rgb(sess)
        np.testing.assert_allclose(mean_rgb, np.array([10, 200, 40]) / 255.0, atol=0.03)

    def test_a_non_jpeg_file_still_embeds_correctly(self, tmp_path):
        from PIL import Image as PILImage

        p = tmp_path / "a.png"
        PILImage.new("RGB", (900, 900), (5, 5, 220)).save(p, "PNG")                 # draft() only applies to JPEG; must not raise or corrupt
        sess = _mock_session(self._fake_vecs_for_route())
        daclip.embed_path(p, sess)
        mean_rgb = self._preprocessed_mean_rgb(sess)
        np.testing.assert_allclose(mean_rgb, np.array([5, 5, 220]) / 255.0, atol=0.01)

    def test_the_array_passed_to_the_model_is_bgr_order(self, tmp_path):
        """embed_path converts PIL's RGB to BGR before preprocess() (which itself flips BGR->RGB): a red source should reach preprocess()
        as blue-channel-first, the same contract embed()/preprocess() already assume."""
        p = self._solid_jpeg(tmp_path, (300, 300), (255, 0, 0))                     # pure red
        sess = _mock_session(self._fake_vecs_for_route())
        daclip.embed_path(p, sess)
        mean_rgb = self._preprocessed_mean_rgb(sess)                                # preprocess() flips back to RGB for the model
        assert mean_rgb[0] > 0.9 and mean_rgb[1] < 0.1 and mean_rgb[2] < 0.1        # still reads as red once both flips cancel out

    def _fake_vecs_for_route(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.random((2, 512), dtype=np.float64).astype(np.float32)


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
