"""Unit tests for bio_automation.sr_model (hermetic -- no real model loaded)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bio_automation import sr_model


def _make_bgr(size: int = 128) -> np.ndarray:
    """A non-uniform BGR uint8 image: distinct per-channel values so a channel
    swap (BGR<->RGB) would be caught by an exact-value assertion, not just a
    shape check. Blue=200, Green=100, Red=10 at every pixel."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # B
    img[:, :, 1] = 100  # G
    img[:, :, 2] = 10   # R
    return img


def _mock_session_identity() -> MagicMock:
    """A recording stub: echoes back exactly the [1,3,H,W] float32 tile it was
    fed, scaled 2x via nearest-neighbour repeat -- so the test can compute the
    expected output independently and compare exact values, not just shape."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="input")]

    def _run(_output_names, feed):
        tile = next(iter(feed.values()))  # [1, C, H, W]
        upsampled = np.repeat(np.repeat(tile, 2, axis=2), 2, axis=3)
        return [upsampled]

    sess.run.side_effect = _run
    return sess


class TestUpscale2x:
    def test_output_shape_is_2x(self):
        img = _make_bgr(128)
        sess = _mock_session_identity()
        out = sr_model.upscale_2x(img, sess)
        assert out.shape == (256, 256, 3)

    def test_output_dtype_uint8(self):
        img = _make_bgr(128)
        sess = _mock_session_identity()
        out = sr_model.upscale_2x(img, sess)
        assert out.dtype == np.uint8

    def test_bgr_channels_not_swapped(self):
        """The real risk in this function: two channel flips (BGR->RGB going
        in, RGB->BGR coming out) that must cancel out exactly. A solid
        (B=200, G=100, R=10) tile identity-echoed through the mock session
        must come back as the SAME (B=200, G=100, R=10), not swapped."""
        img = _make_bgr(128)
        sess = _mock_session_identity()
        out = sr_model.upscale_2x(img, sess)
        assert out[0, 0, 0] == 200  # B
        assert out[0, 0, 1] == 100  # G
        assert out[0, 0, 2] == 10   # R

    def test_different_inputs_produce_different_outputs(self):
        """Positive twin for the exact-value test above: a different solid
        color must come back different, not some constant/broken passthrough."""
        sess = _mock_session_identity()
        out_a = sr_model.upscale_2x(_make_bgr(128), sess)
        img_b = np.zeros((128, 128, 3), dtype=np.uint8)
        img_b[:, :, 0] = 5
        img_b[:, :, 1] = 250
        img_b[:, :, 2] = 80
        out_b = sr_model.upscale_2x(img_b, sess)
        assert not np.array_equal(out_a, out_b)

    def test_uses_get_session_when_none_passed(self, monkeypatch):
        """Positive-for-the-default-path twin: omitting *session* must reach
        the process-wide singleton, not silently skip inference."""
        sess = _mock_session_identity()
        monkeypatch.setattr(sr_model, "get_session", lambda: sess)
        out = sr_model.upscale_2x(_make_bgr(128))
        assert out.shape == (256, 256, 3)
        assert sess.run.called


class TestLoadSession:
    def test_missing_model_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="bio-sr2x-v1 model not found"):
            sr_model.load_session(cache_dir=tmp_path)

    def test_missing_cuda_dirs_raises_runtime_error_not_silent_cpu(self, tmp_path, monkeypatch):
        """GPU-only contract (owner rule, same as daclip): a missing CUDA DLL
        folder must raise, never fall back to CPU silently."""
        model_dir = tmp_path / "bio-sr2x-v1"
        model_dir.mkdir()
        (model_dir / sr_model._ONNX_NAME).write_bytes(b"not a real onnx file")
        monkeypatch.setattr(sr_model, "_MISSING_CUDA_DIRS", [Path(r"C:\nonexistent\cuda")])
        with pytest.raises(RuntimeError, match="CUDA DLL folders not found"):
            sr_model.load_session(cache_dir=tmp_path)
