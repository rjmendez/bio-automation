"""Unit tests for bio_automation.preclean (hermetic -- no real raws decoded,
no real OpenVINO models loaded)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile

from bio_automation import preclean
from bio_automation.preclean import (
    apply_levels,
    compute_fixed_params,
    decode_raw_to_bgr16,
    denoise_stack_frames,
    ev_to_linear_gain,
)


class FakeRawContext:
    """Stands in for rawpy.imread()'s context manager; records postprocess() kwargs."""

    def __init__(self, result: np.ndarray, record: list[dict]):
        self._result = result
        self._record = record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs) -> np.ndarray:
        self._record.append(kwargs)
        return self._result


def _fake_imread(result: np.ndarray, record: list[dict]):
    def imread(path):
        return FakeRawContext(result, record)
    return imread


# ---------------------------------------------------------------------------
# ev_to_linear_gain
# ---------------------------------------------------------------------------

class TestEvToLinearGain:
    def test_zero_ev_is_unity(self):
        assert ev_to_linear_gain(0.0) == 1.0

    def test_minus_one_ev_halves(self):
        assert ev_to_linear_gain(-1.0) == 0.5

    def test_plus_one_ev_doubles(self):
        assert ev_to_linear_gain(1.0) == 2.0

    def test_clamped_to_lower_bound(self):
        assert ev_to_linear_gain(-10.0) == 0.25

    def test_clamped_to_upper_bound(self):
        assert ev_to_linear_gain(10.0) == 8.0


# ---------------------------------------------------------------------------
# decode_raw_to_bgr16 -- proves exposure is NOT applied at decode time
# ---------------------------------------------------------------------------

class TestDecodeRawToBgr16:
    def test_reverses_channels_rgb_to_bgr(self, monkeypatch):
        rgb16 = np.zeros((2, 2, 3), dtype=np.uint16)
        rgb16[0, 0] = [100, 200, 300]  # R, G, B
        record: list[dict] = []
        monkeypatch.setattr(preclean.rawpy, "imread", _fake_imread(rgb16, record))

        bgr16 = decode_raw_to_bgr16(Path("fake.NEF"))
        assert list(bgr16[0, 0]) == [300, 200, 100]

    def test_postprocess_kwargs_exact_and_no_exposure_shift(self, monkeypatch):
        rgb16 = np.zeros((1, 1, 3), dtype=np.uint16)
        record: list[dict] = []
        monkeypatch.setattr(preclean.rawpy, "imread", _fake_imread(rgb16, record))

        decode_raw_to_bgr16(Path("fake.NEF"))

        assert len(record) == 1
        assert record[0] == {"output_bps": 16, "no_auto_bright": True, "use_camera_wb": True}
        assert "exp_shift" not in record[0]


# ---------------------------------------------------------------------------
# apply_levels
# ---------------------------------------------------------------------------

class TestApplyLevels:
    def test_zero_ev_is_noop(self):
        img = np.full((1, 2, 2), 0.5, dtype=np.float32)
        out = apply_levels(img, 0.0)
        assert np.array_equal(out, img)

    def test_positive_ev_scales_up(self):
        img = np.full((1, 2, 2), 0.25, dtype=np.float32)
        out = apply_levels(img, 1.0)  # gain 2.0
        assert np.allclose(out, 0.5)

    def test_clips_to_one_not_beyond(self):
        img = np.full((1, 2, 2), 0.9, dtype=np.float32)
        out = apply_levels(img, 1.0)  # 0.9 * 2 = 1.8 -> must clip to 1.0 exactly
        assert float(out.max()) == 1.0
        assert float(out.min()) == 1.0

    def test_negative_ev_scales_down_and_never_clips_at_zero_edge(self):
        img = np.full((1, 2, 2), 0.5, dtype=np.float32)
        out = apply_levels(img, -1.0)  # gain 0.5
        assert np.allclose(out, 0.25)


# ---------------------------------------------------------------------------
# compute_fixed_params
# ---------------------------------------------------------------------------

class TestComputeFixedParams:
    def test_empty_raw_paths_raises(self):
        with pytest.raises(ValueError, match="requires at least one raw path"):
            compute_fixed_params([], cache=MagicMock())

    def test_uses_middle_frame_only(self, monkeypatch):
        paths = [Path(f"f{i}.NEF") for i in range(5)]  # middle index = 2
        decoded_calls: list[Path] = []

        def fake_decode(p):
            decoded_calls.append(p)
            return np.zeros((1, 1, 3), dtype=np.uint16)

        monkeypatch.setattr(preclean, "decode_raw_to_bgr16", fake_decode)
        monkeypatch.setattr(preclean, "_estimate_sharpen_params",
                             lambda chw, cache: (0.1, 0.42, 0.3, 0.0))

        compute_fixed_params(paths, cache=MagicMock())
        assert decoded_calls == [paths[2]]

    def test_returns_estimated_blur_and_clamped_noise(self, monkeypatch):
        monkeypatch.setattr(preclean, "decode_raw_to_bgr16",
                             lambda p: np.zeros((1, 1, 3), dtype=np.uint16))
        monkeypatch.setattr(preclean, "_estimate_sharpen_params",
                             lambda chw, cache: (0.1, 0.77, 0.3, 0.0))

        blur, noise = compute_fixed_params([Path("f.NEF")], cache=MagicMock(), noise=1.5)
        assert blur == 0.77
        assert noise == 1.0  # clamped from 1.5

    def test_negative_noise_clamped_to_zero(self, monkeypatch):
        monkeypatch.setattr(preclean, "decode_raw_to_bgr16",
                             lambda p: np.zeros((1, 1, 3), dtype=np.uint16))
        monkeypatch.setattr(preclean, "_estimate_sharpen_params",
                             lambda chw, cache: (0.1, 0.5, 0.3, 0.0))

        _, noise = compute_fixed_params([Path("f.NEF")], cache=MagicMock(), noise=-0.5)
        assert noise == 0.0


# ---------------------------------------------------------------------------
# denoise_stack_frames -- the core claim: identical (blur, noise, exposure_ev)
# reused across every frame, never recomputed per frame.
# ---------------------------------------------------------------------------

class TestDenoiseStackFrames:
    def _setup(self, monkeypatch, n=3):
        """n distinguishable fake raws; _sharpen_f32 is an identity recording
        stub so output values are traceable back to input + levels only."""
        paths = [Path(f"f{i}.NEF") for i in range(n)]
        decode_calls: list[Path] = []

        def fake_decode(p):
            decode_calls.append(p)
            idx = paths.index(p)
            # distinguishable per frame: value = (idx+1)/10, same for all channels
            arr = np.full((1, 1, 3), int((idx + 1) / 10 * 65535), dtype=np.uint16)
            return arr

        monkeypatch.setattr(preclean, "decode_raw_to_bgr16", fake_decode)
        monkeypatch.setattr(preclean, "_estimate_sharpen_params",
                             lambda chw, cache: (0.1, 0.6, 0.3, 0.0))

        sharpen_calls: list[tuple] = []

        def fake_sharpen_f32(chw, cache, blur, noise):
            sharpen_calls.append((blur, noise))
            return chw  # identity

        monkeypatch.setattr(preclean, "_sharpen_f32", fake_sharpen_f32)
        return paths, decode_calls, sharpen_calls

    def test_same_blur_noise_pair_used_for_every_frame(self, tmp_path, monkeypatch):
        paths, _, sharpen_calls = self._setup(monkeypatch, n=4)
        denoise_stack_frames(paths, tmp_path, noise=0.9, cache=MagicMock())

        assert len(sharpen_calls) == 4
        assert len(set(sharpen_calls)) == 1  # every call got the exact same (blur, noise)
        assert sharpen_calls[0] == (0.6, 0.9)

    def test_writes_one_tif_per_raw_named_by_stem(self, tmp_path, monkeypatch):
        paths, _, _ = self._setup(monkeypatch, n=3)
        written = denoise_stack_frames(paths, tmp_path, cache=MagicMock())

        assert [p.name for p in written] == ["f0.tif", "f1.tif", "f2.tif"]
        for p in written:
            assert p.exists()

    def test_output_pixel_value_matches_decode_times_levels_gain(self, tmp_path, monkeypatch):
        """End-to-end numeric check: with identity _sharpen_f32, the written
        TIFF must equal decode_raw_to_bgr16's output scaled by the exact
        levels gain -- not some other transform."""
        paths, _, _ = self._setup(monkeypatch, n=1)
        written = denoise_stack_frames(paths, tmp_path, exposure_ev=1.0, cache=MagicMock())

        result = tifffile.imread(str(written[0]))
        # frame 0 -> raw value int(0.1 * 65535); decode reverses BGR but all
        # channels are equal here, so channel order doesn't matter for the value
        expected_input_u16 = int(0.1 * 65535)
        expected_f32 = min((expected_input_u16 / 65535.0) * 2.0, 1.0)  # levels gain=2.0 (EV=1.0)
        expected_u16 = round(expected_f32 * 65535.0)
        assert int(result[0, 0, 0]) == expected_u16
