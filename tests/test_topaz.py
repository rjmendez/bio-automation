"""Unit tests for bio_automation.topaz (hermetic - no real models loaded)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from bio_automation import topaz
from bio_automation.topaz import (
    ModelCache,
    _is_complete_zip,
    _cosine_window,
    _run_tiled,
    _single_output,
    upscale_2x,
    enhance,
    extract_tz,
    sharpen,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tz(dest: Path, members: dict, name: str = "test.tz") -> Path:
    """Create an unencrypted ZIP with *members* {internal_path: bytes}.

    Python's zipfile cannot write ZipCrypto; reading with pwd= silently ignores
    the password for unencrypted entries, so these fixtures test extraction
    logic without requiring a third-party crypto library.
    """
    tz_path = dest / name
    with zipfile.ZipFile(tz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name_in_zip, data in members.items():
            zf.writestr(name_in_zip, data)
    return tz_path


def _make_truncated_tz(dest: Path, name: str = "trunc.tz") -> Path:
    """Create a file with a local-file-header signature but no EOCD."""
    path = dest / name
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    return path


def _make_fake_ov(fake_compiled):
    """Return (ov_mock, core_mock) where core.compile_model returns fake_compiled."""
    core_inst = MagicMock()
    core_inst.compile_model.return_value = fake_compiled
    ov_mod = MagicMock()
    ov_mod.Core.return_value = core_inst
    return ov_mod, core_inst


# ---------------------------------------------------------------------------
# 0. _is_complete_zip
# ---------------------------------------------------------------------------

class TestIsCompleteZip:
    def test_valid_zip_returns_true(self, tmp_path):
        tz = _make_tz(tmp_path, {"f.bin": b"hello"})
        assert _is_complete_zip(tz) is True

    def test_truncated_file_returns_false(self, tmp_path):
        trunc = _make_truncated_tz(tmp_path)
        assert _is_complete_zip(trunc) is False

    def test_empty_file_returns_false(self, tmp_path):
        empty = tmp_path / "empty.tz"
        empty.write_bytes(b"")
        assert _is_complete_zip(empty) is False


# ---------------------------------------------------------------------------
# 1. extract_tz
# ---------------------------------------------------------------------------

class TestExtractTz:
    def test_extracts_and_flattens(self, tmp_path):
        members = {
            "subdir/model.xml": b"<xml/>",
            "subdir/model.bin": b"\x00\x01\x02",
        }
        tz = _make_tz(tmp_path, members)
        dest = tmp_path / "out"
        result = extract_tz(tz, dest)

        assert set(result.keys()) == {"model.xml", "model.bin"}
        assert result["model.xml"].read_bytes() == b"<xml/>"
        assert result["model.bin"].read_bytes() == b"\x00\x01\x02"
        assert result["model.xml"].parent == dest

    def test_nested_subdirs_flattened(self, tmp_path):
        members = {"a/b/c/weights.bin": b"data"}
        tz = _make_tz(tmp_path, members)
        result = extract_tz(tz, tmp_path / "out")
        assert "weights.bin" in result
        assert result["weights.bin"].name == "weights.bin"

    def test_truncated_zip_raises_runtime_error(self, tmp_path):
        trunc = _make_truncated_tz(tmp_path)
        with pytest.raises(RuntimeError, match=r"not yet fully downloaded"):
            extract_tz(trunc, tmp_path / "out")

    def test_complete_zip_does_not_raise(self, tmp_path):
        """Positive twin: a complete archive must NOT raise RuntimeError."""
        tz = _make_tz(tmp_path, {"ok.bin": b"\xff"})
        result = extract_tz(tz, tmp_path / "out")
        assert "ok.bin" in result


# ---------------------------------------------------------------------------
# 2. ModelCache - cache miss / hit / LRU / persistent / truncated guard
# ---------------------------------------------------------------------------

class TestModelCache:
    def _tz_with_xml(self, tmp_path: Path, stem: str = "test") -> Path:
        return _make_tz(tmp_path,
                        {"mymodel.xml": b"<net/>", "mymodel.bin": b"\xff" * 8},
                        name=f"{stem}.tz")

    def test_cache_miss_calls_compile(self, tmp_path, monkeypatch):
        tz = self._tz_with_xml(tmp_path)
        fc = MagicMock(name="compiled")
        ov_mod, core_inst = _make_fake_ov(fc)
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        result = ModelCache(cache_dir=tmp_path / "cache").load(tz)

        assert result is fc
        core_inst.compile_model.assert_called_once()

    def test_cache_hit_skips_recompile(self, tmp_path, monkeypatch):
        tz = self._tz_with_xml(tmp_path)
        fc = MagicMock(name="compiled")
        ov_mod, core_inst = _make_fake_ov(fc)
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        cache = ModelCache(cache_dir=tmp_path / "cache")
        r1 = cache.load(tz)
        r2 = cache.load(tz)

        assert r1 is fc
        assert r2 is fc
        assert core_inst.compile_model.call_count == 1  # compiled exactly once

    def test_lru_eviction(self, tmp_path, monkeypatch):
        """With max_models=1, loading a second model evicts the first from memory."""
        tz_a = _make_tz(tmp_path, {"m1.xml": b"<net/>", "m1.bin": b"\x00"}, name="m1.tz")
        tz_b = _make_tz(tmp_path, {"m2.xml": b"<net/>", "m2.bin": b"\x00"}, name="m2.tz")

        ov_mod = MagicMock()
        ov_mod.Core.return_value.compile_model.side_effect = lambda m, d: MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        cache = ModelCache(max_models=1, cache_dir=tmp_path / "cache")
        cache.load(tz_a)
        assert len(cache._lru) == 1
        assert tz_a.stem in cache._lru

        cache.load(tz_b)
        assert len(cache._lru) == 1      # still 1: tz_a was evicted
        assert tz_b.stem in cache._lru
        assert tz_a.stem not in cache._lru

    def test_context_manager_clears_lru(self, tmp_path, monkeypatch):
        tz = self._tz_with_xml(tmp_path)
        ov_mod, _ = _make_fake_ov(MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        with ModelCache(cache_dir=tmp_path / "cache") as cache:
            cache.load(tz)
            assert len(cache._lru) == 1
        assert len(cache._lru) == 0

    def test_persistent_cache_skips_extraction(self, tmp_path, monkeypatch):
        """If the XML already exists on disk, extract_tz must NOT be called."""
        tz = self._tz_with_xml(tmp_path)
        cache_dir = tmp_path / "cache"
        model_dir = cache_dir / tz.stem
        model_dir.mkdir(parents=True)
        (model_dir / "mymodel.xml").write_bytes(b"<net/>")
        (model_dir / "mymodel.bin").write_bytes(b"\xff" * 8)

        spy = MagicMock(side_effect=AssertionError("extract_tz must not be called on disk hit"))
        monkeypatch.setattr(topaz, "extract_tz", spy)

        fc = MagicMock(name="compiled")
        ov_mod, _ = _make_fake_ov(fc)
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        result = ModelCache(cache_dir=cache_dir).load(tz)
        assert result is fc
        spy.assert_not_called()

    def test_truncated_tz_propagates_runtime_error(self, tmp_path, monkeypatch):
        """A still-downloading .tz must surface RuntimeError, not silently fail."""
        trunc = _make_truncated_tz(tmp_path, name="slb-v1-fp32-512x512-ov.tz")
        ov_mod, _ = _make_fake_ov(MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "openvino", ov_mod)

        cache = ModelCache(cache_dir=tmp_path / "cache")
        with pytest.raises(RuntimeError, match=r"not yet fully downloaded"):
            cache.load(trunc)


# ---------------------------------------------------------------------------
# 3. _run_tiled - shape and value range
# ---------------------------------------------------------------------------

class TestRunTiled:
    def _identity_feed(self, tile: np.ndarray, _compiled) -> np.ndarray:
        return tile.copy()

    def test_output_shape_scale1(self):
        img = np.random.rand(3, 64, 64).astype(np.float32) * 0.8 + 0.1
        out = _run_tiled(None, img, 32, 32, 4, 1, self._identity_feed, "out")
        assert out.shape == (3, 64, 64)

    def test_output_shape_scale2(self):
        img = np.random.rand(3, 32, 32).astype(np.float32) * 0.5

        def feed2x(tile, _):
            return np.repeat(np.repeat(tile, 2, axis=1), 2, axis=2)

        out = _run_tiled(None, img, 16, 16, 2, 2, feed2x, "out")
        assert out.shape == (3, 64, 64)

    def test_values_in_unit_range(self):
        img = np.ones((3, 48, 48), dtype=np.float32) * 0.6
        out = _run_tiled(None, img, 24, 24, 4, 1, self._identity_feed, "out")
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_out_of_range_values_are_clamped(self):
        """feed_fn values > 1.0 must be clamped to 1.0, not zero."""
        def hot_feed(tile, _):
            return np.ones_like(tile) * 2.5

        img = np.zeros((3, 32, 32), dtype=np.float32)
        out = _run_tiled(None, img, 16, 16, 2, 1, hot_feed, "out")
        assert float(out.max()) <= 1.0
        assert float(out.max()) > 0.0  # positive twin: clamped to 1, not zeroed


# ---------------------------------------------------------------------------
# 4. enhance() - PIL I/O and correct operation dispatch
# ---------------------------------------------------------------------------

class TestEnhance:
    def _write_jpeg(self, path: Path, size=(64, 64)):
        img = Image.fromarray(np.full((*size, 3), 128, dtype=np.uint8))
        img.save(str(path), "JPEG", quality=95)

    def test_stock_mode_writes_jpeg_same_size(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src)

        def fake_sharpen(img_bgr, cache, blur=0.5, noise=0.0):
            assert img_bgr.dtype == np.uint8
            assert img_bgr.ndim == 3
            return img_bgr.copy()

        monkeypatch.setattr(topaz, "sharpen", fake_sharpen)
        monkeypatch.setattr(topaz, "upscale_2x",
                            MagicMock(side_effect=AssertionError("upscale_2x must not be called in stock mode")))

        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))

        assert dst.exists()
        result = Image.open(dst)
        assert result.format == "JPEG"
        assert result.size == (64, 64)

    def test_inat_mode_copies_source_unchanged_without_model_calls(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src)
        boom = MagicMock(side_effect=AssertionError("inat mode must not run any model"))
        monkeypatch.setattr(topaz, "sharpen", boom)
        monkeypatch.setattr(topaz, "upscale_2x", boom)

        enhance(src, dst, mode="inat", cache=MagicMock(spec=ModelCache))

        assert dst.read_bytes() == src.read_bytes()
        assert boom.call_count == 0

    def test_upscale_mode_calls_sharpen_then_upscale(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src, size=(32, 32))

        sharpen_calls = []
        upscale_calls = []

        def spy_sharpen(img_bgr, cache, blur=0.5, noise=0.0):
            sharpen_calls.append(1)
            return img_bgr.copy()

        def spy_upscale(img_bgr, cache):
            upscale_calls.append(1)
            h, w = img_bgr.shape[:2]
            return np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        monkeypatch.setattr(topaz, "sharpen", spy_sharpen)
        monkeypatch.setattr(topaz, "upscale_2x", spy_upscale)

        enhance(src, dst, mode="upscale", cache=MagicMock(spec=ModelCache))
        assert sharpen_calls == [1]
        assert upscale_calls == [1]
        assert Image.open(dst).size == (64, 64)  # 2x of 32x32

    def test_unknown_mode_raises_value_error(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        self._write_jpeg(src)
        monkeypatch.setattr(topaz, "sharpen", lambda *a, **k: a[0].copy())

        with pytest.raises(ValueError, match=r"Unknown mode"):
            enhance(src, src, mode="bogus", cache=MagicMock(spec=ModelCache))

    def test_uses_pil_not_cv2(self, tmp_path, monkeypatch):
        """PIL Image.open must be the I/O path; cv2 must not be imported."""
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src)

        open_calls = []
        _real_open = Image.open

        def spy_open(path, *args, **kwargs):
            open_calls.append(str(path))
            return _real_open(path, *args, **kwargs)

        monkeypatch.setattr(Image, "open", spy_open)
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img.copy())

        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))
        assert any(str(src) in p for p in open_calls), "PIL Image.open not called with src"

        import sys
        assert "cv2" not in sys.modules, "cv2 must not be imported by topaz"

    def test_exif_and_icc_profile_survive(self, tmp_path, monkeypatch):
        from PIL import ImageCms

        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        exif = Image.Exif()
        exif[0x010F] = "NIKON CORPORATION"      # Make
        exif[0x0110] = "NIKON Z 7_2"            # Model
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(
            str(src), "JPEG", quality=95, exif=exif.tobytes(), icc_profile=icc)
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img.copy())

        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))

        out = Image.open(dst)
        assert out.getexif()[0x010F] == "NIKON CORPORATION"
        assert out.getexif()[0x0110] == "NIKON Z 7_2"
        assert out.info["icc_profile"] == icc

    def test_source_without_exif_or_icc_stays_without(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src)
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img.copy())

        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))

        out = Image.open(dst)
        assert dict(out.getexif()) == {}
        assert "icc_profile" not in out.info

    def test_disk_guard_raises_on_low_space(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        self._write_jpeg(src)

        monkeypatch.setattr(
            topaz.shutil, "disk_usage",
            lambda _: MagicMock(free=int(0.5 * 1e9))   # 0.5 GB
        )

        with pytest.raises(OSError, match=r"(?i)(disk space|insufficient)"):
            enhance(src, src, mode="stock", cache=MagicMock(spec=ModelCache),
                    min_free_gb=5.0)


# ---------------------------------------------------------------------------
# 5. Blend window and border handling (regression: ramp window blacked out
#    the first output row/column)
# ---------------------------------------------------------------------------

class TestBlendWindow:
    def test_window_is_symmetric_positive_bump(self):
        win = _cosine_window(512)
        assert float(win.min()) > 0.0
        np.testing.assert_allclose(win, win[::-1], atol=1e-6)
        assert int(np.argmax(win)) in (255, 256)
        assert float(win[0]) < 1e-4 and float(win[255]) > 0.99

    def test_constant_image_identity_keeps_value_at_every_pixel(self):
        img = np.full((3, 48, 40), 0.6, dtype=np.float32)
        out = _run_tiled(None, img, 24, 24, 8, 1, lambda t, _: t.copy())
        assert out.shape == (3, 48, 40)
        # corner, first row/col (the old bug) and an interior pixel alike
        assert float(out[0, 0, 0]) == pytest.approx(0.6, abs=1e-5)
        assert float(out[:, 0, :].min()) == pytest.approx(0.6, abs=1e-5)
        assert float(out[:, :, 0].min()) == pytest.approx(0.6, abs=1e-5)
        assert float(out[1, 20, 20]) == pytest.approx(0.6, abs=1e-5)
        assert float(out.max()) == pytest.approx(0.6, abs=1e-5)

    def test_constant_image_scale2_keeps_value_in_corners(self):
        img = np.full((3, 32, 32), 0.3, dtype=np.float32)

        def feed2x(t, _):
            return np.repeat(np.repeat(t, 2, axis=1), 2, axis=2)

        out = _run_tiled(None, img, 16, 16, 4, 2, feed2x)
        assert float(out[0, 0, 0]) == pytest.approx(0.3, abs=1e-5)
        assert float(out[2, -1, -1]) == pytest.approx(0.3, abs=1e-5)

    def test_hot_tile_is_clipped_before_blending(self):
        """A tile returning 5.0 contributes 1.0 (not 5.0) to the weighted mean."""
        calls = {"n": 0}

        def feed(tile, _):
            calls["n"] += 1
            return np.full_like(tile, 5.0 if calls["n"] == 1 else 0.0)

        img = np.zeros((1, 16, 24), dtype=np.float32)
        out = _run_tiled(None, img, 16, 16, 8, 1, feed)
        # tiles start at x=0 and x=8; column 12 has equal window weight from both
        # (tile0 local col 12, tile1 local col 4 -> weights differ), so compute exactly
        win = _cosine_window(16)
        w0, w1 = float(win[12]), float(win[4])
        expected = 1.0 * w0 / (w0 + w1)
        assert float(out[0, 8, 12]) == pytest.approx(expected, abs=1e-4)
        assert float(out[0, 8, 12]) < 1.0


class TestSingleOutput:
    def test_one_output_returned(self):
        arr = np.zeros((1, 4))
        assert _single_output({"o": arr}) is arr

    def test_two_outputs_raise(self):
        with pytest.raises(RuntimeError, match=r"exactly 1 model output, got 2"):
            _single_output({"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# 6. sharpen() / upscale_2x() with a recording fake model
# ---------------------------------------------------------------------------

class _RecordingModel:
    """Callable compiled-model stand-in that records every feed."""

    def __init__(self, fn):
        self.fn = fn
        self.feeds: list[dict] = []

    def __call__(self, feed):
        self.feeds.append(feed)
        return {"out": self.fn(feed)}


class _FakeCache:
    def __init__(self, model):
        self.model = model
        self.loaded: list[str] = []

    def load(self, path):
        self.loaded.append(Path(path).name)
        return self.model


def _textured_bgr(h=600, w=600, seed=1):
    rng = np.random.RandomState(seed)
    return rng.randint(64, 192, size=(h, w, 3)).astype(np.uint8)


@pytest.fixture
def no_autoparam(monkeypatch, tmp_path):
    """Point MODELS_DIR at an empty dir so autoparam falls back to motion 0.0."""
    monkeypatch.setattr(topaz, "MODELS_DIR", tmp_path / "no-models")


class TestSharpen:
    def test_feed_names_shapes_defaults_and_channel_order(self, no_autoparam):
        img = _textured_bgr()
        model = _RecordingModel(lambda f: f["netInput"] + 0.04)
        cache = _FakeCache(model)

        out = sharpen(img, cache)

        assert cache.loaded[0] == "slb-v1-fp32-512x512-ov.tz"
        first = model.feeds[0]
        assert set(first) == {"netInput", "t_param", "t_param1"}
        assert first["netInput"].shape == (1, 3, 512, 512)
        assert first["netInput"].dtype == np.float32
        assert first["t_param"].tolist() == [0.25]
        assert first["t_param1"].tolist() == [0.0]
        # model sees RGB: channel 0 of the first tile is the BGR image's channel 2
        np.testing.assert_allclose(first["netInput"][0, 0],
                                   img[:512, :512, 2].astype(np.float32) / 255.0)
        # output = in + 0.5 * (model - in) = in + 0.02 -> +5.1 levels, BGR order kept
        assert out.shape == img.shape and out.dtype == np.uint8
        expected = img.astype(np.float32) + 5.1
        assert float(np.abs(out.astype(np.float32) - expected).max()) <= 0.51
        # first row / column are real pixels, not black (regression)
        assert float(out[0].mean()) == pytest.approx(float(img[0].mean()) + 5.1, abs=0.6)
        assert float(out[:, 0].mean()) == pytest.approx(float(img[:, 0].mean()) + 5.1, abs=0.6)

    def test_explicit_params_reach_the_model(self, no_autoparam):
        model = _RecordingModel(lambda f: f["netInput"])
        sharpen(_textured_bgr(520, 520), _FakeCache(model), blur=0.1, noise=0.2)
        assert model.feeds[0]["t_param"].tolist() == pytest.approx([0.1])
        assert model.feeds[0]["t_param1"].tolist() == pytest.approx([0.2])

    def test_soft_input_is_returned_untouched_with_no_tile_inference(self, no_autoparam):
        grad = np.tile(np.linspace(60, 200, 600).astype(np.uint8)[None, :, None], (600, 1, 3))
        model = _RecordingModel(lambda f: f["netInput"])
        out = sharpen(grad, _FakeCache(model))
        assert out is grad
        assert model.feeds == []

    def test_textured_input_is_processed(self, no_autoparam):
        """Positive twin of the soft-input skip."""
        img = _textured_bgr()
        model = _RecordingModel(lambda f: f["netInput"])
        out = sharpen(img, _FakeCache(model))
        assert len(model.feeds) == 4
        assert out is not img
        assert np.array_equal(out, img)   # identity model, alpha blend: unchanged values

    def test_high_motion_score_skips(self, monkeypatch):
        monkeypatch.setattr(topaz, "_estimate_sharpen_params",
                            lambda chw, cache: (0.0, 0.0, 0.5, 0.0))
        img = _textured_bgr()
        model = _RecordingModel(lambda f: f["netInput"])
        assert sharpen(img, _FakeCache(model)) is img
        assert model.feeds == []

    def test_low_motion_score_runs(self, monkeypatch):
        """Positive twin of the motion skip."""
        monkeypatch.setattr(topaz, "_estimate_sharpen_params",
                            lambda chw, cache: (0.0, 0.0, 0.29, 0.0))
        model = _RecordingModel(lambda f: f["netInput"])
        sharpen(_textured_bgr(), _FakeCache(model))
        assert len(model.feeds) == 4

    def test_halo_producing_model_output_is_rejected(self, no_autoparam):
        img = _textured_bgr()

        def boost(f):
            x = f["netInput"]
            return np.clip(x + 6.0 * (x - 0.5), 0, 1)

        out = sharpen(img, _FakeCache(_RecordingModel(boost)), alpha=1.0)
        assert out is img

    def test_benign_model_output_is_accepted(self, no_autoparam):
        """Positive twin: a small brightness shift is not a halo."""
        img = _textured_bgr()
        out = sharpen(img, _FakeCache(_RecordingModel(lambda f: f["netInput"] + 0.04)))
        assert out is not img


class TestUpscale2x:
    def test_feed_names_and_corner_values(self):
        img = np.full((40, 40, 3), 90, dtype=np.uint8)
        model = _RecordingModel(
            lambda f: np.repeat(np.repeat(f["netInput"], 2, axis=2), 2, axis=3))
        out = upscale_2x(img, _FakeCache(model))
        assert set(model.feeds[0]) == {"netInput", "pn", "pb", "pc"}
        assert model.feeds[0]["netInput"].shape == (1, 3, 128, 128)
        assert out.shape == (80, 80, 3)
        assert out[0, 0].tolist() == [90, 90, 90]
        assert out[-1, -1].tolist() == [90, 90, 90]


# ---------------------------------------------------------------------------
# 7. enhance() output policy
# ---------------------------------------------------------------------------

class TestEnhancePolicy:
    def _jpeg(self, path, size=(64, 64)):
        Image.fromarray(np.full((*size, 3), 128, dtype=np.uint8)).save(
            str(path), "JPEG", quality=90)

    def test_reencode_uses_quality_95(self, tmp_path, monkeypatch):
        src, dst, ref = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "ref.jpg"
        rng = np.random.RandomState(3)
        arr = rng.randint(0, 255, size=(64, 64, 3)).astype(np.uint8)
        Image.fromarray(arr).save(str(src), "JPEG", quality=90)
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img.copy())
        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))
        Image.fromarray(np.array(Image.open(src).convert("RGB"))).save(
            str(ref), "JPEG", quality=95, subsampling=0)
        assert Image.open(dst).quantization == Image.open(ref).quantization

    def test_skipped_sharpen_copies_source_bytes(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._jpeg(src)
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img)
        enhance(src, dst, mode="stock", cache=MagicMock(spec=ModelCache))
        assert dst.read_bytes() == src.read_bytes()

    def test_upscale_refuses_inat_sized_input(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._jpeg(src, size=(1500, 2048))
        boom = MagicMock(side_effect=AssertionError("must refuse before any model runs"))
        monkeypatch.setattr(topaz, "sharpen", boom)
        with pytest.raises(ValueError, match=r"iNat-sized"):
            enhance(src, dst, mode="upscale", cache=MagicMock(spec=ModelCache))
        assert not dst.exists()

    def test_upscale_refuses_result_over_100_mp(self, tmp_path, monkeypatch):
        src, dst = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._jpeg(src, size=(4900, 5200))   # 25.5 MP -> 102 MP
        monkeypatch.setattr(topaz, "sharpen",
                            MagicMock(side_effect=AssertionError("refuse first")))
        with pytest.raises(ValueError, match=r"exceeds 100 MP"):
            enhance(src, dst, mode="upscale", cache=MagicMock(spec=ModelCache))

    def test_upscale_allows_full_sensor_under_limit(self, tmp_path, monkeypatch):
        """Positive twin: 6000x4000 (24 MP -> 96 MP) is not refused."""
        src, dst = tmp_path / "a.jpg", tmp_path / "b.jpg"
        self._jpeg(src, size=(4000, 6000))
        monkeypatch.setattr(topaz, "sharpen", lambda img, cache, **kw: img.copy())
        monkeypatch.setattr(topaz, "upscale_2x", lambda img, cache: img[:8, :8].copy())
        enhance(src, dst, mode="upscale", cache=MagicMock(spec=ModelCache))
        assert Image.open(dst).size == (8, 8)


class TestEnsureExtractedCleanup:
    def test_failed_extraction_leaves_no_empty_cache_dir(self, tmp_path):
        trunc = _make_truncated_tz(tmp_path, name="slb-x.tz")
        cache_dir = tmp_path / "cache"
        with pytest.raises(RuntimeError, match=r"not yet fully downloaded"):
            ModelCache(cache_dir=cache_dir)._ensure_extracted(trunc)
        assert not (cache_dir / "slb-x").exists()

    def test_successful_extraction_keeps_cache_dir(self, tmp_path):
        tz = _make_tz(tmp_path, {"m.xml": b"<net/>", "m.bin": b"\x00"}, name="ok.tz")
        cache_dir = tmp_path / "cache"
        got = ModelCache(cache_dir=cache_dir)._ensure_extracted(tz)
        assert got == cache_dir / "ok"
        assert (got / "m.xml").read_bytes() == b"<net/>"
