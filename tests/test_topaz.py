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
    _run_tiled,
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

    def test_inat_mode_calls_sharpen_not_upscale(self, tmp_path, monkeypatch):
        src = tmp_path / "input.jpg"
        dst = tmp_path / "output.jpg"
        self._write_jpeg(src)

        sharpen_calls = []

        def spy_sharpen(img_bgr, cache, blur=0.5, noise=0.0):
            sharpen_calls.append(img_bgr.shape)
            return img_bgr.copy()

        monkeypatch.setattr(topaz, "sharpen", spy_sharpen)
        monkeypatch.setattr(topaz, "upscale_2x",
                            MagicMock(side_effect=AssertionError("upscale_2x must not be called in inat mode")))

        enhance(src, dst, mode="inat", cache=MagicMock(spec=ModelCache))
        assert len(sharpen_calls) == 1

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
