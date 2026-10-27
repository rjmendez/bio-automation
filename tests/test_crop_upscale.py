"""Unit tests for bio_automation.crop_upscale (hermetic -- lce.models.upscale.run,
disk_guard.wait_for_output and cuda_dlls.add_cuda_dll_dirs are mocked; no real
GPU/model inference)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image as RealImage

from bio_automation import crop_upscale


class TestPickScale:
    def test_small_crop_at_the_scale4_boundary_gets_scale_4(self):
        # 2500x2500 = 6.25MP exactly; 6.25*16 = 100 <= 100 -> still fits at scale=4
        assert crop_upscale.pick_scale(2500, 2500) == 4

    def test_just_over_the_scale4_boundary_gets_scale_2(self):
        # 2500x2501 = 6.2525MP; *16 = 100.04 > 100 -> drops to scale=2
        assert crop_upscale.pick_scale(2500, 2501) == 2

    def test_tiny_crop_gets_scale_4(self):
        assert crop_upscale.pick_scale(1000, 1000) == 4

    def test_medium_crop_at_the_scale2_boundary_gets_scale_2(self):
        # 5000x5000 = 25MP exactly; 25*4 = 100 <= 100 -> still fits at scale=2
        assert crop_upscale.pick_scale(5000, 5000) == 2

    def test_just_over_the_scale2_boundary_gets_none(self):
        # 5000x5001 = 25.005MP; *4 = 100.02 > 100 -> too large for any scale
        assert crop_upscale.pick_scale(5000, 5001) is None

    def test_large_crop_gets_none(self):
        assert crop_upscale.pick_scale(8000, 6000) is None


class _FakeImg:
    def __init__(self, size):
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakePIL:
    """A recording-free stub for PIL.Image: run() only reads .size through
    it, never pixel data, so a fake sized object is a faithful dependency
    stub, not a stub of the unit under test's own decision logic."""

    def __init__(self, sizes: dict):
        self._sizes = sizes

    def open(self, path):
        return _FakeImg(self._sizes[Path(path).name])


def _make_jpgs(src_dir: Path, names: list[str]) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (src_dir / name).write_bytes(b"not a real jpeg, just a fixture")


def _recording_upscale_one(monkeypatch):
    calls = []

    def _fake(src, dst, scale, cache):
        calls.append({"src": src, "dst": dst, "scale": scale, "cache": cache})
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"upscaled fixture bytes")

    monkeypatch.setattr(crop_upscale, "upscale_one", _fake)
    return calls


def _always_room(monkeypatch):
    monkeypatch.setattr(crop_upscale.disk_guard, "wait_for_output", lambda *a, **k: True)


def _cuda_ok(monkeypatch):
    monkeypatch.setattr(crop_upscale, "_cuda_ready", False)
    monkeypatch.setattr(crop_upscale.cuda_dlls, "add_cuda_dll_dirs", lambda *a, **k: [])


class TestRun:
    def test_upscales_small_crop_with_scale_4(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["small.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"small.jpg": (1000, 1000)}))
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result == {"done": 1, "skipped_existing": 0, "skipped_full_frame": 0,
                           "skipped_too_large": 0, "failed": 0, "total": 1}
        assert len(calls) == 1 and calls[0]["scale"] == 4

    def test_upscales_medium_crop_with_scale_2(self, tmp_path, monkeypatch):
        """Positive twin to the scale=4 test above, on the same kind of
        fixture: a mid-size crop must get scale=2, not scale=4."""
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["medium.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"medium.jpg": (4000, 3000)}))  # 12MP
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result["done"] == 1
        assert calls[0]["scale"] == 2

    def test_skips_full_frame_image(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["full.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"full.jpg": (8256, 5504)}))
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result == {"done": 0, "skipped_existing": 0, "skipped_full_frame": 1,
                           "skipped_too_large": 0, "failed": 0, "total": 1}
        assert calls == []

    def test_skips_image_too_large_to_upscale(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["huge_crop.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"huge_crop.jpg": (6000, 5000)}))  # 30MP
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result["skipped_too_large"] == 1
        assert calls == []

    def test_skips_file_already_in_out_dir(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["a.jpg"])
        out_dir.mkdir()
        (out_dir / "a.jpg").write_bytes(b"already done")
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"a.jpg": (1000, 1000)}))
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result == {"done": 0, "skipped_existing": 1, "skipped_full_frame": 0,
                           "skipped_too_large": 0, "failed": 0, "total": 1}
        assert calls == []

    def test_failed_upscale_cleans_up_partial_file(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["a.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"a.jpg": (1000, 1000)}))
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        def _raising(src, dst, scale, cache):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"corrupt partial write")
            raise RuntimeError("boom")

        monkeypatch.setattr(crop_upscale, "upscale_one", _raising)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result["failed"] == 1
        assert not (out_dir / "a.jpg").exists()

    def test_disk_guard_refusal_stops_before_any_upscale_call(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["a.jpg", "b.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"a.jpg": (1000, 1000), "b.jpg": (1000, 1000)}))
        calls = _recording_upscale_one(monkeypatch)
        monkeypatch.setattr(crop_upscale.disk_guard, "wait_for_output", lambda *a, **k: False)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result["done"] == 0
        assert calls == []

    def test_limit_caps_the_number_of_files_processed(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["a.jpg", "b.jpg", "c.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({n: (1000, 1000) for n in ("a.jpg", "b.jpg", "c.jpg")}))
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        _cuda_ok(monkeypatch)

        result = crop_upscale.run(src_dir, out_dir=out_dir, limit=2)

        assert result["total"] == 2
        assert len(calls) == 2

    def test_missing_cuda_dlls_raises_and_counts_failed(self, tmp_path, monkeypatch):
        """GPU-only contract (same rule as daclip/sr_model): a missing CUDA
        DLL folder must raise, never fall back to CPU silently."""
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _make_jpgs(src_dir, ["a.jpg"])
        monkeypatch.setattr(crop_upscale, "Image", _FakePIL({"a.jpg": (1000, 1000)}))
        calls = _recording_upscale_one(monkeypatch)
        _always_room(monkeypatch)
        monkeypatch.setattr(crop_upscale, "_cuda_ready", False)
        monkeypatch.setattr(crop_upscale.cuda_dlls, "add_cuda_dll_dirs", lambda *a, **k: [Path(r"C:\nonexistent\cuda")])

        result = crop_upscale.run(src_dir, out_dir=out_dir)

        assert result["failed"] == 1
        assert calls == []  # upscale_one (mocked) never reached: the raise happens before it


class TestUpscaleOne:
    def _make_bgr_like_source(self, path: Path, size=(32, 24)) -> None:
        img = RealImage.new("RGB", size, color=(200, 100, 10))
        img.save(path, "JPEG", quality=95)

    def test_calls_lce_upscale_with_variant_stock_and_writes_output(self, tmp_path, monkeypatch):
        src = tmp_path / "in.jpg"
        dst = tmp_path / "out" / "in.jpg"
        self._make_bgr_like_source(src)

        calls = []

        def _fake_run(chw, cache, scale, variant):
            calls.append({"scale": scale, "variant": variant, "shape": chw.shape, "cache": cache})
            return np.repeat(np.repeat(chw, scale, axis=1), scale, axis=2)

        import lce.models.upscale as real_lce_upscale
        monkeypatch.setattr(real_lce_upscale, "run", _fake_run)

        sentinel_cache = object()
        crop_upscale.upscale_one(src, dst, scale=2, cache=sentinel_cache)

        assert dst.exists()
        assert calls == [{"scale": 2, "variant": "stock", "shape": (3, 24, 32), "cache": sentinel_cache}]
        with RealImage.open(dst) as out_img:
            assert out_img.size == (64, 48)  # 32x24 input, scale=2
