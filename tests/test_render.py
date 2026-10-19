"""Unit tests for bio_automation.render (extract_preview_bytes / rawpy.imread
mocked -- external dependencies already tested elsewhere (exif.py,
preclean.py's own rawpy usage); Pillow resize/encode itself runs for real on
real decoded pixel data, not mocked)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from bio_automation import render
from bio_automation.render import export_inat_jpeg, export_stock_jpeg, render_review_jpeg


def _make_jpeg_bytes(size=(400, 300), color=(200, 100, 50)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


class _FakeRaw:
    """Stands in for rawpy's imread() context manager -- postprocess()
    returns a real uint16 HxWx3 array (a plain gradient, so resize/encode
    sees real, distinguishable pixel data), never a Mock object."""

    def __init__(self, rgb16: np.ndarray):
        self._rgb16 = rgb16
        self.postprocess_calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def postprocess(self, **kwargs):
        self.postprocess_calls.append(kwargs)
        return self._rgb16


def _make_raw16(size=(400, 300), color=(40000, 20000, 10000)) -> np.ndarray:
    h, w = size[1], size[0]
    return np.full((h, w, 3), color, dtype=np.uint16)


class TestRenderReviewJpeg:
    def test_resizes_to_max_dim_and_writes_real_jpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes", lambda p: _make_jpeg_bytes((4000, 3000)))
        dest = tmp_path / "review" / "DSC_0001.jpg"

        result = render_review_jpeg(Path("DSC_0001.NEF"), dest, max_dim=500)

        assert result == dest
        assert dest.exists()
        out = Image.open(dest)
        assert out.format == "JPEG"
        # thumbnail() preserves aspect ratio, bounds within max_dim, exact for this 4:3 case
        assert out.size == (500, 375)

    def test_creates_dest_parent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes", lambda p: _make_jpeg_bytes())
        dest = tmp_path / "does" / "not" / "exist" / "out.jpg"

        render_review_jpeg(Path("f.NEF"), dest, max_dim=100)
        assert dest.exists()


class TestExportInatJpeg:
    def test_uses_its_own_default_max_dim_and_quality(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((5000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "inat.jpg"

        export_inat_jpeg(Path("f.NEF"), dest)  # defaults: max_dim=2048, quality=95

        out = Image.open(dest)
        assert out.size == (2048, 819)  # 5000x2000 -> bounded to 2048 on the long edge
        assert out.format == "JPEG"

    def test_decodes_sensor_resolution_not_the_embedded_preview(self, tmp_path, monkeypatch):
        """The 158x118-degenerate-output incident (2026-10-19, see
        pipeline.stack_focus's docstring): extract_preview_bytes must never
        be consulted for this export -- it's a full raw decode now, not the
        camera's embedded JPEG preview."""
        monkeypatch.setattr(render, "extract_preview_bytes",
                             lambda p: (_ for _ in ()).throw(AssertionError("must not use embedded preview")))
        fake_raw = _FakeRaw(_make_raw16((400, 300)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "inat.jpg"

        export_inat_jpeg(Path("f.NEF"), dest)

        assert dest.exists()
        assert fake_raw.postprocess_calls == [{
            "output_bps": 16, "no_auto_bright": True, "use_camera_wb": True,
            "output_color": render.rawpy.ColorSpace.sRGB,
        }]


class TestExportStockJpeg:
    def test_default_max_dim_none_keeps_full_size(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock.jpg"

        export_stock_jpeg(Path("f.NEF"), dest)  # default max_dim=None

        out = Image.open(dest)
        assert out.size == (3000, 2000)

    def test_explicit_max_dim_still_resizes(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock_small.jpg"

        export_stock_jpeg(Path("f.NEF"), dest, max_dim=1000)

        out = Image.open(dest)
        assert out.size == (1000, 667)

    def test_decodes_sensor_resolution_not_the_embedded_preview(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes",
                             lambda p: (_ for _ in ()).throw(AssertionError("must not use embedded preview")))
        fake_raw = _FakeRaw(_make_raw16((400, 300)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock.jpg"

        export_stock_jpeg(Path("f.NEF"), dest)

        assert dest.exists()
        assert fake_raw.postprocess_calls == [{
            "output_bps": 16, "no_auto_bright": True, "use_camera_wb": True,
            "output_color": render.rawpy.ColorSpace.sRGB,
        }]
