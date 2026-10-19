"""Unit tests for bio_automation.render (extract_preview_bytes mocked -- it's
exif.py's own dependency, already tested there; Pillow resize/encode itself
runs for real on a small in-memory JPEG, not mocked)."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from bio_automation import render
from bio_automation.render import export_inat_jpeg, export_stock_jpeg, render_review_jpeg


def _make_jpeg_bytes(size=(400, 300), color=(200, 100, 50)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


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
        monkeypatch.setattr(render, "extract_preview_bytes", lambda p: _make_jpeg_bytes((5000, 2000)))
        dest = tmp_path / "inat.jpg"

        export_inat_jpeg(Path("f.NEF"), dest)  # defaults: max_dim=2048

        out = Image.open(dest)
        assert out.size == (2048, 819)  # 5000x2000 -> bounded to 2048 on the long edge


class TestExportStockJpeg:
    def test_default_max_dim_none_keeps_full_size(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes", lambda p: _make_jpeg_bytes((3000, 2000)))
        dest = tmp_path / "stock.jpg"

        export_stock_jpeg(Path("f.NEF"), dest)  # default max_dim=None

        out = Image.open(dest)
        assert out.size == (3000, 2000)

    def test_explicit_max_dim_still_resizes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes", lambda p: _make_jpeg_bytes((3000, 2000)))
        dest = tmp_path / "stock_small.jpg"

        export_stock_jpeg(Path("f.NEF"), dest, max_dim=1000)

        out = Image.open(dest)
        assert out.size == (1000, 667)
