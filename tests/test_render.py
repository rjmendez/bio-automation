"""Unit tests for bio_automation.render (extract_preview_bytes / rawpy.imread
mocked -- external dependencies already tested elsewhere (exif.py,
preclean.py's own rawpy usage); Pillow resize/encode itself runs for real on
real decoded pixel data, not mocked)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation import render
from bio_automation.render import (
    ExportBoundsError,
    export_inat_jpeg,
    export_stock_jpeg,
    render_review_jpeg,
)

SOURCE_EXIF = {"DateTimeOriginal": "2026:07:04 09:30:15", "Make": "NIKON CORPORATION",
               "Model": "NIKON Z 7_2", "LensModel": "NIKKOR Z MC 105mm f/2.8 VR S"}


@pytest.fixture(autouse=True)
def _source_exif(monkeypatch):
    """exiftool on the (nonexistent) fake raw is the only thing replaced."""
    monkeypatch.setattr(render, "extract_exif", lambda p: dict(SOURCE_EXIF))


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

        export_stock_jpeg(Path("f.NEF"), dest, max_dim=2500)

        out = Image.open(dest)
        assert out.size == (2500, 1667)  # 4.17 MP, still inside Adobe's 4 MP floor

    def test_below_4mp_raises_and_deletes_the_file(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock_small.jpg"

        with pytest.raises(ExportBoundsError, match=r"1000x667 = 0.67 MP is below the 4 MP minimum"):
            export_stock_jpeg(Path("f.NEF"), dest, max_dim=1000)
        assert not dest.exists()

    def test_tight_crop_below_4mp_raises(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))  # 6 MP; a 0.5x0.5 crop is 1.5 MP
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)

        with pytest.raises(ExportBoundsError, match=r"1500x1000 = 1.50 MP is below"):
            export_stock_jpeg(Path("f.NEF"), tmp_path / "c.jpg", crop=[0.0, 0.0, 0.5, 0.5])

    def test_over_45mb_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "STOCK_MAX_BYTES", 1000)
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "big.jpg"

        with pytest.raises(ExportBoundsError, match=r"bytes exceeds the 1000 byte maximum"):
            export_stock_jpeg(Path("f.NEF"), dest)
        assert not dest.exists()


class TestExportMetadata:
    def test_stock_export_carries_capture_time_camera_and_srgb_icc(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((3000, 2000)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock.jpg"

        export_stock_jpeg(Path("f.NEF"), dest)

        out = Image.open(dest)
        exif = out.getexif()
        assert exif.get_ifd(0x8769)[0x9003] == "2026:07:04 09:30:15"
        assert exif.get_ifd(0x8769)[0xA434] == "NIKKOR Z MC 105mm f/2.8 VR S"
        assert exif[0x010F] == "NIKON CORPORATION"
        assert exif[0x0110] == "NIKON Z 7_2"
        assert out.info["icc_profile"] == render._srgb_icc_bytes()
        assert 0x8825 not in exif  # no GPS IFD is ever copied

    def test_inat_export_carries_capture_time(self, tmp_path, monkeypatch):
        fake_raw = _FakeRaw(_make_raw16((400, 300)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "inat.jpg"

        export_inat_jpeg(Path("f.NEF"), dest)

        assert Image.open(dest).getexif().get_ifd(0x8769)[0x9003] == "2026:07:04 09:30:15"

    def test_inat_over_20mb_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "INAT_MAX_BYTES", 500)
        fake_raw = _FakeRaw(_make_raw16((400, 300)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)

        with pytest.raises(ExportBoundsError, match=r"exceeds the 500 byte maximum"):
            export_inat_jpeg(Path("f.NEF"), tmp_path / "inat.jpg")

    def test_decodes_sensor_resolution_not_the_embedded_preview(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "extract_preview_bytes",
                             lambda p: (_ for _ in ()).throw(AssertionError("must not use embedded preview")))
        fake_raw = _FakeRaw(_make_raw16((2400, 1700)))
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "stock.jpg"

        export_stock_jpeg(Path("f.NEF"), dest)

        assert dest.exists()
        assert fake_raw.postprocess_calls == [{
            "output_bps": 16, "no_auto_bright": True, "use_camera_wb": True,
            "output_color": render.rawpy.ColorSpace.sRGB,
        }]


def _quadrant_image(size=(400, 300)) -> Image.Image:
    """A real image with four distinguishable solid-color quadrants, so a
    crop's exact position/size can be verified by sampling real pixels, not
    just by checking the output's bounding-box size."""
    w, h = size
    img = Image.new("RGB", size)
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), (
                (255, 0, 0) if x < w // 2 and y < h // 2 else
                (0, 255, 0) if x >= w // 2 and y < h // 2 else
                (0, 0, 255) if x < w // 2 and y >= h // 2 else
                (255, 255, 0)
            ))
    return img


class TestApplyCrop:
    def test_none_crop_returns_the_same_image_unchanged(self):
        img = _quadrant_image()
        assert render._apply_crop(img, None) is img

    def test_crops_to_the_exact_normalized_rect(self):
        """Top-left quadrant only: [0, 0, 0.5, 0.5] on a 400x300 image must
        crop to exactly (0, 0, 200, 150), real pixels all red."""
        img = _quadrant_image((400, 300))
        out = render._apply_crop(img, [0.0, 0.0, 0.5, 0.5])
        assert out.size == (200, 150)
        assert out.getpixel((0, 0)) == (255, 0, 0)
        assert out.getpixel((199, 149)) == (255, 0, 0)

    def test_bottom_right_quadrant_crop(self):
        """Positive twin to the top-left case: a different rect must crop
        to a different region, not just any region of the same size."""
        img = _quadrant_image((400, 300))
        out = render._apply_crop(img, [0.5, 0.5, 1.0, 1.0])
        assert out.size == (200, 150)
        assert out.getpixel((0, 0)) == (255, 255, 0)
        assert out.getpixel((199, 149)) == (255, 255, 0)

    def test_reversed_coordinates_are_sorted_not_rejected(self):
        """[x1,y1,x0,y0]-style input (e.g. a UI drag that went up-left
        instead of down-right) must produce the same crop as the sorted
        rect, not a degenerate/empty one."""
        img = _quadrant_image((400, 300))
        out = render._apply_crop(img, [0.5, 0.5, 0.0, 0.0])
        assert out.size == (200, 150)
        assert out.getpixel((0, 0)) == (255, 0, 0)

    def test_degenerate_rect_returns_the_image_unchanged(self):
        """A zero-width/height rect (e.g. a UI click with no drag) must not
        produce an empty image -- it's treated as no crop."""
        img = _quadrant_image((400, 300))
        out = render._apply_crop(img, [0.3, 0.3, 0.3, 0.6])
        assert out is img

    def test_out_of_range_coordinates_are_clamped(self):
        img = _quadrant_image((400, 300))
        out = render._apply_crop(img, [-0.5, -0.5, 1.5, 1.5])
        assert out.size == (400, 300)


class TestExportInatJpegCrop:
    def test_crop_is_applied_before_resize(self, tmp_path, monkeypatch):
        # Pure red on the left half, pure blue on the right half, at full
        # sensor resolution -- cropping to the right half before the 2048px
        # resize must leave only blue in the output.
        rgb16 = np.zeros((300, 400, 3), dtype=np.uint16)
        rgb16[:, 200:, 2] = 65535
        fake_raw = _FakeRaw(rgb16)
        monkeypatch.setattr(render.rawpy, "imread", lambda p: fake_raw)
        dest = tmp_path / "cropped.jpg"

        export_inat_jpeg(Path("f.NEF"), dest, crop=[0.5, 0.0, 1.0, 1.0])

        out = Image.open(dest)
        assert out.size == (200, 300)  # right half of 400x300, uncapped (200 < max_dim=2048)
        r, g, b = out.getpixel((100, 150))
        assert b > 200 and r < 20


class TestApplyCropRejectsBadNumbers:
    def test_string_element_raises_value_error(self):
        with pytest.raises(ValueError, match=r"crop values must be finite numbers"):
            render._apply_crop(_quadrant_image(), ["a", 0, 1, 1])

    def test_nan_raises_value_error(self):
        with pytest.raises(ValueError, match=r"crop values must be finite numbers"):
            render._apply_crop(_quadrant_image(), [float("nan"), 0, 1, 1])

    def test_finite_numbers_still_crop(self):
        assert render._apply_crop(_quadrant_image((400, 300)), [0, 0, 0.5, 1]).size == (200, 300)
