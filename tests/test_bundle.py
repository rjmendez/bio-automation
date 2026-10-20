"""bundle.py: validation against the export standard and the flat-zip builder. Real JPEGs made with Pillow, nothing stubbed but disk/zip failure injection."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from bio_automation import bundle
from bio_automation.bundle import BundleError, build_bundle, validate_jpeg

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def make_jpeg(path: Path, size=(64, 48), *, quality=95, icc=SRGB, gps=False, mode="RGB", color=(120, 90, 60)) -> Path:
    img = Image.new(mode, size, color if mode == "RGB" else 128)
    kw = {}
    if icc is not None:
        kw["icc_profile"] = icc
    if gps:
        ex = Image.Exif()
        g = ex.get_ifd(0x8825)
        g[1], g[2], g[3], g[4] = "N", (40.0, 0.0, 0.0), "W", (75.0, 0.0, 0.0)
        kw["exif"] = ex.tobytes()
    img.save(path, "JPEG", quality=quality, **kw)
    return path


class TestValidateInat:
    def test_conforming_file_has_no_problems(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", (2048, 100)), "inat") == []

    def test_long_edge_2049_is_rejected_and_2048_is_not(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", (2049, 100)), "inat") == ["long edge 2049 px exceeds 2048"]
        assert validate_jpeg(make_jpeg(tmp_path / "b.jpg", (100, 2048)), "inat") == []

    def test_quality_90_is_rejected_and_96_is_accepted(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", quality=90), "inat") == ["JPEG quality below 95 (quantisation table coarser than q95)"]
        assert validate_jpeg(make_jpeg(tmp_path / "b.jpg", quality=96), "inat") == []

    def test_missing_icc_profile(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", icc=None), "inat") == ["no ICC profile (sRGB required)"]

    def test_non_srgb_profile_is_named(self, tmp_path):
        lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", icc=lab), "inat") == ["ICC profile is 'Lab identity built-in', not sRGB"]

    def test_size_over_the_byte_limit(self, tmp_path, monkeypatch):
        f = make_jpeg(tmp_path / "a.jpg", (400, 300))
        size = f.stat().st_size
        monkeypatch.setattr(bundle, "INAT_MAX_BYTES", size - 1)
        assert validate_jpeg(f, "inat") == [f"{size} bytes exceeds the {size - 1} byte iNat maximum"]
        monkeypatch.setattr(bundle, "INAT_MAX_BYTES", size)          # positive twin: exactly at the limit passes
        assert validate_jpeg(f, "inat") == []

    def test_gps_is_allowed_for_inat(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", gps=True), "inat") == []

    def test_png_with_a_jpg_name_is_not_a_jpeg(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (8, 8)).save(p, "PNG")
        assert validate_jpeg(p, "inat") == ["not a JPEG (format PNG)"]

    def test_grayscale_is_not_rgb(self, tmp_path):
        assert "mode L, not RGB" in validate_jpeg(make_jpeg(tmp_path / "a.jpg", mode="L", icc=SRGB), "inat")

    def test_unreadable_file_is_reported_not_raised(self, tmp_path):
        p = tmp_path / "a.jpg"
        p.write_bytes(b"not an image")
        [problem] = validate_jpeg(p, "inat")
        assert problem.startswith("cannot be read as an image:")

    def test_unknown_kind_raises(self, tmp_path):
        with pytest.raises(BundleError, match=r"kind must be one of \('inat', 'stock'\), got 'adobe'"):
            validate_jpeg(make_jpeg(tmp_path / "a.jpg"), "adobe")


class TestValidateStock:
    def test_exactly_4_megapixels_passes_and_one_column_less_fails(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", (2000, 2000)), "stock") == []
        assert validate_jpeg(make_jpeg(tmp_path / "b.jpg", (1999, 2000)), "stock") == ["1999x2000 = 4.00 MP is below the 4 MP minimum"]

    def test_stock_has_no_long_edge_limit(self, tmp_path):
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", (4000, 2000)), "stock") == []

    def test_above_the_maximum_megapixels(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bundle, "STOCK_MIN_PIXELS", 1000)
        monkeypatch.setattr(bundle, "STOCK_MAX_PIXELS", 2_000_000)
        assert validate_jpeg(make_jpeg(tmp_path / "a.jpg", (1500, 1500)), "stock") == ["1500x1500 = 2.25 MP is above the 2 MP maximum"]
        assert validate_jpeg(make_jpeg(tmp_path / "b.jpg", (1000, 2000)), "stock") == []     # exactly 2,000,000 passes

    def test_gps_block_is_rejected_for_stock(self, tmp_path):
        f = make_jpeg(tmp_path / "a.jpg", (2000, 2000), gps=True)
        assert validate_jpeg(f, "stock") == ["contains a GPS block (Stock files must not carry coordinates)"]
        assert validate_jpeg(make_jpeg(tmp_path / "b.jpg", (2000, 2000)), "stock") == []     # twin: same file without GPS

    def test_byte_limit_message_names_stock(self, tmp_path, monkeypatch):
        f = make_jpeg(tmp_path / "a.jpg", (2000, 2000))
        monkeypatch.setattr(bundle, "STOCK_MAX_BYTES", 10)
        assert validate_jpeg(f, "stock") == [f"{f.stat().st_size} bytes exceeds the 10 byte Stock maximum"]


class TestBuildBundle:
    def _set(self, tmp_path):
        src = tmp_path / "set"
        src.mkdir()
        make_jpeg(src / "b_two.jpg", color=(10, 20, 30))
        make_jpeg(src / "A_one.JPG", color=(200, 100, 50))
        (src / "observations.csv").write_bytes(b"group,files\n1,A_one.JPG\n")
        (src / "notes.txt").write_text("not shipped", encoding="utf-8")
        return src

    def test_flat_stored_zip_with_exact_members_and_bytes(self, tmp_path):
        src = self._set(tmp_path)
        r = build_bundle("inat", [src], tmp_path / "out", name="batch")
        assert r.path == tmp_path / "out" / "batch.zip"
        assert (r.photos, r.extras) == (2, ("observations.csv",))
        with zipfile.ZipFile(r.path) as z:
            assert z.namelist() == ["A_one.JPG", "b_two.jpg", "observations.csv"]       # photos sorted case-insensitively, then extras; no folders
            assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_STORED}
            assert z.read("b_two.jpg") == (src / "b_two.jpg").read_bytes()
            assert z.read("observations.csv") == b"group,files\n1,A_one.JPG\n"

    def test_sha256_sidecar_matches_the_zip(self, tmp_path):
        r = build_bundle("inat", [self._set(tmp_path)], tmp_path / "out", name="batch.zip")
        digest = hashlib.sha256(r.path.read_bytes()).hexdigest()
        assert r.sha256 == digest
        assert (tmp_path / "out" / "batch.zip.sha256").read_bytes() == f"{digest}  batch.zip\n".encode("ascii")   # raw bytes: LF, never CRLF
        assert r.size_bytes == r.path.stat().st_size

    def test_default_name_carries_kind_and_count(self, tmp_path):
        r = build_bundle("inat", [self._set(tmp_path)], tmp_path / "out")
        assert r.path.name.startswith("inat-2photos-") and r.path.suffix == ".zip"

    def test_violation_writes_nothing_and_names_the_file(self, tmp_path):
        src = self._set(tmp_path)
        make_jpeg(src / "c_bad.jpg", quality=80)
        with pytest.raises(BundleError, match=r"1 of 3 files violate the inat standard, nothing written\. First: c_bad\.jpg: JPEG quality below 95"):
            build_bundle("inat", [src], tmp_path / "out", name="batch")
        assert not (tmp_path / "out").exists()          # validated before anything is created: no folder, no zip, no .part

    def test_conforming_set_of_the_same_size_builds(self, tmp_path):                     # positive twin of the violation test
        src = self._set(tmp_path)
        make_jpeg(src / "c_ok.jpg", quality=95)
        assert build_bundle("inat", [src], tmp_path / "out", name="batch").photos == 3

    def test_name_collision_across_sources_is_refused(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir(); b.mkdir()
        make_jpeg(a / "same.jpg"); make_jpeg(b / "same.jpg")
        with pytest.raises(BundleError, match=r"a flat zip needs unique file names; 1 collide, e\.g\. \['same\.jpg'\]"):
            build_bundle("inat", [a, b], tmp_path / "out")

    def test_no_jpegs(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "x.csv").write_text("a", encoding="utf-8")
        with pytest.raises(BundleError, match="no JPEG files found in the sources"):
            build_bundle("inat", [empty], tmp_path / "out")

    def test_missing_source(self, tmp_path):
        with pytest.raises(BundleError, match=r"source not found: .*nowhere"):
            build_bundle("inat", [tmp_path / "nowhere"], tmp_path / "out")

    def test_existing_zip_is_not_overwritten(self, tmp_path):
        src = self._set(tmp_path)
        build_bundle("inat", [src], tmp_path / "out", name="batch")
        before = (tmp_path / "out" / "batch.zip").read_bytes()
        with pytest.raises(BundleError, match="already exists; refusing to overwrite"):
            build_bundle("inat", [src], tmp_path / "out", name="batch")
        assert (tmp_path / "out" / "batch.zip").read_bytes() == before

    def test_not_enough_disk(self, tmp_path, monkeypatch):
        src = self._set(tmp_path)
        monkeypatch.setattr(bundle.shutil, "disk_usage", lambda p: type("U", (), {"free": 5})())
        with pytest.raises(BundleError, match=r"not enough disk in .*out: 5 bytes free, need \d+ \+ 1073741824"):
            build_bundle("inat", [src], tmp_path / "out", name="batch")

    def test_failed_write_leaves_neither_zip_nor_part(self, tmp_path, monkeypatch):
        src = self._set(tmp_path)

        def boom(self, *a, **k):
            raise OSError("disk went away")
        monkeypatch.setattr(zipfile.ZipFile, "write", boom)
        with pytest.raises(OSError, match="disk went away"):
            build_bundle("inat", [src], tmp_path / "out", name="batch")
        assert list((tmp_path / "out").iterdir()) == []

    def test_stock_kind_validates_with_stock_rules(self, tmp_path):
        src = tmp_path / "s"
        src.mkdir()
        make_jpeg(src / "big.jpg", (2000, 2000))
        assert build_bundle("stock", [src], tmp_path / "out", name="s").photos == 1
        make_jpeg(src / "small.jpg", (100, 100))
        with pytest.raises(BundleError, match=r"1 of 2 files violate the stock standard.*small\.jpg: 100x100 = 0\.01 MP is below the 4 MP minimum"):
            build_bundle("stock", [src], tmp_path / "out", name="s2")

    def test_default_download_dir_has_no_control_characters(self):
        assert bundle.DEFAULT_DOWNLOAD_DIR == "D:/bio-cache/downloads"
