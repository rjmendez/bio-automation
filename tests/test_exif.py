"""Unit tests for bio_automation.exif (hermetic -- exiftool subprocess calls mocked)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bio_automation import exif
from bio_automation.exif import build_exif_sidecar, extract_preview_bytes, render_xmp


def _fake_run(stdout=b""):
    result = MagicMock()
    result.stdout = stdout
    return result


class TestExtractPreviewBytes:
    def test_returns_preview_image_when_present(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_run(stdout=b"\xff\xd8\xff\xe0jpegbytes")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = extract_preview_bytes(Path("DSC_0001.NEF"))

        assert result == b"\xff\xd8\xff\xe0jpegbytes"
        assert len(calls) == 1
        assert calls[0] == ["exiftool", "-b", "-PreviewImage", "DSC_0001.NEF"]

    def test_falls_back_to_jpg_from_raw_when_preview_empty(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "-PreviewImage" in cmd:
                return _fake_run(stdout=b"")
            return _fake_run(stdout=b"\xff\xd8fallbackbytes")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = extract_preview_bytes(Path("DSC_0002.NEF"))

        assert result == b"\xff\xd8fallbackbytes"
        assert len(calls) == 2
        assert calls[1] == ["exiftool", "-b", "-JpgFromRaw", "DSC_0002.NEF"]

    def test_raises_when_neither_tag_present(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _fake_run(stdout=b""))

        with pytest.raises(RuntimeError, match=r"no embedded PreviewImage or JpgFromRaw tag found"):
            extract_preview_bytes(Path("DSC_0003.NEF"))


class TestBuildExifSidecar:
    def test_builds_exact_schema_with_gps(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "source" / "raw" / "DSC_1234.NEF"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(b"x")

        monkeypatch.setattr(exif, "extract_exif", lambda p: {
            "DateTimeOriginal": "2025:10:09 15:42:31",
            "Make": "NIKON CORPORATION",
            "Model": "NIKON Z 7_2",
            "LensModel": "NIKKOR Z 180-600mm",
        })
        monkeypatch.setattr(exif, "_extract_gps_numeric", lambda p: {
            "lat": 41.1234, "lon": -72.5678, "alt": 12.3,
        })

        sidecar = build_exif_sidecar(raw_path, tmp_path)

        assert sidecar == {
            "file": "source/raw/DSC_1234.NEF",
            "datetime_original": "2025:10:09 15:42:31",
            "camera_make": "NIKON CORPORATION",
            "camera_model": "NIKON Z 7_2",
            "lens_model": "NIKKOR Z 180-600mm",
            "gps": {"lat": 41.1234, "lon": -72.5678, "alt": 12.3},
        }

    def test_gps_is_none_not_fabricated_when_absent(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "source" / "raw" / "DSC_5555.NEF"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(b"x")

        monkeypatch.setattr(exif, "extract_exif", lambda p: {
            "DateTimeOriginal": "2025:08:02 13:20:32",
            "Make": "NIKON CORPORATION",
            "Model": "NIKON Z 7_2",
        })
        monkeypatch.setattr(exif, "_extract_gps_numeric", lambda p: None)

        sidecar = build_exif_sidecar(raw_path, tmp_path)

        assert sidecar["gps"] is None
        assert sidecar["lens_model"] is None

    def test_falls_back_to_lens_tag_when_lensmodel_missing(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "source" / "raw" / "DSC_7.NEF"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(b"x")

        monkeypatch.setattr(exif, "extract_exif", lambda p: {"Lens": "18-55mm"})
        monkeypatch.setattr(exif, "_extract_gps_numeric", lambda p: None)

        sidecar = build_exif_sidecar(raw_path, tmp_path)
        assert sidecar["lens_model"] == "18-55mm"


class TestExtractGpsNumeric:
    def test_parses_numeric_gps_tags(self, monkeypatch):
        import json as json_mod

        def fake_run(cmd, **kwargs):
            assert cmd == ["exiftool", "-j", "-GPSLatitude#", "-GPSLongitude#", "-GPSAltitude#", "f.NEF"]
            result = MagicMock()
            result.stdout = json_mod.dumps([{"GPSLatitude": 41.1234, "GPSLongitude": -72.5678,
                                              "GPSAltitude": 12.3}])
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = exif._extract_gps_numeric(Path("f.NEF"))
        assert result == {"lat": 41.1234, "lon": -72.5678, "alt": 12.3}

    def test_returns_none_when_no_gps_tags(self, monkeypatch):
        import json as json_mod

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.stdout = json_mod.dumps([{"SourceFile": "f.NEF"}])
            return result

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert exif._extract_gps_numeric(Path("f.NEF")) is None


class TestRenderXmp:
    def test_full_data_exact_output(self):
        data = {
            "datetime_original": "2025:10:09 15:42:31",
            "camera_make": "Nikon",
            "camera_model": "Z 7_2",
            "lens_model": "180-600mm",
            "gps": {"lat": 41.1234, "lon": -72.5678},
        }
        result = render_xmp(data)

        assert result == (
            '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
            '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '    <rdf:Description rdf:about=""\n'
            '        xmlns:exif="http://ns.adobe.com/exif/1.0/"\n'
            '        xmlns:tiff="http://ns.adobe.com/tiff/1.0/"\n'
            '        exif:DateTimeOriginal="2025:10:09 15:42:31"\n'
            '        tiff:Make="Nikon"\n'
            '        tiff:Model="Z 7_2"\n'
            '        exif:LensModel="180-600mm"\n'
            '        exif:GPSLatitude="41.1234"\n'
            '        exif:GPSLongitude="-72.5678"/>\n'
            '  </rdf:RDF>\n'
            '</x:xmpmeta>\n'
            '<?xpacket end="w"?>\n'
        )

    def test_missing_gps_omits_gps_attrs_not_crashes(self):
        data = {
            "datetime_original": None, "camera_make": None,
            "camera_model": None, "lens_model": None, "gps": None,
        }
        result = render_xmp(data)
        assert "GPSLatitude" not in result
        assert 'exif:DateTimeOriginal=""' in result

    def test_escapes_xml_special_characters(self):
        data = {
            "datetime_original": None, "camera_make": 'A & B <Co> "Inc"',
            "camera_model": None, "lens_model": None, "gps": None,
        }
        result = render_xmp(data)
        assert 'tiff:Make="A &amp; B &lt;Co&gt; &quot;Inc&quot;"' in result
        assert "<Co>" not in result
