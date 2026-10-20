"""Unit tests for the pipeline.py functions this session implemented:
write_sidecars, render_review, export_inat, export_stock,
build_and_save_keywords. Hermetic -- exif/render/stock's real I/O
(exiftool, Pillow, DA-CLIP) are each tested in their own module; here
their functions are mocked as pipeline.py's dependencies."""
from __future__ import annotations

from pathlib import Path

import pytest

from bio_automation import pipeline
from bio_automation.io import read_json
from bio_automation.models import Asset, Observation
from bio_automation.pipeline import (
    build_and_save_keywords,
    ensure_layout,
    export_inat,
    export_stock,
    render_review,
    save_assets,
    save_observation,
    scan_assets,
    write_sidecars,
)
from bio_automation.paths import export_dir, metadata_dir, review_dir, source_raw_dir


def _seed_two_kept_one_rejected(tmp_path):
    ensure_layout(tmp_path)
    raw_dir = source_raw_dir(tmp_path)
    for name in ("DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"):
        (raw_dir / name).write_bytes(b"x")
    save_assets(tmp_path, [
        Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0003.NEF", role="original", status="rejected", selected=False),
    ])


class TestWriteSidecars:
    def test_writes_exif_json_and_xmp_for_kept_raws_only(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)
        sidecar_calls = []

        def fake_build_sidecar(raw_path, observation_dir):
            sidecar_calls.append(raw_path.name)
            return {"file": raw_path.name, "camera_make": "Nikon"}

        monkeypatch.setattr(pipeline, "build_exif_sidecar", fake_build_sidecar)
        monkeypatch.setattr(pipeline, "render_xmp", lambda data: f"<xmp make={data['camera_make']}/>")

        written = write_sidecars(tmp_path)

        assert sidecar_calls == ["DSC_0001.NEF", "DSC_0002.NEF"]  # rejected raw excluded
        assert sorted(p.name for p in written) == [
            "DSC_0001.exif.json", "DSC_0001.xmp", "DSC_0002.exif.json", "DSC_0002.xmp",
        ]
        exif_json = read_json(metadata_dir(tmp_path) / "DSC_0001.exif.json")
        assert exif_json == {"file": "DSC_0001.NEF", "camera_make": "Nikon"}
        xmp_text = (metadata_dir(tmp_path) / "DSC_0001.xmp").read_text(encoding="utf-8")
        assert xmp_text == "<xmp make=Nikon/>"

    def test_no_kept_raws_writes_nothing(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="ingested", selected=False),
        ])
        monkeypatch.setattr(pipeline, "build_exif_sidecar",
                             lambda *a: pytest.fail("must not be called with no kept raws"))

        assert write_sidecars(tmp_path) == []


class TestRenderReview:
    def test_writes_jpeg_per_kept_raw_and_appends_review_assets(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)
        calls = []

        def fake_render(raw_path, dest_path, crop=None):
            calls.append((raw_path.name, dest_path, crop))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "render_review_jpeg", fake_render)

        written = render_review(tmp_path)

        assert [c[0] for c in calls] == ["DSC_0001.NEF", "DSC_0002.NEF"]
        assert sorted(p.name for p in written) == ["DSC_0001.jpg", "DSC_0002.jpg"]
        for p in written:
            assert p.parent == review_dir(tmp_path)
            assert p.exists()

        assets = scan_assets(tmp_path)
        review_assets = [a for a in assets if a.role == "review"]
        assert sorted(a.file for a in review_assets) == ["review/DSC_0001.jpg", "review/DSC_0002.jpg"]
        assert all(a.status == "generated" for a in review_assets)

    def test_no_kept_raws_does_not_touch_assets_jsonl(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [])
        monkeypatch.setattr(pipeline, "render_review_jpeg",
                             lambda *a, **k: pytest.fail("must not be called"))

        assert render_review(tmp_path) == []
        assert scan_assets(tmp_path) == []

    def test_forwards_each_kept_assets_own_crop_rect(self, tmp_path, monkeypatch):
        """A kept raw's Asset.crop (set by the review console) must reach
        render_review_jpeg as its crop= kwarg -- not dropped, not shared
        across raws. Positive twin to the uncropped case above, where every
        fake_render call's crop is implicitly None."""
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        (raw_dir / "DSC_0001.NEF").write_bytes(b"x")
        (raw_dir / "DSC_0002.NEF").write_bytes(b"x")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True,
                  crop=[0.1, 0.2, 0.9, 0.8]),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        calls = {}

        def fake_render(raw_path, dest_path, crop=None):
            calls[raw_path.name] = crop
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "render_review_jpeg", fake_render)

        render_review(tmp_path)

        assert calls == {"DSC_0001.NEF": [0.1, 0.2, 0.9, 0.8], "DSC_0002.NEF": None}


class TestExportInat:
    def test_writes_into_export_inat_dir_with_upload_role(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)

        def fake_render(raw_path, dest_path, crop=None):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "export_inat_jpeg", fake_render)

        written = export_inat(tmp_path)

        assert all(p.parent == export_dir(tmp_path, "inat") for p in written)
        assets = scan_assets(tmp_path)
        upload_assets = [a for a in assets if a.role == "upload"]
        assert len(upload_assets) == 2
        assert all(a.status == "ready" for a in upload_assets)
        assert all(a.file.startswith("export/inat/") for a in upload_assets)


class TestExportStock:
    def test_writes_into_export_stock_dir_with_stock_role(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)

        def fake_render(raw_path, dest_path, crop=None):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "export_stock_jpeg", fake_render)

        written = export_stock(tmp_path)

        assert all(p.parent == export_dir(tmp_path, "stock") for p in written)
        assets = scan_assets(tmp_path)
        stock_assets = [a for a in assets if a.role == "stock"]
        assert len(stock_assets) == 2
        assert all(a.file.startswith("export/stock/") for a in stock_assets)


class TestBuildAndSaveKeywords:
    def test_builds_from_observation_and_writes_keywords_json(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        save_observation(tmp_path, Observation(
            observation_id=123, obsdate="2025-08-02", source="iNaturalist",
            taxon={"name": "Araneus diadematus", "common_name": "Garden spider", "rank": "species",
                   "iconic_taxon_name": "Arachnida"},
            place_guess="Connecticut, USA",
        ))

        result = build_and_save_keywords(tmp_path)

        on_disk = read_json(metadata_dir(tmp_path) / "keywords.json")
        assert on_disk == result
        assert result["source_taxon"] == "Araneus diadematus"
        assert result["source_observation_id"] == 123
        assert "spider" in result["keywords"]
        assert "macro" in result["keywords"]  # Arachnida is in stock._MACRO_TAXA


class TestExportIdempotenceAndFailure:
    def test_running_export_twice_keeps_one_row_per_file(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)

        def fake_render(raw_path, dest_path, crop=None):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "export_inat_jpeg", fake_render)

        export_inat(tmp_path)
        export_inat(tmp_path)

        rows = sorted(a.file for a in scan_assets(tmp_path) if a.role == "upload")
        assert rows == ["export/inat/DSC_0001.jpg", "export/inat/DSC_0002.jpg"]

    def test_failure_on_second_file_keeps_the_first_files_row(self, tmp_path, monkeypatch):
        _seed_two_kept_one_rejected(tmp_path)

        def flaky_render(raw_path, dest_path, crop=None):
            if raw_path.name == "DSC_0002.NEF":
                raise RuntimeError("boom")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")

        monkeypatch.setattr(pipeline, "export_inat_jpeg", flaky_render)

        with pytest.raises(RuntimeError, match="boom"):
            export_inat(tmp_path)

        rows = [a.file for a in scan_assets(tmp_path) if a.role == "upload"]
        assert rows == ["export/inat/DSC_0001.jpg"]


class TestSetAssetCropValidation:
    def _seed(self, tmp_path):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [Asset(file="source/raw/A.NEF", role="original", status="kept", selected=True)])

    @pytest.mark.parametrize("bad,msg", [
        (["a", 0, 1, 1], "crop values must be finite numbers"),
        ([float("nan"), 0, 1, 1], "crop values must be finite numbers"),
        ([0, 0, 1.5, 1], "crop values must be within"),
        ([0.6, 0, 0.4, 1], "crop must satisfy x0 < x1 and y0 < y1"),
        ([0.2, 0.2, 0.2, 0.9], "crop must satisfy x0 < x1 and y0 < y1"),
    ])
    def test_bad_crop_is_rejected_and_not_persisted(self, tmp_path, bad, msg):
        self._seed(tmp_path)
        with pytest.raises(ValueError, match=msg):
            pipeline.set_asset_crop(tmp_path, "A.NEF", bad)
        assert scan_assets(tmp_path)[0].crop is None

    def test_valid_crop_is_stored(self, tmp_path):
        self._seed(tmp_path)
        pipeline.set_asset_crop(tmp_path, "A.NEF", [0.1, 0.2, 0.8, 0.9])
        assert scan_assets(tmp_path)[0].crop == [0.1, 0.2, 0.8, 0.9]
