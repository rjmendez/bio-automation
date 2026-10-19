"""Live, unmocked proof that bio_automation.auto_ingest's real EXIF read
(read_capture_info, via Pillow -- see that module's docstring for why
Pillow and not exiftool/rawpy) and the idempotent-merge path actually work
against real raw files -- not gated behind an opt-in env var or marker,
runs by default on every `pytest` invocation (see AGENTS.md rule 10 and
tests/test_preclean_live.py's own docstring for why).

Cheap by construction: this only reads each real NEF's own TIFF/EXIF header
and copies its bytes -- no raw decode, no model inference -- so it adds
a negligible amount of time to the suite, unlike test_preclean_live.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from bio_automation.auto_ingest import (
    CaptureInfo,
    auto_ingest_volume,
    group_by_capture,
    read_capture_info,
)
from bio_automation.models import Asset
from bio_automation.pipeline import scan_assets, set_asset_status

SPIDER = Path(__file__).parent / "fixtures" / "sample_raw.NEF"
FUNGUS = Path(__file__).parent / "fixtures" / "sample_raw_fungus.NEF"


class TestReadCaptureInfoLive:
    def test_real_spider_fixture_exact_capture_date_and_camera_model(self):
        assert SPIDER.exists(), f"missing committed fixture: {SPIDER}"
        info = read_capture_info(SPIDER)
        assert info == CaptureInfo(capture_date=dt.date(2025, 8, 2), camera_model="NIKON Z 7_2")

    def test_real_fungus_fixture_exact_capture_date_and_camera_model(self):
        """Positive twin of the spider-fixture test above, on the other
        real committed fixture -- both real files, same real camera, same
        real day (2026-10-19 live measurement, see grouping.py's module
        docstring for the incident both fixtures come from)."""
        assert FUNGUS.exists(), f"missing committed fixture: {FUNGUS}"
        info = read_capture_info(FUNGUS)
        assert info == CaptureInfo(capture_date=dt.date(2025, 8, 2), camera_model="NIKON Z 7_2")

    def test_real_fixtures_share_one_group_end_to_end(self):
        """Through the real, public group_by_capture() -- not just the raw
        read_capture_info() values above -- proving the real (date, model)
        read integrates correctly with the grouping logic on real data."""
        groups = group_by_capture([SPIDER, FUNGUS])
        assert groups == {(dt.date(2025, 8, 2), "NIKON Z 7_2"): [SPIDER, FUNGUS]}


class TestAutoIngestVolumeIdempotentMergeLive:
    """The idempotent-merge proof demanded for this feature, run end to end
    against real raw bytes and real EXIF reads -- nothing stubbed: a
    pre-existing 'kept' asset must survive a second real auto-ingest pass
    against the same card plus one genuinely new file."""

    @staticmethod
    def _build_card(volume: Path) -> None:
        dcim = volume / "DCIM" / "100NIKON"
        dcim.mkdir(parents=True)
        (dcim / "A.NEF").write_bytes(SPIDER.read_bytes())
        (dcim / "B.NEF").write_bytes(FUNGUS.read_bytes())

    def test_first_pass_ingests_both_real_files_into_one_real_group(self, tmp_path):
        volume = tmp_path / "card"
        self._build_card(volume)
        base_dir = tmp_path / "Bio"

        result = auto_ingest_volume(volume, base_dir)

        assert result["observation_folders_created"] == 1
        assert result["observation_folders_updated"] == 0
        assert result["files_ingested"] == 2
        assert result["files_already_present"] == 0
        [group] = result["groups"]
        observation_dir = Path(group["folder"])
        assert observation_dir == base_dir / "2025" / "2025-08-02" / "obs-20250802-nikon-z-7-2"
        assets_by_file = {a.file: a for a in scan_assets(observation_dir)}
        assert assets_by_file == {
            "source/raw/A.NEF": Asset(file="source/raw/A.NEF", role="original",
                                       status="ingested", selected=False),
            "source/raw/B.NEF": Asset(file="source/raw/B.NEF", role="original",
                                       status="ingested", selected=False),
        }

    def test_reinsertion_with_one_kept_asset_and_one_new_file_merges_correctly(self, tmp_path):
        volume = tmp_path / "card"
        self._build_card(volume)
        base_dir = tmp_path / "Bio"

        first = auto_ingest_volume(volume, base_dir)
        [group] = first["groups"]
        observation_dir = Path(group["folder"])

        # A human curates: A.NEF is a keeper.
        set_asset_status(observation_dir, ["A.NEF"], "kept", True)

        # The same card comes back with one genuinely new file (same real
        # subject/day/camera -- a byte-identical copy of the real spider
        # fixture under a new name, so it is real, independently-readable
        # EXIF, not a synthetic row).
        (volume / "DCIM" / "100NIKON" / "C.NEF").write_bytes(SPIDER.read_bytes())

        second = auto_ingest_volume(volume, base_dir)

        assert second["observation_folders_created"] == 0
        assert second["observation_folders_updated"] == 1
        assert second["files_ingested"] == 1
        assert second["files_already_present"] == 2

        assets_by_file = {a.file: a for a in scan_assets(observation_dir)}
        assert assets_by_file == {
            "source/raw/A.NEF": Asset(file="source/raw/A.NEF", role="original",
                                       status="kept", selected=True),
            "source/raw/B.NEF": Asset(file="source/raw/B.NEF", role="original",
                                       status="ingested", selected=False),
            "source/raw/C.NEF": Asset(file="source/raw/C.NEF", role="original",
                                       status="ingested", selected=False),
        }
