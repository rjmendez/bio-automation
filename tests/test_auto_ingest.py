"""Unit tests for bio_automation.auto_ingest (hermetic -- read_capture_info,
this module's own real-EXIF-read dependency, is monkeypatched for the
grouping/volume-detection tests below; it is proven for real, against real
NEF fixtures, in tests/test_auto_ingest_live.py instead of being reimplemented
or stubbed-but-still-exercised here)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from bio_automation import auto_ingest
from bio_automation.auto_ingest import (
    CaptureInfo,
    ExifReadError,
    NotACameraVolumeError,
    UNKNOWN_CAMERA_SLUG,
    auto_ingest_volume,
    camera_model_slug,
    group_by_capture,
    ingest_capture_group,
    is_camera_volume,
    upsert_assets,
    upsert_observation,
)
from bio_automation.models import Asset, Observation
from bio_automation.pipeline import (
    ensure_layout,
    load_observation,
    save_assets,
    save_observation,
    scan_assets,
)


# --------------------------------------------------------------- is_camera_volume

class TestIsCameraVolume:
    def test_top_level_dcim_folder_is_a_camera_source(self, tmp_path):
        (tmp_path / "DCIM").mkdir()
        assert is_camera_volume(tmp_path) is True

    def test_lowercase_dcim_folder_is_still_detected(self, tmp_path):
        (tmp_path / "dcim").mkdir()
        assert is_camera_volume(tmp_path) is True

    def test_raw_files_without_dcim_folder_are_a_camera_source(self, tmp_path):
        (tmp_path / "DSC_0001.NEF").write_bytes(b"x")
        assert is_camera_volume(tmp_path) is True

    def test_unrelated_drive_with_no_images_is_not_a_camera_source(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "backups").mkdir()
        (tmp_path / "backups" / "archive.zip").write_bytes(b"x")
        assert is_camera_volume(tmp_path) is False

    def test_nonexistent_path_is_not_a_camera_source(self, tmp_path):
        assert is_camera_volume(tmp_path / "does_not_exist") is False

    def test_file_path_is_not_a_camera_source(self, tmp_path):
        f = tmp_path / "a_file.txt"
        f.write_text("x")
        assert is_camera_volume(f) is False


# ------------------------------------------------------------- camera_model_slug

class TestCameraModelSlug:
    def test_real_camera_model_slugifies_exact(self):
        assert camera_model_slug("NIKON Z 7_2") == "nikon-z-7-2"

    def test_different_model_produces_a_different_slug(self):
        assert camera_model_slug("Canon EOS R5") == "canon-eos-r5"

    def test_none_model_falls_back_to_unknown_camera(self):
        assert camera_model_slug(None) == UNKNOWN_CAMERA_SLUG

    def test_blank_model_falls_back_to_unknown_camera(self):
        assert camera_model_slug("   ") == UNKNOWN_CAMERA_SLUG


# --------------------------------------------------------------- group_by_capture

class TestGroupByCapture:
    def test_same_date_and_model_are_grouped_together(self, monkeypatch):
        info_by_path = {
            Path("a.NEF"): CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
            Path("b.NEF"): CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
        }
        monkeypatch.setattr(auto_ingest, "read_capture_info", lambda p: info_by_path[p])

        groups = group_by_capture([Path("a.NEF"), Path("b.NEF")])

        assert groups == {(dt.date(2025, 8, 2), "NIKON Z 7_2"): [Path("a.NEF"), Path("b.NEF")]}

    def test_different_capture_date_splits_into_separate_groups(self, monkeypatch):
        """Negative/contrast twin of the test above, same model, same 3
        files: a differing capture date must land in its OWN group, proving
        this is real per-file day grouping, not one group for the whole
        card and not dated by today."""
        info_by_path = {
            Path("a.NEF"): CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
            Path("b.NEF"): CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
            Path("c.NEF"): CaptureInfo(dt.date(2025, 8, 3), "NIKON Z 7_2"),
        }
        monkeypatch.setattr(auto_ingest, "read_capture_info", lambda p: info_by_path[p])

        groups = group_by_capture([Path("a.NEF"), Path("b.NEF"), Path("c.NEF")])

        assert groups == {
            (dt.date(2025, 8, 2), "NIKON Z 7_2"): [Path("a.NEF"), Path("b.NEF")],
            (dt.date(2025, 8, 3), "NIKON Z 7_2"): [Path("c.NEF")],
        }

    def test_different_camera_model_same_date_also_splits_into_separate_groups(self, monkeypatch):
        info_by_path = {
            Path("a.NEF"): CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
            Path("b.CR2"): CaptureInfo(dt.date(2025, 8, 2), "Canon EOS R5"),
        }
        monkeypatch.setattr(auto_ingest, "read_capture_info", lambda p: info_by_path[p])

        groups = group_by_capture([Path("a.NEF"), Path("b.CR2")])

        assert groups == {
            (dt.date(2025, 8, 2), "NIKON Z 7_2"): [Path("a.NEF")],
            (dt.date(2025, 8, 2), "Canon EOS R5"): [Path("b.CR2")],
        }

    def test_unreadable_exif_raises_exif_read_error_not_silently_grouped(self, monkeypatch):
        def fake_read(path):
            raise ExifReadError(f"{path}: no DateTimeOriginal/DateTime EXIF tag found")

        monkeypatch.setattr(auto_ingest, "read_capture_info", fake_read)

        with pytest.raises(ExifReadError, match=r"no DateTimeOriginal/DateTime EXIF tag found"):
            group_by_capture([Path("broken.NEF")])


# ----------------------------------------------------------------- upsert_assets

class TestUpsertAssets:
    def test_adds_all_candidates_when_assets_jsonl_is_empty(self, tmp_path):
        ensure_layout(tmp_path)
        candidates = [
            Asset(file="source/raw/a.NEF", role="original", status="ingested", selected=False),
            Asset(file="source/raw/b.NEF", role="original", status="ingested", selected=False),
        ]

        result = upsert_assets(tmp_path, candidates)

        assert result == {"added": ["source/raw/a.NEF", "source/raw/b.NEF"], "skipped": []}
        assert scan_assets(tmp_path) == candidates

    def test_reinsertion_preserves_kept_status_and_adds_only_the_new_file(self, tmp_path):
        """The required idempotent-merge proof: a pre-existing 'kept' asset
        must survive a second auto-ingest pass against the same + one new
        file untouched, and only the genuinely new file is added."""
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/a.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/b.NEF", role="original", status="ingested", selected=False),
        ])

        candidates = [
            Asset(file="source/raw/a.NEF", role="original", status="ingested", selected=False),
            Asset(file="source/raw/b.NEF", role="original", status="ingested", selected=False),
            Asset(file="source/raw/c.NEF", role="original", status="ingested", selected=False),
        ]

        result = upsert_assets(tmp_path, candidates)

        assert result == {"added": ["source/raw/c.NEF"], "skipped": ["source/raw/a.NEF", "source/raw/b.NEF"]}
        assets_by_file = {a.file: a for a in scan_assets(tmp_path)}
        assert assets_by_file["source/raw/a.NEF"] == Asset(
            file="source/raw/a.NEF", role="original", status="kept", selected=True,
        )
        assert assets_by_file["source/raw/b.NEF"] == Asset(
            file="source/raw/b.NEF", role="original", status="ingested", selected=False,
        )
        assert assets_by_file["source/raw/c.NEF"] == Asset(
            file="source/raw/c.NEF", role="original", status="ingested", selected=False,
        )
        assert len(assets_by_file) == 3

    def test_no_new_candidates_does_not_write_assets_jsonl_at_all(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/a.NEF", role="original", status="kept", selected=True),
        ])
        calls = []
        monkeypatch.setattr(auto_ingest, "save_assets", lambda folder, assets: calls.append((folder, assets)))

        result = upsert_assets(tmp_path, [
            Asset(file="source/raw/a.NEF", role="original", status="ingested", selected=False),
        ])

        assert result == {"added": [], "skipped": ["source/raw/a.NEF"]}
        assert calls == []  # save_assets must not be called when nothing new


# ------------------------------------------------------------- upsert_observation

class TestUpsertObservation:
    def test_first_insert_writes_proposed_observation_as_is(self, tmp_path):
        ensure_layout(tmp_path)
        proposed = Observation(obsdate="2025-08-02", source="auto-ingest", place_guess="F:\\Bio\\2025")

        result = upsert_observation(tmp_path, proposed)

        assert result == proposed
        assert load_observation(tmp_path) == proposed

    def test_existing_linked_inat_observation_id_is_never_overwritten(self, tmp_path):
        """The merge-safety proof this module exists for: a folder already
        linked to a real iNat observation must keep that link (and every
        other already-set field) exactly as found on a second auto-ingest
        pass, even though the newly-proposed record differs."""
        ensure_layout(tmp_path)
        save_observation(tmp_path, Observation(
            observation_id=999, obsdate="2025-08-02", source="iNaturalist",
            place_guess="Connecticut, USA", quality_grade="research",
        ))
        proposed = Observation(obsdate="2025-08-02", source="auto-ingest", place_guess="F:\\Bio\\2025")

        result = upsert_observation(tmp_path, proposed)

        assert result == Observation(
            observation_id=999, obsdate="2025-08-02", source="iNaturalist",
            place_guess="Connecticut, USA", quality_grade="research",
        )
        assert load_observation(tmp_path) == result

    def test_fields_still_at_their_dataclass_default_get_filled_in(self, tmp_path):
        """Positive twin of the test above: a record where obsdate/source/
        place_guess are still untouched (at Observation's own defaults) DOES
        get filled in from the proposed record -- the merge only refuses to
        clobber a field something else already set, it is not a no-op."""
        ensure_layout(tmp_path)
        save_observation(tmp_path, Observation())  # every field at its dataclass default
        proposed = Observation(obsdate="2025-08-02", source="auto-ingest", place_guess="F:\\Bio\\2025")

        result = upsert_observation(tmp_path, proposed)

        assert result == proposed
        assert load_observation(tmp_path) == proposed


# ----------------------------------------------------------- ingest_capture_group

class TestIngestCaptureGroup:
    def test_creates_folder_copies_files_and_records_new_assets(self, tmp_path, monkeypatch):
        base_dir = tmp_path / "Bio"
        source_a = tmp_path / "card" / "A.NEF"
        source_b = tmp_path / "card" / "B.NEF"
        source_a.parent.mkdir(parents=True)
        source_a.write_bytes(b"raw-a")
        source_b.write_bytes(b"raw-b")

        result = ingest_capture_group(base_dir, dt.date(2025, 8, 2), "NIKON Z 7_2", [source_a, source_b])

        observation_dir = base_dir / "2025" / "2025-08-02" / "obs-20250802-nikon-z-7-2"
        assert result["folder"] == str(observation_dir)
        assert result["obsdate"] == "2025-08-02"
        assert result["camera_model"] == "NIKON Z 7_2"
        assert result["folder_is_new"] is True
        assert sorted(result["new_files"]) == ["source/raw/A.NEF", "source/raw/B.NEF"]
        assert result["already_present"] == []
        assert (observation_dir / "source" / "raw" / "A.NEF").read_bytes() == b"raw-a"
        assert (observation_dir / "source" / "raw" / "B.NEF").read_bytes() == b"raw-b"
        observation = load_observation(observation_dir)
        assert observation.obsdate == "2025-08-02"
        assert observation.source == "auto-ingest"

    def test_rerunning_against_an_existing_folder_reports_folder_is_new_false(self, tmp_path):
        base_dir = tmp_path / "Bio"
        source_a = tmp_path / "card" / "A.NEF"
        source_a.parent.mkdir(parents=True)
        source_a.write_bytes(b"raw-a")

        first = ingest_capture_group(base_dir, dt.date(2025, 8, 2), "NIKON Z 7_2", [source_a])
        assert first["folder_is_new"] is True

        second = ingest_capture_group(base_dir, dt.date(2025, 8, 2), "NIKON Z 7_2", [source_a])
        assert second["folder_is_new"] is False
        assert second["new_files"] == []
        assert second["already_present"] == ["source/raw/A.NEF"]


# ------------------------------------------------------------- auto_ingest_volume

class TestAutoIngestVolume:
    def test_raises_not_a_camera_volume_error_for_unrelated_drive(self, tmp_path):
        volume = tmp_path / "volume"
        volume.mkdir()
        (volume / "readme.txt").write_text("hello")

        with pytest.raises(NotACameraVolumeError, match=r"does not look like a camera/card source"):
            auto_ingest_volume(volume, tmp_path / "Bio")

    def test_real_camera_volume_with_no_raw_files_yet_does_not_raise(self, tmp_path):
        """Positive twin of the error-path test above, same shape of call:
        a volume that DOES pass is_camera_volume() (has DCIM) must not
        raise, even if it happens to hold zero raw files right now."""
        volume = tmp_path / "volume"
        (volume / "DCIM").mkdir(parents=True)

        result = auto_ingest_volume(volume, tmp_path / "Bio")

        assert result["groups"] == []
        assert result["observation_folders_created"] == 0
        assert result["files_ingested"] == 0

    def test_two_groups_on_one_card_each_get_their_own_folder(self, tmp_path, monkeypatch):
        volume = tmp_path / "volume"
        dcim = volume / "DCIM" / "100NIKON"
        dcim.mkdir(parents=True)
        (dcim / "A.NEF").write_bytes(b"day1")
        (dcim / "B.NEF").write_bytes(b"day2")

        info_by_name = {
            "A.NEF": CaptureInfo(dt.date(2025, 8, 2), "NIKON Z 7_2"),
            "B.NEF": CaptureInfo(dt.date(2025, 8, 3), "NIKON Z 7_2"),
        }
        monkeypatch.setattr(auto_ingest, "read_capture_info", lambda p: info_by_name[p.name])

        base_dir = tmp_path / "Bio"
        result = auto_ingest_volume(volume, base_dir)

        assert result["observation_folders_created"] == 2
        assert result["observation_folders_updated"] == 0
        assert result["files_ingested"] == 2
        assert result["files_already_present"] == 0
        folders = sorted(g["folder"] for g in result["groups"])
        assert folders == [
            str(base_dir / "2025" / "2025-08-02" / "obs-20250802-nikon-z-7-2"),
            str(base_dir / "2025" / "2025-08-03" / "obs-20250803-nikon-z-7-2"),
        ]

        # The base directory must not be stored as the place.
        for group in result["groups"]:
            assert load_observation(Path(group["folder"])).place_guess is None
