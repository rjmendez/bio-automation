from __future__ import annotations

import json

import pytest

from bio_automation.stack_sync import (
    ManifestError,
    find_new_synced_stacks,
    ingest_synced_stack,
    process_synced_stacks,
)
from bio_automation.pipeline import load_observation, scan_assets
from bio_automation.paths import metadata_dir, source_raw_dir


def _write_stack(sync_dir, name, *, files=("DSC_0001.NEF", "DSC_0002.NEF"),
                  stack_id=None, captured_at="2026-10-19T17:34:23",
                  camera_model="NIKON Z 7_2", fstop=11.0, content=b"RAW"):
    d = sync_dir / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(content * 100)
    manifest = {
        "stack_id": stack_id or name,
        "captured_at": captured_at,
        "camera_model": camera_model,
        "fstop": fstop,
        "shot_count": len(files),
        "files": list(files),
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


class TestFindNewSyncedStacks:
    def test_missing_sync_dir_returns_empty(self, tmp_path):
        assert find_new_synced_stacks(tmp_path / "nope") == []

    def test_finds_a_real_complete_stack(self, tmp_path):
        d = _write_stack(tmp_path, "20261019T173423-abcd1234")
        assert find_new_synced_stacks(tmp_path) == [d]

    def test_excludes_incoming_staging_dirs(self, tmp_path):
        _write_stack(tmp_path, ".incoming-stack1")
        assert find_new_synced_stacks(tmp_path) == []

    def test_excludes_dirs_without_manifest(self, tmp_path):
        (tmp_path / "not-a-stack").mkdir()
        assert find_new_synced_stacks(tmp_path) == []

    def test_excludes_already_ingested(self, tmp_path):
        d = _write_stack(tmp_path, "stack1")
        (d / ".ingested").write_text("")
        assert find_new_synced_stacks(tmp_path) == []

    def test_positive_twin_not_yet_ingested_is_included(self, tmp_path):
        d = _write_stack(tmp_path, "stack1")
        assert find_new_synced_stacks(tmp_path) == [d]


class TestIngestSyncedStack:
    def test_raises_on_missing_manifest(self, tmp_path):
        d = tmp_path / "no-manifest"
        d.mkdir()
        with pytest.raises(ManifestError, match="manifest.json"):
            ingest_synced_stack(d, tmp_path / "bio")

    def test_creates_real_observation_with_exact_fields(self, tmp_path):
        stack_dir = _write_stack(
            tmp_path / "sync", "stack1",
            stack_id="20261019T173423-abcd1234",
            captured_at="2026-10-19T17:34:23", camera_model="NIKON Z 7_2", fstop=11.0,
        )
        base_dir = tmp_path / "bio"
        observation_dir = ingest_synced_stack(stack_dir, base_dir)

        assert observation_dir == base_dir / "2026" / "2026-10-19" / "obs-20261019-nikon-stack-abcd1234"
        obs = load_observation(observation_dir)
        assert obs.obsdate == "2026-10-19"
        assert obs.source == "nikon-stack-sync"

        assets = scan_assets(observation_dir)
        assert sorted(a.file for a in assets) == [
            "source/raw/DSC_0001.NEF", "source/raw/DSC_0002.NEF",
        ]
        assert all(a.status == "ingested" for a in assets)

    def test_raw_file_bytes_are_exact(self, tmp_path):
        stack_dir = _write_stack(tmp_path / "sync", "stack1", content=b"REALDATA")
        base_dir = tmp_path / "bio"
        observation_dir = ingest_synced_stack(stack_dir, base_dir)
        copied = (source_raw_dir(observation_dir) / "DSC_0001.NEF").read_bytes()
        assert copied == b"REALDATA" * 100

    def test_rerun_on_already_curated_folder_never_resets_status(self, tmp_path):
        """The idempotent-merge guarantee auto_ingest.py already proved --
        confirmed here too since stack_sync reruns through the same path."""
        stack_dir = _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"
        observation_dir = ingest_synced_stack(stack_dir, base_dir)

        from bio_automation.pipeline import set_asset_status
        set_asset_status(observation_dir, ["source/raw/DSC_0001.NEF"], "kept", True)

        ingest_synced_stack(stack_dir, base_dir)  # rerun, same stack

        assets = {a.file: a for a in scan_assets(observation_dir)}
        assert assets["source/raw/DSC_0001.NEF"].status == "kept"
        assert assets["source/raw/DSC_0001.NEF"].selected is True


class TestProcessSyncedStacks:
    def test_ingests_and_calls_processor_then_cleans_up_by_default(self, tmp_path):
        stack_dir = _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"
        calls = []

        def recording_processor(observation_dir, **kwargs):
            calls.append(observation_dir)

        results = process_synced_stacks(
            tmp_path / "sync", base_dir, processor=recording_processor,
        )

        assert len(calls) == 1
        assert results == [{
            "stack_dir": str(stack_dir),
            "observation_dir": str(calls[0]),
            "status": "ok",
        }]
        assert not stack_dir.exists()  # cleaned up -- D: is scratch, not retention

    def test_no_cleanup_keeps_landing_copy_and_marks_ingested(self, tmp_path):
        stack_dir = _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"

        process_synced_stacks(
            tmp_path / "sync", base_dir,
            processor=lambda observation_dir, **kw: None, cleanup=False,
        )

        assert stack_dir.exists()
        assert (stack_dir / ".ingested").exists()

    def test_processor_failure_leaves_stack_for_retry_negative_twin(self, tmp_path):
        stack_dir = _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"

        def failing_processor(observation_dir, **kwargs):
            raise RuntimeError("stacking engine exploded")

        results = process_synced_stacks(
            tmp_path / "sync", base_dir, processor=failing_processor,
        )

        assert results[0]["status"] == "error"
        assert "stacking engine exploded" in results[0]["error"]
        assert stack_dir.exists()  # never deleted on failure
        assert not (stack_dir / ".ingested").exists()
        # A second call with a working processor must still pick it up.
        calls = []
        process_synced_stacks(
            tmp_path / "sync", base_dir,
            processor=lambda observation_dir, **kw: calls.append(observation_dir),
        )
        assert len(calls) == 1

    def test_already_ingested_stack_is_skipped(self, tmp_path):
        stack_dir = _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"
        calls = []

        def recording_processor(observation_dir, **kwargs):
            calls.append(observation_dir)

        process_synced_stacks(tmp_path / "sync", base_dir,
                               processor=recording_processor, cleanup=False)
        process_synced_stacks(tmp_path / "sync", base_dir,
                               processor=recording_processor, cleanup=False)

        assert len(calls) == 1  # second run found nothing new

    def test_passes_through_process_kwargs_to_the_real_processor(self, tmp_path):
        _write_stack(tmp_path / "sync", "stack1")
        base_dir = tmp_path / "bio"
        received = {}

        def recording_processor(observation_dir, **kwargs):
            received.update(kwargs)

        process_synced_stacks(
            tmp_path / "sync", base_dir, processor=recording_processor,
            method="focus-stack", exposure_ev=0.5,
        )
        assert received == {"method": "focus-stack", "exposure_ev": 0.5}
