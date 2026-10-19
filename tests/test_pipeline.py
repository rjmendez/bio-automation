"""Unit tests for bio_automation.pipeline functions touched by Zerene wiring
(hermetic -- zerene.run_batch is monkeypatched, never actually launched)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bio_automation import preclean
from bio_automation.grouping import SubjectMismatchError
from bio_automation.io import read_json
from bio_automation.models import Asset
from bio_automation.pipeline import (
    clean_and_stack,
    discover_observation_folders,
    ensure_layout,
    save_assets,
    scan_assets,
    stack_all,
    stack_focus,
)
from bio_automation.paths import make_observation_path, source_raw_dir, work_dir
from bio_automation.zerene import ZereneError


class TestEnsureLayout:
    def test_creates_work_dir_alongside_existing_subdirs(self, tmp_path):
        ensure_layout(tmp_path)
        assert work_dir(tmp_path).is_dir()
        assert source_raw_dir(tmp_path).is_dir()


class RecordingBatchRun:
    """Stands in for zerene.run_batch: records calls, writes a fake output file."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, folder, method="pmax", timeout=1800.0, min_free_gb=5.0):
        self.calls.append({"folder": Path(folder), "method": method, "timeout": timeout,
                            "min_free_gb": min_free_gb})
        out = Path(folder) / "ZS-OutputImage.tif"
        out.write_bytes(b"stacked")
        return out


class RecordingSubjectCheck:
    """Stands in for grouping.check_subject_consistency: records calls and
    returns a configured (default: empty) outlier list."""

    def __init__(self, outliers: list[tuple[Path, float]] | None = None):
        self.calls: list[dict] = []
        self.outliers = outliers if outliers is not None else []

    def __call__(self, raw_paths, session, threshold=0.75):
        self.calls.append({"raw_paths": list(raw_paths), "session": session, "threshold": threshold})
        return self.outliers


class TestStackFocus:
    def _prep(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        raw_dir = source_raw_dir(tmp_path)
        leftover_script = raw_dir / "ZereneBatch.xml"
        leftover_script.write_text("<leftover/>", encoding="utf-8")
        fake_run = RecordingBatchRun()
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_run)
        # The subject-consistency gate is a separate concern (see
        # TestSubjectConsistencyGate below); stub it out here so these
        # Zerene-wiring tests are unaffected by the gate being on by default.
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))
        return fake_run, leftover_script

    def test_calls_zerene_with_raw_dir_method_and_timeout(self, tmp_path, monkeypatch):
        fake_run, _ = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, method="pmax", timeout=42.0)
        assert fake_run.calls == [
            {"folder": source_raw_dir(tmp_path), "method": "pmax", "timeout": 42.0,
             "min_free_gb": 5.0}
        ]

    def test_passes_through_non_default_min_free_gb(self, tmp_path, monkeypatch):
        fake_run, _ = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, min_free_gb=12.5)
        assert fake_run.calls == [
            {"folder": source_raw_dir(tmp_path), "method": "pmax", "timeout": 1800.0,
             "min_free_gb": 12.5}
        ]

    def test_moves_output_into_work_dir_not_raw_dir(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path)
        assert (work_dir(tmp_path) / "ZS-OutputImage.tif").exists()
        assert not (source_raw_dir(tmp_path) / "ZS-OutputImage.tif").exists()

    def test_removes_leftover_batch_script_from_raw_dir(self, tmp_path, monkeypatch):
        _, leftover_script = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path)
        assert not leftover_script.exists()

    def test_appends_exact_stack_output_asset_and_returns_it(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        returned = stack_focus(tmp_path)
        expected = Asset(file="work/ZS-OutputImage.tif", role="stack_output",
                          status="ready", selected=True)
        assert returned == expected
        assets = scan_assets(tmp_path)
        assert assets == [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            expected,
        ]


class RecordingDenoise:
    """Stands in for preclean.denoise_stack_frames: records calls, writes fake TIFFs."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, raw_paths, dest_dir, noise=1.0, exposure_ev=0.0, cache=None,
                 min_free_gb=5.0):
        self.calls.append({
            "raw_paths": list(raw_paths), "dest_dir": Path(dest_dir),
            "noise": noise, "exposure_ev": exposure_ev, "min_free_gb": min_free_gb,
        })
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for p in raw_paths:
            out = dest_dir / f"{Path(p).stem}.tif"
            out.write_bytes(b"denoised")
            written.append(out)
        return written


class TestCleanAndStack:
    def _prep(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        raw_a = raw_dir / "DSC_0001.NEF"
        raw_b = raw_dir / "DSC_0002.NEF"
        raw_a.write_bytes(b"a")
        raw_b.write_bytes(b"b")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        fake_denoise = RecordingDenoise()
        fake_run = RecordingBatchRun()
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_run)
        # See TestStackFocus._prep: the subject-consistency gate is a separate
        # concern (TestSubjectConsistencyGate below), stubbed out here too.
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))
        return raw_a, raw_b, fake_denoise, fake_run

    def test_denoises_all_raw_dir_files_into_work_denoised_with_exact_params(self, tmp_path, monkeypatch):
        raw_a, raw_b, fake_denoise, _ = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path, noise=0.7, exposure_ev=-0.5)

        assert len(fake_denoise.calls) == 1
        call = fake_denoise.calls[0]
        assert sorted(call["raw_paths"]) == sorted([raw_a, raw_b])
        assert call["dest_dir"] == work_dir(tmp_path) / "denoised"
        assert call["noise"] == 0.7
        assert call["exposure_ev"] == -0.5

    def test_zerene_runs_against_denoised_dir_not_raw_dir(self, tmp_path, monkeypatch):
        _, _, _, fake_run = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path)

        assert fake_run.calls == [
            {"folder": work_dir(tmp_path) / "denoised", "method": "pmax", "timeout": 1800.0,
             "min_free_gb": 5.0}
        ]

    def test_passes_non_default_min_free_gb_to_both_denoise_and_zerene(self, tmp_path, monkeypatch):
        _, _, fake_denoise, fake_run = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path, min_free_gb=9.0)

        assert fake_denoise.calls[0]["min_free_gb"] == 9.0
        assert fake_run.calls == [
            {"folder": work_dir(tmp_path) / "denoised", "method": "pmax", "timeout": 1800.0,
             "min_free_gb": 9.0}
        ]

    def test_appends_stack_output_asset_from_denoised_output(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        asset = clean_and_stack(tmp_path)
        assert asset == Asset(file="work/ZS-OutputImage.tif", role="stack_output",
                               status="ready", selected=True)
        assert (work_dir(tmp_path) / "ZS-OutputImage.tif").exists()


class TestSubjectConsistencyGate:
    """Covers the hard safety gate stack_focus()/clean_and_stack() run before
    handing off to Zerene (see bio_automation.grouping.check_subject_consistency).
    check_subject_consistency and load_session are both monkeypatched to
    controlled stubs -- never a real ONNX session or real images -- so these
    tests isolate the *wiring* (did the gate run, in the right order, with
    the right inputs, and did it actually block Zerene) from the gate's own
    embedding logic, which tests/test_grouping.py covers."""

    def _prep(self, tmp_path, monkeypatch, outliers=None):
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        raw_a = raw_dir / "DSC_0001.NEF"
        raw_b = raw_dir / "DSC_0002.NEF"
        raw_a.write_bytes(b"a")
        raw_b.write_bytes(b"b")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        fake_run = RecordingBatchRun()
        fake_check = RecordingSubjectCheck(outliers=outliers)
        fake_session = MagicMock(name="daclip-session")
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_run)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: fake_session)
        return raw_a, raw_b, fake_run, fake_check, fake_session

    def test_proceeds_to_zerene_when_no_outliers_found(self, tmp_path, monkeypatch):
        """Positive twin to test_raises_before_touching_zerene_when_outliers_found
        below: the empty-outliers case must let the real work (Zerene) run."""
        raw_a, raw_b, fake_run, fake_check, fake_session = self._prep(
            tmp_path, monkeypatch, outliers=[]
        )

        stack_focus(tmp_path)

        assert len(fake_check.calls) == 1
        assert sorted(fake_check.calls[0]["raw_paths"]) == sorted([raw_a, raw_b])
        assert fake_check.calls[0]["session"] is fake_session
        assert len(fake_run.calls) == 1

    def test_raises_before_touching_zerene_when_outliers_found(self, tmp_path, monkeypatch):
        outliers = [(Path("source/raw/DSC_0001.NEF"), 0.2)]
        _, _, fake_run, fake_check, _ = self._prep(tmp_path, monkeypatch, outliers=outliers)

        with pytest.raises(SubjectMismatchError, match=r"1 frame\(s\) fall below"):
            stack_focus(tmp_path)

        assert fake_run.calls == []

    def test_check_subjects_false_skips_gate_entirely(self, tmp_path, monkeypatch):
        """Negative twin to the two tests above: even outliers that would
        normally raise must never be consulted when check_subjects=False."""
        _, _, fake_run, fake_check, _ = self._prep(
            tmp_path, monkeypatch, outliers=[(Path("whatever.NEF"), 0.0)]
        )

        stack_focus(tmp_path, check_subjects=False)

        assert fake_check.calls == []
        assert len(fake_run.calls) == 1

    def test_missing_model_raises_clear_error_before_zerene(self, tmp_path, monkeypatch):
        """Design decision: a missing DA-CLIP model fails loudly (wrapped
        FileNotFoundError naming the check) rather than silently skipping
        the safety gate and proceeding to Zerene anyway."""
        _, _, fake_run, fake_check, _ = self._prep(tmp_path, monkeypatch, outliers=[])
        model_path = r"C:\TopazModels\daclip-v2-fp32-224x224-1x-ox\daclip-v2-fp32-224x224-1x.onnx"

        def _raise_missing():
            raise FileNotFoundError(f"DA-CLIP model not found: {model_path}")

        monkeypatch.setattr("bio_automation.pipeline.load_session", _raise_missing)

        with pytest.raises(FileNotFoundError, match="Subject-consistency check unavailable"):
            stack_focus(tmp_path)

        assert fake_run.calls == []
        assert fake_check.calls == []

    def test_clean_and_stack_forwards_check_subjects_true_and_raises(self, tmp_path, monkeypatch):
        """clean_and_stack() must thread its own check_subjects through to
        stack_focus() rather than always running (or always skipping) the
        gate regardless of what the caller asked for."""
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        (raw_dir / "DSC_0001.NEF").write_bytes(b"a")
        (raw_dir / "DSC_0002.NEF").write_bytes(b"b")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        fake_denoise = RecordingDenoise()
        fake_run = RecordingBatchRun()
        fake_check = RecordingSubjectCheck(outliers=[(Path("DSC_0001.NEF"), 0.1)])
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_run)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

        with pytest.raises(SubjectMismatchError, match=r"1 frame\(s\) fall below"):
            clean_and_stack(tmp_path)

        assert fake_run.calls == []
        assert len(fake_check.calls) == 1
        # Fails fast on the raw files, before paying for denoise at all --
        # not after denoising every frame and only then discovering the
        # mismatch inside stack_focus.
        assert fake_denoise.calls == []

    def test_clean_and_stack_forwards_check_subjects_false(self, tmp_path, monkeypatch):
        """Positive twin to the test above: the same outliers must not
        block anything when the caller explicitly disabled the gate."""
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        (raw_dir / "DSC_0001.NEF").write_bytes(b"a")
        (raw_dir / "DSC_0002.NEF").write_bytes(b"b")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        fake_denoise = RecordingDenoise()
        fake_run = RecordingBatchRun()
        fake_check = RecordingSubjectCheck(outliers=[(Path("DSC_0001.NEF"), 0.1)])
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_run)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

        clean_and_stack(tmp_path, check_subjects=False)

        assert fake_check.calls == []
        assert len(fake_run.calls) == 1


class TestCleanAndStackCleanup:
    """Covers clean_and_stack()'s own disposable-intermediate cleanup:
    work/denoised/ (the per-frame denoised TIFFs that fed Zerene) must be
    gone once stacking is done, whether it succeeded or failed. These tests
    stub clean_and_stack()'s own direct dependencies (denoise_stack_frames,
    zerene_run_batch, the subject-consistency gate) so clean_and_stack()'s
    own cleanup logic is what's exercised, not theirs."""

    def _prep(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        raw_a = raw_dir / "DSC_0001.NEF"
        raw_b = raw_dir / "DSC_0002.NEF"
        raw_a.write_bytes(b"a")
        raw_b.write_bytes(b"b")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", RecordingDenoise())
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

    def test_success_deletes_denoised_dir_but_keeps_stack_output(self, tmp_path, monkeypatch):
        """Positive twin to the two failure tests below, on the same
        fixture: a clean run must still leave the final stack_output asset
        in work/ in place even though work/denoised/ (the disposable
        per-frame intermediates) is removed."""
        self._prep(tmp_path, monkeypatch)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", RecordingBatchRun())

        clean_and_stack(tmp_path)

        assert not (work_dir(tmp_path) / "denoised").exists()
        assert (work_dir(tmp_path) / "ZS-OutputImage.tif").exists()

    def test_failure_from_zerene_deletes_denoised_dir_and_reraises_exact_exception(self, tmp_path, monkeypatch):
        """Negative twin to the test above, same fixture: zerene_run_batch
        failing (inside stack_focus) must still clean up work/denoised/,
        and the original ZereneError -- exact type and message -- must
        propagate unchanged, not be swallowed or replaced."""
        self._prep(tmp_path, monkeypatch)

        def failing_run(folder, method="pmax", timeout=1800.0, min_free_gb=5.0):
            raise ZereneError("Zerene exited with code 1: disk full")

        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", failing_run)

        with pytest.raises(ZereneError, match=r"^Zerene exited with code 1: disk full$") as exc_info:
            clean_and_stack(tmp_path)

        assert str(exc_info.value) == "Zerene exited with code 1: disk full"
        assert not (work_dir(tmp_path) / "denoised").exists()

    def test_failure_from_denoise_deletes_partial_denoised_dir_and_reraises_exact_exception(self, tmp_path, monkeypatch):
        """Second negative twin, same fixture: a failure inside
        denoise_stack_frames itself (before stack_focus ever runs, so
        work/denoised/ may hold only some of the frames) must also leave no
        partial output behind, and must propagate the exact original
        exception -- stack_focus/zerene never even run."""
        self._prep(tmp_path, monkeypatch)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", RecordingBatchRun())

        def failing_denoise(raw_paths, dest_dir, noise=1.0, exposure_ev=0.0, cache=None,
                             min_free_gb=5.0):
            dest_dir = Path(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            (dest_dir / "DSC_0001.tif").write_bytes(b"partial")  # frame 1 written...
            raise OSError("Insufficient disk space: 1.0 GB free < 5.0 GB required")  # ...frame 2 fails

        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", failing_denoise)

        with pytest.raises(OSError, match=r"^Insufficient disk space: 1\.0 GB free < 5\.0 GB required$") as exc_info:
            clean_and_stack(tmp_path)

        assert str(exc_info.value) == "Insufficient disk space: 1.0 GB free < 5.0 GB required"
        assert not (work_dir(tmp_path) / "denoised").exists()


class TestCleanAndStackCacheReuse:
    """Proves the cache param is real reuse, not just accepted-and-ignored:
    when the caller passes an explicit ModelCache through clean_and_stack(),
    denoise_stack_frames() must never construct its own. Runs the *real*
    denoise_stack_frames() (only its own dependencies -- rawpy decode and
    the OpenVINO-backed model estimate/apply calls -- are stubbed, same
    pattern as tests/test_preclean.py), so the "own_cache = cache is None"
    branch in preclean.py is genuinely exercised, not assumed."""

    def test_explicit_cache_means_denoise_stack_frames_never_builds_its_own(self, tmp_path, monkeypatch):
        ensure_layout(tmp_path)
        raw_dir = source_raw_dir(tmp_path)
        (raw_dir / "DSC_0001.NEF").write_bytes(b"a")
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])

        monkeypatch.setattr(preclean, "decode_raw_to_bgr16",
                             lambda p: np.zeros((1, 1, 3), dtype=np.uint16))
        monkeypatch.setattr(preclean, "_estimate_sharpen_params",
                             lambda chw, cache: (0.1, 0.5, 0.3, 0.0))
        monkeypatch.setattr(preclean, "_sharpen_f32", lambda chw, cache, blur, noise: chw)

        construct_calls: list[tuple] = []

        class RecordingModelCache:
            def __init__(self, *args, **kwargs):
                construct_calls.append((args, kwargs))

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(preclean, "ModelCache", RecordingModelCache)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", RecordingBatchRun())
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

        explicit_cache = MagicMock(name="shared-model-cache")
        clean_and_stack(tmp_path, cache=explicit_cache)

        assert construct_calls == []


class TestDiscoverObservationFolders:
    def test_finds_only_folders_that_hold_metadata_assets_jsonl(self, tmp_path):
        """Positive case (the two real observation folders) and negative
        case (a stray directory with no metadata/assets.jsonl) on the same
        fixture, since both must be told apart by one call."""
        folder_a = make_observation_path(tmp_path, "2026-01-05", "spider")
        folder_b = make_observation_path(tmp_path, "2026-02-09", "fungus")
        ensure_layout(folder_a)
        ensure_layout(folder_b)
        save_assets(folder_a, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        save_assets(folder_b, [
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        # A directory that merely looks like it could be one (same tree,
        # similar name) but has no metadata/assets.jsonl must not count.
        stray = tmp_path / "2026" / "not-an-observation"
        stray.mkdir(parents=True)

        found = discover_observation_folders(tmp_path)

        assert found == sorted([folder_a, folder_b])


class RecordingCleanAndStack:
    """Stands in for pipeline.clean_and_stack: records every call (folder
    and the cache object it was given) in call order, and raises or
    succeeds per folder according to *behaviors* (keyed by folder name)."""

    def __init__(self, behaviors: dict[str, Exception | None]):
        self.calls: list[dict] = []
        self.behaviors = behaviors

    def __call__(self, folder, noise=1.0, exposure_ev=0.0, method="pmax", timeout=1800.0,
                 min_free_gb=5.0, check_subjects=True, cache=None):
        folder = Path(folder)
        self.calls.append({
            "folder": folder, "noise": noise, "exposure_ev": exposure_ev, "method": method,
            "timeout": timeout, "min_free_gb": min_free_gb, "check_subjects": check_subjects,
            "cache": cache,
        })
        behavior = self.behaviors[folder.name]
        if behavior is not None:
            raise behavior
        return Asset(file="work/ZS-OutputImage.tif", role="stack_output",
                      status="ready", selected=True)


class RecordingModelCache:
    """Stands in for topaz.ModelCache: records every construction so tests
    can prove exactly one was built for a whole stack_all() batch."""

    instances_built: list["RecordingModelCache"] = []

    def __init__(self, *args, **kwargs):
        RecordingModelCache.instances_built.append(self)
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


class TestStackAll:
    """stack_all() orchestrates cull + clean_and_stack() over every
    observation folder under a base_dir, sharing one ModelCache and
    surviving per-folder failures. clean_and_stack() itself is stubbed
    (its own behavior is covered by TestCleanAndStack above); these tests
    isolate the *batch orchestration* -- discovery, cache sharing, cull,
    per-folder exception handling, and the manifest it writes."""

    def _prep(self, tmp_path, monkeypatch, behaviors):
        RecordingModelCache.instances_built = []
        folder_ok = tmp_path / "obs-a"
        folder_zerene_fail = tmp_path / "obs-b"
        folder_other_fail = tmp_path / "obs-c"
        for folder in (folder_ok, folder_zerene_fail, folder_other_fail):
            ensure_layout(folder)
            save_assets(folder, [
                Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            ])
        # A rejected raw actually on disk in the folder that succeeds, to
        # prove cull (delete_rejected_raws) really ran per folder rather
        # than being skipped -- not stubbed, so this is real deletion.
        rejected_raw = source_raw_dir(folder_ok) / "DSC_bad.NEF"
        rejected_raw.write_bytes(b"bad frame")
        save_assets(folder_ok, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_bad.NEF", role="original", status="rejected", selected=False),
        ])

        fake_clean_and_stack = RecordingCleanAndStack(behaviors)
        monkeypatch.setattr("bio_automation.pipeline.clean_and_stack", fake_clean_and_stack)
        monkeypatch.setattr("bio_automation.pipeline.ModelCache", RecordingModelCache)
        return folder_ok, folder_zerene_fail, folder_other_fail, fake_clean_and_stack, rejected_raw

    def test_continues_past_failures_and_records_each_folders_outcome(self, tmp_path, monkeypatch):
        zerene_error = ZereneError("Zerene exited with code 1: disk full")
        other_error = OSError("Insufficient disk space: 1.0 GB free < 5.0 GB required")
        folder_ok, folder_zerene_fail, folder_other_fail, fake_clean_and_stack, rejected_raw = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": zerene_error, "obs-c": other_error},
        )

        results = stack_all(tmp_path)

        # Processing continued past both failures: all three folders were
        # actually called, in discovery (sorted) order, with obs-c called
        # *after* obs-b raised -- proof the batch did not abort.
        assert [call["folder"] for call in fake_clean_and_stack.calls] == [
            folder_ok, folder_zerene_fail, folder_other_fail,
        ]

        assert results == [
            {"folder": str(folder_ok), "status": "ok", "error": None,
             "asset": "work/ZS-OutputImage.tif"},
            {"folder": str(folder_zerene_fail), "status": "failed",
             "error": "Zerene exited with code 1: disk full", "asset": None},
            {"folder": str(folder_other_fail), "status": "failed",
             "error": "Insufficient disk space: 1.0 GB free < 5.0 GB required", "asset": None},
        ]

        # Real cull: the rejected raw in the successful folder is gone.
        assert not rejected_raw.exists()

    def test_writes_exact_manifest_json_to_base_dir(self, tmp_path, monkeypatch):
        folder_ok, folder_zerene_fail, folder_other_fail, _, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": ZereneError("boom"), "obs-c": None},
        )

        results = stack_all(tmp_path)

        manifest_path = tmp_path / "stack_batch_manifest.json"
        assert manifest_path.exists()
        assert read_json(manifest_path) == results
        assert read_json(manifest_path) == [
            {"folder": str(folder_ok), "status": "ok", "error": None,
             "asset": "work/ZS-OutputImage.tif"},
            {"folder": str(folder_zerene_fail), "status": "failed", "error": "boom", "asset": None},
            {"folder": str(folder_other_fail), "status": "ok", "error": None,
             "asset": "work/ZS-OutputImage.tif"},
        ]

    def test_builds_exactly_one_shared_model_cache_for_the_whole_batch(self, tmp_path, monkeypatch):
        _, _, _, fake_clean_and_stack, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": ZereneError("boom"), "obs-c": OSError("disk")},
        )

        stack_all(tmp_path)

        assert len(RecordingModelCache.instances_built) == 1
        shared_cache = RecordingModelCache.instances_built[0]
        assert shared_cache.exited is True
        # Every clean_and_stack() call -- including the two that then
        # raised -- got the *same* cache object, not one each.
        assert len(fake_clean_and_stack.calls) == 3
        assert all(call["cache"] is shared_cache for call in fake_clean_and_stack.calls)

    def test_passes_through_non_default_params_to_every_folder(self, tmp_path, monkeypatch):
        folder_ok, folder_b, folder_c, fake_clean_and_stack, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": None, "obs-c": None},
        )

        stack_all(tmp_path, noise=0.3, exposure_ev=-1.0, method="pmax", timeout=99.0,
                  min_free_gb=7.5, check_subjects=False)

        expected_common = {"noise": 0.3, "exposure_ev": -1.0, "method": "pmax", "timeout": 99.0,
                            "min_free_gb": 7.5, "check_subjects": False}
        shared_cache = RecordingModelCache.instances_built[0]
        assert [
            {k: v for k, v in call.items() if k != "cache"} for call in fake_clean_and_stack.calls
        ] == [
            {"folder": folder_ok, **expected_common},
            {"folder": folder_b, **expected_common},
            {"folder": folder_c, **expected_common},
        ]
        assert all(call["cache"] is shared_cache for call in fake_clean_and_stack.calls)
