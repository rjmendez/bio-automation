"""Unit tests for bio_automation.pipeline functions touched by Zerene wiring
(hermetic -- zerene.run_batch is monkeypatched, never actually launched)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bio_automation.grouping import SubjectMismatchError
from bio_automation.models import Asset
from bio_automation.pipeline import clean_and_stack, ensure_layout, save_assets, scan_assets, stack_focus
from bio_automation.paths import source_raw_dir, work_dir


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
