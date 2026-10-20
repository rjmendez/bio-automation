"""Unit tests for bio_automation.pipeline touched by the stacking-engine
dispatch (hermetic -- both zerene.run_batch and focus_stack.run_focus_stack
are monkeypatched, never actually launched)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from bio_automation import preclean
from bio_automation.focus_stack import FocusStackError
from bio_automation.grouping import SubjectMismatchError
from bio_automation.io import read_json
from bio_automation.models import Asset, Observation
from bio_automation.pipeline import (
    clean_and_stack,
    discover_observation_folders,
    ensure_layout,
    process_all,
    process_observation,
    save_assets,
    save_observation,
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


class RecordingFocusStackRun:
    """Stands in for focus_stack.run_focus_stack: records calls, writes a fake
    output file. No 'method' kwarg -- unlike zerene.run_batch, the
    focus-stack engine has no internal task code to select (see
    pipeline.STACK_ENGINES and stack_focus()'s docstring)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, folder, timeout=1800.0, min_free_gb=5.0):
        self.calls.append({"folder": Path(folder), "timeout": timeout, "min_free_gb": min_free_gb})
        out = Path(folder) / "FS-OutputImage.tiff"
        out.write_bytes(b"stacked")
        return out


class RecordingDecode:
    """Stands in for preclean.decode_raw_frames_to_tiff: records calls,
    writes fake TIFFs (so the engine fakes that then read *their* folder's
    contents, e.g. RecordingFocusStackRun via _discover_input_images in the
    real module, see real output files in dest_dir)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, raw_paths, dest_dir):
        self.calls.append({"raw_paths": list(raw_paths), "dest_dir": Path(dest_dir)})
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for p in raw_paths:
            out = dest_dir / f"{Path(p).stem}.tif"
            out.write_bytes(b"decoded")
            written.append(out)
        return written


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
        fake_zerene = RecordingBatchRun()
        fake_focus_stack = RecordingFocusStackRun()
        fake_decode = RecordingDecode()
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_zerene)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", fake_focus_stack)
        monkeypatch.setattr("bio_automation.pipeline.decode_raw_frames_to_tiff", fake_decode)
        # The subject-consistency gate is a separate concern (see
        # TestSubjectConsistencyGate below); stub it out here so these
        # stacking-engine-wiring tests are unaffected by the gate being on
        # by default.
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))
        return fake_zerene, fake_focus_stack, leftover_script, fake_decode

    def test_calls_focus_stack_with_decoded_dir_and_timeout_by_default(self, tmp_path, monkeypatch):
        """focus-stack is the default engine (2026-10-19) -- see
        pipeline.STACK_ENGINES. The default raw source is decoded to TIFF
        first (2026-10-19 fix -- see decode_raw_frames_to_tiff's docstring
        for why the engine never gets raw NEFs directly), so the engine sees
        work/stack_decoded/, not source/raw/ itself."""
        fake_zerene, fake_focus_stack, _, fake_decode = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, timeout=42.0)
        decoded_dir = work_dir(tmp_path) / "stack_decoded"
        assert fake_focus_stack.calls == [
            {"folder": decoded_dir, "timeout": 42.0, "min_free_gb": 5.0}
        ]
        # _prep only registers DSC_0001.NEF in assets.jsonl, never writes it
        # to disk (the stacking engines are fully mocked here and never read
        # folder contents) -- so raw_dir's real directory listing is empty
        # apart from the leftover ZereneBatch.xml, which is filtered out.
        assert fake_decode.calls == [{"raw_paths": [], "dest_dir": decoded_dir}]
        assert fake_zerene.calls == []

    def test_calls_zerene_with_decoded_dir_and_pmax_task_code_when_method_is_zerene(self, tmp_path, monkeypatch):
        """Positive twin to the default-engine test above: the explicit
        alternate engine must still be reachable, and its *internal* task
        code stays fixed at "pmax" (the only one zerene.py documents) --
        pipeline's own 'method' is now an engine selector, not that task
        code, so it must not leak "zerene" through as a bogus task code.
        Also gets the decoded dir, same as the default engine."""
        fake_zerene, fake_focus_stack, _, fake_decode = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, method="zerene", timeout=42.0)
        decoded_dir = work_dir(tmp_path) / "stack_decoded"
        assert fake_zerene.calls == [
            {"folder": decoded_dir, "method": "pmax", "timeout": 42.0,
             "min_free_gb": 5.0}
        ]
        assert fake_decode.calls == [{"raw_paths": [], "dest_dir": decoded_dir}]
        assert fake_focus_stack.calls == []

    def test_unknown_method_raises_before_touching_either_engine(self, tmp_path, monkeypatch):
        fake_zerene, fake_focus_stack, _, fake_decode = self._prep(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match=r"Unknown stacking method 'dmap'"):
            stack_focus(tmp_path, method="dmap")
        assert fake_zerene.calls == []
        assert fake_focus_stack.calls == []
        assert fake_decode.calls == []

    def test_passes_through_non_default_min_free_gb(self, tmp_path, monkeypatch):
        fake_zerene, fake_focus_stack, _, fake_decode = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, min_free_gb=12.5)
        decoded_dir = work_dir(tmp_path) / "stack_decoded"
        assert fake_focus_stack.calls == [
            {"folder": decoded_dir, "timeout": 1800.0, "min_free_gb": 12.5}
        ]
        assert fake_zerene.calls == []

    def test_decoded_dir_is_removed_after_a_successful_stack(self, tmp_path, monkeypatch):
        """The disposable work/stack_decoded/ intermediate must not survive a
        successful run -- same cleanup contract clean_and_stack() already
        has for work/denoised/ (see TestCleanAndStackCleanup)."""
        self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path)
        assert not (work_dir(tmp_path) / "stack_decoded").exists()

    def test_decoded_dir_is_removed_when_the_engine_fails(self, tmp_path, monkeypatch):
        """Negative twin to the cleanup test above: a failing engine must
        not leave work/stack_decoded/ behind, and the original exception
        must propagate unchanged."""
        self._prep(tmp_path, monkeypatch)

        def failing_run(folder, timeout=1800.0, min_free_gb=5.0):
            raise FocusStackError("focus-stack exited 1 on folder: disk full")

        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", failing_run)

        with pytest.raises(FocusStackError, match=r"^focus-stack exited 1 on folder: disk full$"):
            stack_focus(tmp_path)

        assert not (work_dir(tmp_path) / "stack_decoded").exists()

    def test_explicit_source_dir_skips_the_decode_step(self, tmp_path, monkeypatch):
        """Positive twin proving the decode step is specific to the default
        raw source, not unconditional: an explicit source_dir (clean_and_stack's
        own contract) is trusted as already-decoded, same as before this fix."""
        _, fake_focus_stack, _, fake_decode = self._prep(tmp_path, monkeypatch)
        already_decoded = work_dir(tmp_path) / "denoised"
        already_decoded.mkdir(parents=True)
        (already_decoded / "DSC_0001.tif").write_bytes(b"denoised")

        stack_focus(tmp_path, source_dir=already_decoded)

        assert fake_decode.calls == []
        assert fake_focus_stack.calls == [
            {"folder": already_decoded, "timeout": 1800.0, "min_free_gb": 5.0}
        ]

    def test_moves_output_into_work_dir_not_raw_dir(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path)
        assert (work_dir(tmp_path) / "FS-OutputImage.tiff").exists()
        assert not (source_raw_dir(tmp_path) / "FS-OutputImage.tiff").exists()

    def test_removes_leftover_batch_script_from_raw_dir(self, tmp_path, monkeypatch):
        _, _, leftover_script, _ = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path)
        assert not leftover_script.exists()

    def test_appends_exact_stack_output_asset_and_returns_it(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        returned = stack_focus(tmp_path)
        expected = Asset(file="work/FS-OutputImage.tiff", role="stack_output",
                          status="ready", selected=True)
        assert returned == expected
        assets = scan_assets(tmp_path)
        assert assets == [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            expected,
        ]

    def test_appends_exact_stack_output_asset_for_the_zerene_engine_too(self, tmp_path, monkeypatch):
        """Positive twin to the default-engine test above, same fixture:
        proves the alternate engine's own output also lands as a correctly
        shaped Asset, not just the default path."""
        self._prep(tmp_path, monkeypatch)
        returned = stack_focus(tmp_path, method="zerene")
        expected = Asset(file="work/ZS-OutputImage.tif", role="stack_output",
                          status="ready", selected=True)
        assert returned == expected


class RecordingDenoise:
    """Stands in for preclean.denoise_stack_frames: records calls, writes fake TIFFs."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, raw_paths, dest_dir, noise=1.0, exposure_ev=0.0,
                 subject_boost_ev=0.0, cache=None, min_free_gb=5.0):
        self.calls.append({
            "raw_paths": list(raw_paths), "dest_dir": Path(dest_dir),
            "noise": noise, "exposure_ev": exposure_ev,
            "subject_boost_ev": subject_boost_ev, "min_free_gb": min_free_gb,
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
        fake_zerene = RecordingBatchRun()
        fake_focus_stack = RecordingFocusStackRun()
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_zerene)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", fake_focus_stack)
        # See TestStackFocus._prep: the subject-consistency gate is a separate
        # concern (TestSubjectConsistencyGate below), stubbed out here too.
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", RecordingSubjectCheck())
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))
        return raw_a, raw_b, fake_denoise, fake_zerene, fake_focus_stack

    def test_denoises_all_raw_dir_files_into_work_denoised_with_exact_params(self, tmp_path, monkeypatch):
        raw_a, raw_b, fake_denoise, _, _ = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path, noise=0.7, exposure_ev=-0.5)

        assert len(fake_denoise.calls) == 1
        call = fake_denoise.calls[0]
        assert sorted(call["raw_paths"]) == sorted([raw_a, raw_b])
        assert call["dest_dir"] == work_dir(tmp_path) / "denoised"
        assert call["noise"] == 0.7
        assert call["exposure_ev"] == -0.5

    def test_focus_stack_runs_against_denoised_dir_not_raw_dir_by_default(self, tmp_path, monkeypatch):
        _, _, _, fake_zerene, fake_focus_stack = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path)

        assert fake_focus_stack.calls == [
            {"folder": work_dir(tmp_path) / "denoised", "timeout": 1800.0, "min_free_gb": 5.0}
        ]
        assert fake_zerene.calls == []

    def test_zerene_runs_against_denoised_dir_not_raw_dir_when_method_is_zerene(self, tmp_path, monkeypatch):
        """Positive twin: the explicit alternate engine must receive the
        same denoised dir, not the raw dir, while the default engine's fake
        stays untouched."""
        _, _, _, fake_zerene, fake_focus_stack = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path, method="zerene")

        assert fake_zerene.calls == [
            {"folder": work_dir(tmp_path) / "denoised", "method": "pmax", "timeout": 1800.0,
             "min_free_gb": 5.0}
        ]
        assert fake_focus_stack.calls == []

    def test_passes_non_default_min_free_gb_to_both_denoise_and_the_engine(self, tmp_path, monkeypatch):
        _, _, fake_denoise, fake_zerene, fake_focus_stack = self._prep(tmp_path, monkeypatch)
        clean_and_stack(tmp_path, min_free_gb=9.0)

        assert fake_denoise.calls[0]["min_free_gb"] == 9.0
        assert fake_focus_stack.calls == [
            {"folder": work_dir(tmp_path) / "denoised", "timeout": 1800.0, "min_free_gb": 9.0}
        ]
        assert fake_zerene.calls == []

    def test_appends_stack_output_asset_from_denoised_output(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        asset = clean_and_stack(tmp_path)
        assert asset == Asset(file="work/FS-OutputImage.tiff", role="stack_output",
                               status="ready", selected=True)
        assert (work_dir(tmp_path) / "FS-OutputImage.tiff").exists()


class TestSubjectConsistencyGate:
    """Covers the hard safety gate stack_focus()/clean_and_stack() run before
    handing off to a stacking engine (see bio_automation.grouping.check_subject_consistency).
    check_subject_consistency and load_session are both monkeypatched to
    controlled stubs -- never a real ONNX session or real images -- so these
    tests isolate the *wiring* (did the gate run, in the right order, with
    the right inputs, and did it actually block the engine) from the gate's
    own embedding logic, which tests/test_grouping.py covers. Both engines
    are stubbed too, so a test can assert exactly which one (if either) ran."""

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
        fake_zerene = RecordingBatchRun()
        fake_focus_stack = RecordingFocusStackRun()
        fake_decode = RecordingDecode()
        fake_check = RecordingSubjectCheck(outliers=outliers)
        fake_session = MagicMock(name="daclip-session")
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_zerene)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", fake_focus_stack)
        monkeypatch.setattr("bio_automation.pipeline.decode_raw_frames_to_tiff", fake_decode)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: fake_session)
        return raw_a, raw_b, fake_zerene, fake_focus_stack, fake_check, fake_session, fake_decode

    def test_proceeds_to_the_stacking_engine_when_no_outliers_found(self, tmp_path, monkeypatch):
        """Positive twin to test_raises_before_touching_either_engine_when_outliers_found
        below: the empty-outliers case must let the real work (the default
        focus-stack engine) run."""
        raw_a, raw_b, fake_zerene, fake_focus_stack, fake_check, fake_session, fake_decode = self._prep(
            tmp_path, monkeypatch, outliers=[]
        )

        stack_focus(tmp_path)

        assert len(fake_check.calls) == 1
        assert sorted(fake_check.calls[0]["raw_paths"]) == sorted([raw_a, raw_b])
        assert fake_check.calls[0]["session"] is fake_session
        assert len(fake_focus_stack.calls) == 1
        assert len(fake_decode.calls) == 1
        assert fake_zerene.calls == []

    def test_raises_before_touching_either_engine_when_outliers_found(self, tmp_path, monkeypatch):
        outliers = [(Path("source/raw/DSC_0001.NEF"), 0.2)]
        _, _, fake_zerene, fake_focus_stack, fake_check, _, fake_decode = self._prep(
            tmp_path, monkeypatch, outliers=outliers
        )

        with pytest.raises(SubjectMismatchError, match=r"1 frame\(s\) fall below"):
            stack_focus(tmp_path)

        assert fake_zerene.calls == []
        assert fake_focus_stack.calls == []
        assert fake_decode.calls == []

    def test_check_subjects_false_skips_gate_entirely(self, tmp_path, monkeypatch):
        """Negative twin to the two tests above: even outliers that would
        normally raise must never be consulted when check_subjects=False."""
        _, _, fake_zerene, fake_focus_stack, fake_check, _, fake_decode = self._prep(
            tmp_path, monkeypatch, outliers=[(Path("whatever.NEF"), 0.0)]
        )

        stack_focus(tmp_path, check_subjects=False)

        assert fake_check.calls == []
        assert len(fake_focus_stack.calls) == 1
        assert len(fake_decode.calls) == 1
        assert fake_zerene.calls == []

    def test_missing_model_raises_clear_error_before_either_engine(self, tmp_path, monkeypatch):
        """Design decision: a missing DA-CLIP model fails loudly (wrapped
        FileNotFoundError naming the check) rather than silently skipping
        the safety gate and proceeding to a stacking engine anyway."""
        _, _, fake_zerene, fake_focus_stack, fake_check, _, fake_decode = self._prep(
            tmp_path, monkeypatch, outliers=[]
        )
        model_path = r"C:\TopazModels\daclip-v2-fp32-224x224-1x-ox\daclip-v2-fp32-224x224-1x.onnx"

        def _raise_missing():
            raise FileNotFoundError(f"DA-CLIP model not found: {model_path}")

        monkeypatch.setattr("bio_automation.pipeline.load_session", _raise_missing)

        with pytest.raises(FileNotFoundError, match="Subject-consistency check unavailable"):
            stack_focus(tmp_path)

        assert fake_zerene.calls == []
        assert fake_focus_stack.calls == []
        assert fake_check.calls == []
        assert fake_decode.calls == []

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
        fake_zerene = RecordingBatchRun()
        fake_focus_stack = RecordingFocusStackRun()
        fake_check = RecordingSubjectCheck(outliers=[(Path("DSC_0001.NEF"), 0.1)])
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_zerene)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", fake_focus_stack)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

        with pytest.raises(SubjectMismatchError, match=r"1 frame\(s\) fall below"):
            clean_and_stack(tmp_path)

        assert fake_zerene.calls == []
        assert fake_focus_stack.calls == []
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
        fake_zerene = RecordingBatchRun()
        fake_focus_stack = RecordingFocusStackRun()
        fake_check = RecordingSubjectCheck(outliers=[(Path("DSC_0001.NEF"), 0.1)])
        monkeypatch.setattr("bio_automation.pipeline.denoise_stack_frames", fake_denoise)
        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", fake_zerene)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", fake_focus_stack)
        monkeypatch.setattr("bio_automation.pipeline.check_subject_consistency", fake_check)
        monkeypatch.setattr("bio_automation.pipeline.load_session", lambda: MagicMock(name="daclip-session"))

        clean_and_stack(tmp_path, check_subjects=False)

        assert fake_check.calls == []
        assert len(fake_focus_stack.calls) == 1
        assert fake_zerene.calls == []


class TestCleanAndStackCleanup:
    """Covers clean_and_stack()'s own disposable-intermediate cleanup:
    work/denoised/ (the per-frame denoised TIFFs that fed the stacking
    engine) must be gone once stacking is done, whether it succeeded or
    failed, for either engine. These tests stub clean_and_stack()'s own
    direct dependencies (denoise_stack_frames, the stacking engines, the
    subject-consistency gate) so clean_and_stack()'s own cleanup logic is
    what's exercised, not theirs."""

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
        per-frame intermediates) is removed. Exercises the default
        (focus-stack) engine."""
        self._prep(tmp_path, monkeypatch)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", RecordingFocusStackRun())

        clean_and_stack(tmp_path)

        assert not (work_dir(tmp_path) / "denoised").exists()
        assert (work_dir(tmp_path) / "FS-OutputImage.tiff").exists()

    def test_failure_from_the_default_engine_deletes_denoised_dir_and_reraises_exact_exception(self, tmp_path, monkeypatch):
        """Negative twin to the test above, same fixture: the default
        (focus-stack) engine failing (inside stack_focus) must still clean
        up work/denoised/, and the original FocusStackError -- exact type
        and message -- must propagate unchanged, not be swallowed or
        replaced."""
        self._prep(tmp_path, monkeypatch)

        def failing_run(folder, timeout=1800.0, min_free_gb=5.0):
            raise FocusStackError("focus-stack exited 1 on folder: disk full")

        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", failing_run)

        with pytest.raises(FocusStackError, match=r"^focus-stack exited 1 on folder: disk full$") as exc_info:
            clean_and_stack(tmp_path)

        assert str(exc_info.value) == "focus-stack exited 1 on folder: disk full"
        assert not (work_dir(tmp_path) / "denoised").exists()

    def test_failure_from_zerene_deletes_denoised_dir_and_reraises_exact_exception(self, tmp_path, monkeypatch):
        """Same cleanup contract, exercised through the explicit alternate
        engine instead of the default -- proves the cleanup logic is not
        specific to either engine's own exception type."""
        self._prep(tmp_path, monkeypatch)

        def failing_run(folder, method="pmax", timeout=1800.0, min_free_gb=5.0):
            raise ZereneError("Zerene exited with code 1: disk full")

        monkeypatch.setattr("bio_automation.pipeline.zerene_run_batch", failing_run)

        with pytest.raises(ZereneError, match=r"^Zerene exited with code 1: disk full$") as exc_info:
            clean_and_stack(tmp_path, method="zerene")

        assert str(exc_info.value) == "Zerene exited with code 1: disk full"
        assert not (work_dir(tmp_path) / "denoised").exists()

    def test_failure_from_denoise_deletes_partial_denoised_dir_and_reraises_exact_exception(self, tmp_path, monkeypatch):
        """Third twin, same fixture: a failure inside denoise_stack_frames
        itself (before stack_focus ever runs, so work/denoised/ may hold
        only some of the frames) must also leave no partial output behind,
        and must propagate the exact original exception -- stack_focus/the
        engine never even run."""
        self._prep(tmp_path, monkeypatch)
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", RecordingFocusStackRun())

        def failing_denoise(raw_paths, dest_dir, noise=1.0, exposure_ev=0.0,
                             subject_boost_ev=0.0, cache=None, min_free_gb=5.0):
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
        monkeypatch.setattr("bio_automation.pipeline.focus_stack_run_batch", RecordingFocusStackRun())
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

    def __call__(self, folder, noise=1.0, exposure_ev=0.0, subject_boost_ev=0.0,
                 method="focus-stack", timeout=1800.0,
                 min_free_gb=5.0, check_subjects=True, cache=None):
        folder = Path(folder)
        self.calls.append({
            "folder": folder, "noise": noise, "exposure_ev": exposure_ev,
            "subject_boost_ev": subject_boost_ev, "method": method,
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

    def test_default_method_forwarded_to_every_folder_is_focus_stack(self, tmp_path, monkeypatch):
        """focus-stack is stack_all()'s default engine too (see
        pipeline.STACK_ENGINES) -- positive twin to the explicit "zerene"
        forwarding test below, same fixture shape."""
        _, _, _, fake_clean_and_stack, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": None, "obs-c": None},
        )

        stack_all(tmp_path)

        assert {call["method"] for call in fake_clean_and_stack.calls} == {"focus-stack"}

    def test_passes_through_non_default_params_to_every_folder(self, tmp_path, monkeypatch):
        folder_ok, folder_b, folder_c, fake_clean_and_stack, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": None, "obs-c": None},
        )

        stack_all(tmp_path, noise=0.3, exposure_ev=-1.0, subject_boost_ev=0.8,
                  method="zerene", timeout=99.0,
                  min_free_gb=7.5, check_subjects=False)

        expected_common = {"noise": 0.3, "exposure_ev": -1.0, "subject_boost_ev": 0.8,
                            "method": "zerene", "timeout": 99.0,
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


class _RecordingStage:
    """Generic recording stub for one of process_observation()'s own direct
    dependencies: appends its name to a shared *call_order* list (so tests
    can assert cross-stage ordering) and records its own call's arguments
    separately (so tests can assert exact per-stage arguments). Never
    raises on its own -- a test that needs a stage to fail passes its own
    plain function instead (see TestProcessObservationFailures)."""

    def __init__(self, name: str, call_order: list[str], return_value):
        self.name = name
        self.call_order = call_order
        self.return_value = return_value
        self.calls: list[dict] = []

    def __call__(self, folder, **kwargs):
        self.call_order.append(self.name)
        self.calls.append({"folder": Path(folder), **kwargs})
        return self.return_value


class TestProcessObservation:
    """process_observation() orchestrates the mechanical post-curation
    chain. Its own direct dependencies (stack_focus/clean_and_stack,
    write_sidecars, render_review, export_inat, export_stock,
    build_and_save_keywords) are stubbed as RECORDING stubs -- never
    raising ones -- so these tests exercise the real orchestrator and
    assert exactly which stages ran, in what order, and with what
    arguments. load_observation/save_observation are the real functions
    (plain JSON I/O already covered by their own tests) so observation_id
    linkage is read from a real round-tripped observation.json, not faked."""

    def _prep(self, tmp_path, monkeypatch, *, observation_id=None):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        save_observation(tmp_path, Observation(observation_id=observation_id, obsdate="2026-01-05"))

        call_order: list[str] = []
        stack_asset = Asset(file="work/FS-OutputImage.tiff", role="stack_output",
                             status="ready", selected=True)
        stages = {
            "stack_focus": _RecordingStage("stack_focus", call_order, stack_asset),
            "clean_and_stack": _RecordingStage("clean_and_stack", call_order, stack_asset),
            "write_sidecars": _RecordingStage("write_sidecars", call_order,
                                               [tmp_path / "metadata" / "DSC_0001.xmp"]),
            "render_review": _RecordingStage("render_review", call_order,
                                              [tmp_path / "review" / "DSC_0001.jpg"]),
            "export_inat": _RecordingStage("export_inat", call_order,
                                            [tmp_path / "export" / "inat" / "DSC_0001.jpg"]),
            "export_stock": _RecordingStage("export_stock", call_order,
                                             [tmp_path / "export" / "stock" / "DSC_0001.jpg"]),
            "build_and_save_keywords": _RecordingStage("build_and_save_keywords", call_order,
                                                         {"keywords": ["spider"]}),
        }
        for name, stub in stages.items():
            monkeypatch.setattr(f"bio_automation.pipeline.{name}", stub)
        return call_order, stages, stack_asset

    def test_default_chain_skips_export_inat_when_not_inat_linked(self, tmp_path, monkeypatch):
        """observation_id=None (the Observation default, never set until
        `bio fetch-inat` runs) must skip export_inat cleanly -- not call it
        with zero files, not raise -- while every other stage still runs
        in the documented order with exactly the folder argument."""
        call_order, stages, stack_asset = self._prep(tmp_path, monkeypatch, observation_id=None)

        result = process_observation(tmp_path)

        assert call_order == [
            "stack_focus", "write_sidecars", "render_review", "export_stock",
            "build_and_save_keywords",
        ]
        assert stages["export_inat"].calls == []
        assert stages["clean_and_stack"].calls == []
        assert stages["stack_focus"].calls == [{
            "folder": tmp_path, "method": "focus-stack", "timeout": 1800.0,
            "min_free_gb": 5.0, "check_subjects": True,
        }]
        for name in ("write_sidecars", "render_review", "export_stock",
                     "build_and_save_keywords"):
            assert stages[name].calls == [{"folder": tmp_path}]

        assert result == {
            "folder": str(tmp_path),
            "stack_asset": "work/FS-OutputImage.tiff",
            "sidecars": [str(tmp_path / "metadata" / "DSC_0001.xmp")],
            "review": [str(tmp_path / "review" / "DSC_0001.jpg")],
            "inat": None,
            "stock": [str(tmp_path / "export" / "stock" / "DSC_0001.jpg")],
            "keywords": {"keywords": ["spider"]},
        }

    def test_runs_export_inat_when_observation_is_inat_linked(self, tmp_path, monkeypatch):
        """Positive twin to the test above, same fixture shape: a real
        observation_id (as merge_inat_fields()/`bio fetch-inat` would have
        set) must make export_inat run, in order after render_review and
        before export_stock."""
        call_order, stages, _ = self._prep(tmp_path, monkeypatch, observation_id=987654)

        result = process_observation(tmp_path)

        assert call_order == [
            "stack_focus", "write_sidecars", "render_review", "export_inat",
            "export_stock", "build_and_save_keywords",
        ]
        assert stages["export_inat"].calls == [{"folder": tmp_path}]
        assert result["inat"] == [str(tmp_path / "export" / "inat" / "DSC_0001.jpg")]

    def test_denoise_true_dispatches_to_clean_and_stack_with_exact_params(self, tmp_path, monkeypatch):
        """denoise=True must route stacking through clean_and_stack() (with
        its own noise/exposure_ev/subject_boost_ev/cache forwarded exactly)
        instead of stack_focus() -- positive twin to the default-False case
        covered above, where clean_and_stack().calls == []."""
        call_order, stages, _ = self._prep(tmp_path, monkeypatch, observation_id=None)
        sentinel_cache = object()

        process_observation(tmp_path, denoise=True, noise=0.4, exposure_ev=-0.6,
                             subject_boost_ev=0.2, cache=sentinel_cache)

        assert stages["clean_and_stack"].calls == [{
            "folder": tmp_path, "noise": 0.4, "exposure_ev": -0.6,
            "subject_boost_ev": 0.2, "method": "focus-stack", "timeout": 1800.0,
            "min_free_gb": 5.0, "check_subjects": True, "cache": sentinel_cache,
        }]
        assert stages["stack_focus"].calls == []
        assert call_order[0] == "clean_and_stack"

    def test_forwards_non_default_method_timeout_and_check_subjects_to_stack_focus(
        self, tmp_path, monkeypatch
    ):
        _, stages, _ = self._prep(tmp_path, monkeypatch, observation_id=None)

        process_observation(tmp_path, method="zerene", timeout=42.0, min_free_gb=9.0,
                             check_subjects=False)

        assert stages["stack_focus"].calls == [{
            "folder": tmp_path, "method": "zerene", "timeout": 42.0,
            "min_free_gb": 9.0, "check_subjects": False,
        }]


class TestProcessObservationFailures:
    """Covers process_observation()'s error-propagation contract: any stage
    raising must stop the chain immediately (no later stage runs) and the
    original exception must propagate unchanged, never be swallowed or
    reported as a partial success."""

    def _prep(self, tmp_path, monkeypatch, *, observation_id, failing_stage, error):
        ensure_layout(tmp_path)
        save_assets(tmp_path, [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        save_observation(tmp_path, Observation(observation_id=observation_id, obsdate="2026-01-05"))

        call_order: list[str] = []
        stack_asset = Asset(file="work/FS-OutputImage.tiff", role="stack_output",
                             status="ready", selected=True)
        return_values = {
            "stack_focus": stack_asset,
            "write_sidecars": [tmp_path / "metadata" / "DSC_0001.xmp"],
            "render_review": [tmp_path / "review" / "DSC_0001.jpg"],
            "export_inat": [tmp_path / "export" / "inat" / "DSC_0001.jpg"],
            "export_stock": [tmp_path / "export" / "stock" / "DSC_0001.jpg"],
            "build_and_save_keywords": {"keywords": ["spider"]},
        }
        stages: dict[str, _RecordingStage] = {}
        for name, value in return_values.items():
            if name == failing_stage:
                def _raise(folder, _name=name, _error=error, **kwargs):
                    call_order.append(_name)
                    raise _error
                monkeypatch.setattr(f"bio_automation.pipeline.{name}", _raise)
            else:
                stub = _RecordingStage(name, call_order, value)
                stages[name] = stub
                monkeypatch.setattr(f"bio_automation.pipeline.{name}", stub)
        return call_order, stages

    def test_stacking_failure_propagates_before_any_later_stage_runs(self, tmp_path, monkeypatch):
        error = FocusStackError("focus-stack exited 1 on folder: disk full")
        call_order, stages = self._prep(
            tmp_path, monkeypatch, observation_id=None,
            failing_stage="stack_focus", error=error,
        )

        with pytest.raises(FocusStackError, match=r"^focus-stack exited 1 on folder: disk full$") as exc_info:
            process_observation(tmp_path)

        assert exc_info.value is error
        assert call_order == ["stack_focus"]
        for name in ("write_sidecars", "render_review", "export_inat", "export_stock",
                     "build_and_save_keywords"):
            assert stages[name].calls == []

    def test_export_inat_failure_propagates_and_stops_before_export_stock(self, tmp_path, monkeypatch):
        """A stage partway through an iNat-linked chain fails: the exact
        exception propagates, and neither export_stock nor
        build_and_save_keywords (the two stages after it) ever run."""
        error = RuntimeError("inat export boom")
        call_order, stages = self._prep(
            tmp_path, monkeypatch, observation_id=12345,
            failing_stage="export_inat", error=error,
        )

        with pytest.raises(RuntimeError, match=r"^inat export boom$") as exc_info:
            process_observation(tmp_path)

        assert exc_info.value is error
        assert call_order == ["stack_focus", "write_sidecars", "render_review", "export_inat"]
        assert stages["export_stock"].calls == []
        assert stages["build_and_save_keywords"].calls == []


class RecordingProcessObservation:
    """Stands in for pipeline.process_observation: records every call (in
    order) and raises or succeeds per folder according to *behaviors*
    (keyed by folder name), same shape as RecordingCleanAndStack above."""

    def __init__(self, behaviors: dict[str, Exception | None]):
        self.calls: list[dict] = []
        self.behaviors = behaviors

    def __call__(self, folder, method="focus-stack", timeout=1800.0, min_free_gb=5.0,
                 check_subjects=True, denoise=False, noise=1.0, exposure_ev=0.0,
                 subject_boost_ev=0.0, cache=None):
        folder = Path(folder)
        self.calls.append({
            "folder": folder, "method": method, "timeout": timeout,
            "min_free_gb": min_free_gb, "check_subjects": check_subjects,
            "denoise": denoise, "noise": noise, "exposure_ev": exposure_ev,
            "subject_boost_ev": subject_boost_ev, "cache": cache,
        })
        behavior = self.behaviors[folder.name]
        if behavior is not None:
            raise behavior
        return {"folder": str(folder), "stack_asset": "work/FS-OutputImage.tiff"}


class TestProcessAll:
    """process_all() orchestrates process_observation() over every
    observation folder under a base_dir, reusing discover_observation_folders()
    (the real function, not stubbed -- same choice TestStackAll makes for
    stack_all()) and surviving per-folder failures. process_observation()
    itself is stubbed (its own behavior is covered by TestProcessObservation
    above); these tests isolate the batch orchestration."""

    def _prep(self, tmp_path, monkeypatch, behaviors):
        RecordingModelCache.instances_built = []
        folder_ok = tmp_path / "obs-a"
        folder_fail_a = tmp_path / "obs-b"
        folder_fail_b = tmp_path / "obs-c"
        for folder in (folder_ok, folder_fail_a, folder_fail_b):
            ensure_layout(folder)
            save_assets(folder, [
                Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            ])
        fake_process_observation = RecordingProcessObservation(behaviors)
        monkeypatch.setattr("bio_automation.pipeline.process_observation", fake_process_observation)
        monkeypatch.setattr("bio_automation.pipeline.ModelCache", RecordingModelCache)
        return folder_ok, folder_fail_a, folder_fail_b, fake_process_observation

    def test_continues_past_failures_and_records_each_folders_outcome(self, tmp_path, monkeypatch):
        error_a = FocusStackError("focus-stack exited 1 on folder: disk full")
        error_b = OSError("Insufficient disk space: 1.0 GB free < 5.0 GB required")
        folder_ok, folder_fail_a, folder_fail_b, fake_process_observation = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": error_a, "obs-c": error_b},
        )

        results = process_all(tmp_path)

        assert [call["folder"] for call in fake_process_observation.calls] == [
            folder_ok, folder_fail_a, folder_fail_b,
        ]
        assert results == [
            {"folder": str(folder_ok), "status": "ok", "error": None,
             "result": {"folder": str(folder_ok), "stack_asset": "work/FS-OutputImage.tiff"}},
            {"folder": str(folder_fail_a), "status": "failed",
             "error": "focus-stack exited 1 on folder: disk full", "result": None},
            {"folder": str(folder_fail_b), "status": "failed",
             "error": "Insufficient disk space: 1.0 GB free < 5.0 GB required", "result": None},
        ]

    def test_writes_exact_manifest_json_to_a_distinct_filename_from_stack_all(self, tmp_path, monkeypatch):
        folder_ok, folder_fail_a, folder_fail_b, _ = self._prep(
            tmp_path, monkeypatch,
            behaviors={"obs-a": None, "obs-b": ZereneError("boom"), "obs-c": None},
        )

        results = process_all(tmp_path)

        manifest_path = tmp_path / "process_batch_manifest.json"
        assert manifest_path.exists()
        assert read_json(manifest_path) == results
        # Distinct from stack_all()'s own "stack_batch_manifest.json" so the
        # two batch commands' manifests never collide on disk.
        assert not (tmp_path / "stack_batch_manifest.json").exists()

    def test_denoise_false_never_builds_a_model_cache_and_passes_cache_none(self, tmp_path, monkeypatch):
        _, _, _, fake_process_observation = self._prep(
            tmp_path, monkeypatch, behaviors={"obs-a": None, "obs-b": None, "obs-c": None},
        )

        process_all(tmp_path)

        assert RecordingModelCache.instances_built == []
        assert all(call["cache"] is None for call in fake_process_observation.calls)

    def test_denoise_true_builds_exactly_one_shared_model_cache_for_the_whole_batch(
        self, tmp_path, monkeypatch
    ):
        """Positive twin to the test above, same fixture: denoise=True must
        build exactly one ModelCache and share it across every folder, not
        build one per folder (or none at all)."""
        _, _, _, fake_process_observation = self._prep(
            tmp_path, monkeypatch, behaviors={"obs-a": None, "obs-b": None, "obs-c": None},
        )

        process_all(tmp_path, denoise=True)

        assert len(RecordingModelCache.instances_built) == 1
        shared_cache = RecordingModelCache.instances_built[0]
        assert shared_cache.exited is True
        assert len(fake_process_observation.calls) == 3
        assert all(call["cache"] is shared_cache for call in fake_process_observation.calls)
        assert all(call["denoise"] is True for call in fake_process_observation.calls)
