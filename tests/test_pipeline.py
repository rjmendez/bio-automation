"""Unit tests for bio_automation.pipeline functions touched by Zerene wiring
(hermetic -- zerene.run_batch is monkeypatched, never actually launched)."""
from __future__ import annotations

from pathlib import Path

import pytest

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

    def __call__(self, folder, method="pmax", timeout=1800.0):
        self.calls.append({"folder": Path(folder), "method": method, "timeout": timeout})
        out = Path(folder) / "ZS-OutputImage.tif"
        out.write_bytes(b"stacked")
        return out


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
        return fake_run, leftover_script

    def test_calls_zerene_with_raw_dir_method_and_timeout(self, tmp_path, monkeypatch):
        fake_run, _ = self._prep(tmp_path, monkeypatch)
        stack_focus(tmp_path, method="pmax", timeout=42.0)
        assert fake_run.calls == [
            {"folder": source_raw_dir(tmp_path), "method": "pmax", "timeout": 42.0}
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

    def __call__(self, raw_paths, dest_dir, noise=1.0, exposure_ev=0.0, cache=None):
        self.calls.append({
            "raw_paths": list(raw_paths), "dest_dir": Path(dest_dir),
            "noise": noise, "exposure_ev": exposure_ev,
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
            {"folder": work_dir(tmp_path) / "denoised", "method": "pmax", "timeout": 1800.0}
        ]

    def test_appends_stack_output_asset_from_denoised_output(self, tmp_path, monkeypatch):
        self._prep(tmp_path, monkeypatch)
        asset = clean_and_stack(tmp_path)
        assert asset == Asset(file="work/ZS-OutputImage.tif", role="stack_output",
                               status="ready", selected=True)
        assert (work_dir(tmp_path) / "ZS-OutputImage.tif").exists()
