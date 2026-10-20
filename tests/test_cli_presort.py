"""Unit tests for the `bio presort` CLI subcommand.

Hermetic: suggest_culling is mocked as cli.py's own dependency (same pattern
test_pipeline_exports.py already uses for its cmd-adjacent pipeline
functions), so what's actually exercised is cli.py's own argument parsing,
dispatch, and printed summary -- not a reimplementation of suggest_culling.
"""
from __future__ import annotations

import argparse

import pytest
from pathlib import Path

from bio_automation import cli
from bio_automation.cli import build_parser, cmd_presort, cmd_reject, main


class TestPresortArgParsing:
    def test_parses_folder_positional(self):
        args = build_parser().parse_args(["presort", r"F:\Bio\2026\obs-x"])
        assert args.command == "presort"
        assert args.folder == r"F:\Bio\2026\obs-x"

    def test_dispatches_to_cmd_presort_from_main(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, "suggest_culling", lambda folder: calls.append(Path(folder)) or {
            "folder": str(folder), "assets": [],
            "summary": {"total": 0, "keep_candidates": 0, "cull_candidates": 0},
        })
        monkeypatch.setattr("sys.argv", ["bio", "presort", str(tmp_path)])

        exit_code = main()

        assert exit_code == 0
        assert calls == [tmp_path]


class TestCmdPresort:
    def test_calls_suggest_culling_with_the_given_folder(self, tmp_path, monkeypatch):
        calls = []

        def fake_suggest_culling(folder):
            calls.append(Path(folder))
            return {
                "folder": str(folder), "assets": [{"file": "a"}, {"file": "b"}],
                "summary": {"total": 2, "keep_candidates": 1, "cull_candidates": 1},
            }

        monkeypatch.setattr(cli, "suggest_culling", fake_suggest_culling)

        result = cmd_presort(argparse.Namespace(folder=str(tmp_path)))

        assert result == 0
        assert calls == [tmp_path]

    def test_prints_exact_summary_and_written_path(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "suggest_culling", lambda folder: {
            "folder": str(folder), "assets": [{"file": "a"}, {"file": "b"}],
            "summary": {"total": 2, "keep_candidates": 1, "cull_candidates": 1},
        })

        cmd_presort(argparse.Namespace(folder=str(tmp_path)))

        out = capsys.readouterr().out
        assert out == (
            "2 ingested asset(s) scored: 1 keep candidate(s), 1 cull candidate(s)\n"
            f"{tmp_path / 'presort_suggestions.json'}\n"
        )

    def test_zero_cull_candidates_prints_exact_zero_not_a_truthy_check(self, tmp_path, monkeypatch, capsys):
        """Negative twin to the mixed-counts test above: all-keep must print
        the literal 0, proving the counts are read from the result, not just
        'truthy therefore printed'."""
        monkeypatch.setattr(cli, "suggest_culling", lambda folder: {
            "folder": str(folder), "assets": [{"file": "a"}, {"file": "b"}, {"file": "c"}],
            "summary": {"total": 3, "keep_candidates": 3, "cull_candidates": 0},
        })

        cmd_presort(argparse.Namespace(folder=str(tmp_path)))

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "3 ingested asset(s) scored: 3 keep candidate(s), 0 cull candidate(s)"


class TestPresortNeverMutatesAssetStatus:
    """Rule 4, and the single most important test given the owner's literal
    'do not cull yourself' instruction, exercised at the actual CLI entry
    point: recording stubs on the exact functions cmd_reject/cmd_keep/cmd_cull
    use from this same module, proven empty after `bio presort` runs."""

    def test_cmd_presort_never_calls_set_asset_status_or_delete_rejected_raws(self, tmp_path, monkeypatch):
        set_status_calls: list[tuple] = []
        delete_calls: list[tuple] = []
        monkeypatch.setattr(cli, "set_asset_status", lambda *a, **k: set_status_calls.append((a, k)))
        monkeypatch.setattr(cli, "delete_rejected_raws", lambda *a, **k: delete_calls.append(a))
        # Non-empty assets/summary -- this must not change whether cmd_presort
        # calls either mutating function, so this fixture (not the vacuous
        # empty-assets case) is what actually proves it.
        monkeypatch.setattr(cli, "suggest_culling", lambda folder: {
            "folder": str(folder), "assets": [{"file": "a"}, {"file": "b"}],
            "summary": {"total": 2, "keep_candidates": 1, "cull_candidates": 1},
        })

        cmd_presort(argparse.Namespace(folder=str(tmp_path)))

        assert set_status_calls == []
        assert delete_calls == []

    def test_positive_twin_cmd_reject_does_call_set_asset_status(self, tmp_path, monkeypatch):
        """Proves the patch target above is the real call site (rule 3): the
        same monkeypatch target, exercised by the command that is actually
        supposed to call it, really does record a call."""
        set_status_calls: list[tuple] = []
        monkeypatch.setattr(cli, "set_asset_status", lambda *a, **k: set_status_calls.append((a, k)))

        cmd_reject(argparse.Namespace(folder=str(tmp_path), files=["DSC_0001.NEF"]))

        assert set_status_calls == [((Path(str(tmp_path)), ["DSC_0001.NEF"], "rejected", False), {})]


class TestServeDefaults:
    def test_serve_binds_loopback_by_default(self):
        assert build_parser().parse_args(["serve"]).host == "127.0.0.1"

    def test_serve_launches_the_current_interpreter(self, monkeypatch):
        import subprocess
        import sys
        calls = []
        monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)

        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_serve(build_parser().parse_args(["serve", "--port", "9"]))

        assert exc_info.value.code == 0
        assert calls[0][0] == sys.executable
        assert calls[0][2:] == ["--host", "127.0.0.1", "--port", "9", "--device", "AUTO"]


class TestIngestDoesNotStorePathAsPlace:
    def test_ingest_leaves_place_guess_unset(self, tmp_path):
        from bio_automation.paths import make_observation_path
        from bio_automation.pipeline import load_observation
        src = tmp_path / "in" / "A.NEF"
        src.parent.mkdir()
        src.write_bytes(b"raw")
        args = build_parser().parse_args(["ingest", str(src), "--base-dir", str(tmp_path / "Bio"),
                                          "--obsdate", "2026-07-04", "--slug", "spider"])

        assert cli.cmd_ingest(args) == 0

        folder = make_observation_path(tmp_path / "Bio", "2026-07-04", "spider")
        assert load_observation(folder).place_guess is None
