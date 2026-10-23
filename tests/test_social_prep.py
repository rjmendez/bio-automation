"""social-prep: the incremental gallery index (only new or changed files are read) and the prep run (order, preflight, failure isolation, audit)."""
from __future__ import annotations

import json
import os

import pytest
from test_social import RecordingTitles, cand, jpeg, write_csv

from bio_automation import social, social_hooks, social_prep
from bio_automation.social_rules import Audit

RAWS = {"A": "2025/2025-10/2025-10-02_DSC_1.jpg", "B": "2025/2025-10/2025-10-02_DSC_2.jpg", "C": "2025/2025-10/2025-10-02_DSC_3.jpg"}


class World:
    def __init__(self, tmp_path):
        self.g = tmp_path / "gallery"
        for rel in RAWS.values():
            jpeg(self.g / rel)
        self.csv = write_csv(tmp_path / "c.csv", [cand("100", "Bee", "2025-10-02", "2025/2025-10-02/DSC_1.NEF", "1"),
                                                  cand("200", "Moth", "2025-10-02", "2025/2025-10-02/DSC_2.NEF", "2"),
                                                  cand("300", "Fly", "2025-10-02", "2025/2025-10-02/DSC_3.NEF", "3")])
        self.cache = tmp_path / "state" / "titles.json"
        self.titles = RecordingTitles({"2025-10-02_DSC_1.jpg": "Bee", "2025-10-02_DSC_2.jpg": "Moth", "2025-10-02_DSC_3.jpg": "Fly"})

    def build(self, **kw):
        return social.build_obs_index(self.g, [], [self.csv], titles=self.titles, **kw)


class TestIncrementalIndex:
    def test_cached_run_gives_the_same_index_as_the_uncached_one(self, tmp_path):
        w = World(tmp_path)
        plain = w.build()
        first = w.build(title_cache=w.cache)
        second = w.build(title_cache=w.cache)
        assert plain == first == second
        assert json.dumps(plain, sort_keys=True) == json.dumps(second, sort_keys=True)
        assert plain == {"files": {RAWS["A"]: 100, RAWS["B"]: 200, RAWS["C"]: 300}, "unresolved": {}}

    def test_second_run_reads_nothing_and_the_first_reads_everything(self, tmp_path):
        w = World(tmp_path)
        w.build(title_cache=w.cache)
        w.build(title_cache=w.cache)
        assert w.titles.calls == [["2025-10-02_DSC_1.jpg", "2025-10-02_DSC_2.jpg", "2025-10-02_DSC_3.jpg"]]

    def test_a_changed_file_is_reread_and_only_that_one(self, tmp_path):
        w = World(tmp_path)
        w.build(title_cache=w.cache)
        changed = w.g / RAWS["B"]
        jpeg(changed, size=(80, 60))
        os.utime(changed, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        w.titles.titles["2025-10-02_DSC_2.jpg"] = "Other"
        index = w.build(title_cache=w.cache)
        assert w.titles.calls[1:] == [["2025-10-02_DSC_2.jpg"]]
        assert index["files"] == {RAWS["A"]: 100, RAWS["C"]: 300}
        assert index["unresolved"] == {RAWS["B"]: "every csv row's title differs from the one embedded in the file"}

    def test_a_file_with_the_same_size_but_a_new_mtime_is_reread(self, tmp_path):
        w = World(tmp_path)
        w.build(title_cache=w.cache)
        p = w.g / RAWS["A"]
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        w.build(title_cache=w.cache)
        assert w.titles.calls[1:] == [["2025-10-02_DSC_1.jpg"]]

    def test_a_file_with_a_new_size_but_the_same_mtime_is_reread(self, tmp_path):
        w = World(tmp_path)
        p = w.g / RAWS["A"]
        os.utime(p, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        w.build(title_cache=w.cache)
        jpeg(p, size=(90, 70))
        os.utime(p, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        w.build(title_cache=w.cache)
        assert w.titles.calls[1:] == [["2025-10-02_DSC_1.jpg"]]

    def test_an_unchanged_file_is_not_reread_when_the_stub_would_answer_differently(self, tmp_path):
        w = World(tmp_path)
        w.build(title_cache=w.cache)
        w.titles.titles["2025-10-02_DSC_2.jpg"] = "Other"          # the file did not change, so the cached title still stands
        index = w.build(title_cache=w.cache)
        assert index["files"][RAWS["B"]] == 200
        assert len(w.titles.calls) == 1

    def test_a_deleted_file_is_dropped_from_the_cache(self, tmp_path):
        w = World(tmp_path)
        w.build(title_cache=w.cache)
        (w.g / RAWS["C"]).unlink()
        w.build(title_cache=w.cache)
        assert sorted(json.loads(w.cache.read_text(encoding="utf-8"))["titles"]) == [RAWS["A"], RAWS["B"]]

    def test_a_damaged_cache_costs_a_full_read_not_a_failure(self, tmp_path):
        w = World(tmp_path)
        w.cache.parent.mkdir(parents=True)
        w.cache.write_text("{not json", encoding="utf-8")
        index = w.build(title_cache=w.cache)
        assert index["files"] == {RAWS["A"]: 100, RAWS["B"]: 200, RAWS["C"]: 300}
        assert w.titles.calls == [["2025-10-02_DSC_1.jpg", "2025-10-02_DSC_2.jpg", "2025-10-02_DSC_3.jpg"]]

    def test_cache_stores_size_mtime_and_title(self, tmp_path):
        w = World(tmp_path)
        p = w.g / RAWS["A"]
        os.utime(p, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        w.build(title_cache=w.cache)
        entry = json.loads(w.cache.read_text(encoding="utf-8"))["titles"][RAWS["A"]]
        assert entry == {"size": p.stat().st_size, "mtime_ns": 1_700_000_000_000_000_000, "title": "Bee"}


class Steps:
    """Recording step runners; `results` maps a step name to an exit code or an exception to raise."""

    def __init__(self, **results):
        self.results, self.calls = results, []

    def runners(self):
        def make(name):
            def run():
                self.calls.append(name)
                r = self.results.get(name, 0)
                if isinstance(r, BaseException):
                    raise r
                return r
            return run
        return {n: make(n) for n in social_prep.STEPS}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 2.5
        return self.t


def prep(tmp_path, steps, *, c_free=100.0, d_free=100.0):
    audit, said = Audit(tmp_path / "audit.jsonl"), []
    code = social_prep.run_prep(steps.runners(), audit=audit, free=lambda drive: c_free if drive.startswith("C") else d_free, clock=Clock(),
                                say=said.append)
    return code, audit.records(), said


class TestPrepRun:
    def test_steps_run_in_order_and_each_is_audited_with_its_seconds(self, tmp_path):
        steps = Steps()
        code, records, said = prep(tmp_path, steps)
        assert code == 0
        assert steps.calls == ["social-index", "darkroom-sync", "social-draft"]
        assert [(r["event"], r["step"], r["ok"], r["seconds"]) for r in records] == [
            ("prep_step", "preflight", True, 2.5), ("prep_step", "social-index", True, 2.5),
            ("prep_step", "darkroom-sync", True, 2.5), ("prep_step", "social-draft", True, 2.5)]
        assert said == ["bio social-prep: social-index ok in 2.5s", "bio social-prep: darkroom-sync ok in 2.5s", "bio social-prep: social-draft ok in 2.5s"]

    def test_low_c_drive_stops_with_one_reason_and_runs_nothing(self, tmp_path):
        steps = Steps()
        code, records, said = prep(tmp_path, steps, c_free=14.0)
        assert code == 1
        assert steps.calls == []
        assert said == ["bio social-prep: stopped, C: has 14.0 GB free, social-prep needs 15 GB"]
        assert [(r["step"], r["ok"], r["reason"]) for r in records] == [("preflight", False, "C: has 14.0 GB free, social-prep needs 15 GB")]

    def test_low_d_drive_stops_too(self, tmp_path):
        steps = Steps()
        code, _, said = prep(tmp_path, steps, d_free=19.9)
        assert (code, steps.calls, said) == (1, [], ["bio social-prep: stopped, D: has 19.9 GB free, social-prep needs 20 GB"])

    def test_exactly_the_minimum_free_is_enough(self, tmp_path):
        steps = Steps()
        code, _, _ = prep(tmp_path, steps, c_free=15.0, d_free=20.0)
        assert (code, steps.calls) == (0, ["social-index", "darkroom-sync", "social-draft"])

    def test_a_failed_darkroom_sync_does_not_stop_the_draft(self, tmp_path):
        steps = Steps(**{"darkroom-sync": RuntimeError("storefront down")})
        code, records, said = prep(tmp_path, steps)
        assert code == 1
        assert steps.calls == ["social-index", "darkroom-sync", "social-draft"]
        failed = [r for r in records if not r["ok"]]
        assert [(r["step"], r["error"]) for r in failed] == [("darkroom-sync", "RuntimeError: storefront down")]
        assert records[-1]["step"] == "social-draft" and records[-1]["ok"] is True
        assert said[-1] == "bio social-prep: failed steps: darkroom-sync"

    def test_a_non_zero_exit_code_is_a_failure_and_later_steps_still_run(self, tmp_path):
        steps = Steps(**{"social-index": 2})
        code, records, _ = prep(tmp_path, steps)
        assert code == 1
        assert steps.calls == ["social-index", "darkroom-sync", "social-draft"]
        assert (records[1]["step"], records[1]["ok"], records[1]["error"]) == ("social-index", False, "exit code 2")

    def test_system_exit_from_a_step_is_a_failure_not_an_abort(self, tmp_path):
        steps = Steps(**{"social-index": SystemExit("no file matches x.csv")})
        code, records, _ = prep(tmp_path, steps)
        assert code == 1
        assert steps.calls == ["social-index", "darkroom-sync", "social-draft"]
        assert records[1]["error"] == "SystemExit: no file matches x.csv"


class TestPrepCommand:
    def test_extension_registers_the_command_and_parser(self):
        import argparse

        assert social_prep.COMMANDS == {"social-prep": social_prep.cmd_social_prep}
        top = argparse.ArgumentParser()
        social_prep.add_parsers(top.add_subparsers(dest="command"))
        args = top.parse_args(["social-prep", "--index", "x.json"])
        assert (args.command, args.index) == ("social-prep", "x.json")
        assert "social_prep" in social_hooks.EXTENSIONS

    def test_default_steps_pass_the_title_cache_to_the_index(self, tmp_path, monkeypatch):
        from bio_automation import cli

        seen = {}
        monkeypatch.setattr(cli, "cmd_social_index", lambda a: seen.update(index=a) or 0)
        import argparse

        a = argparse.Namespace(index="i.json", gallery_dir="g")
        assert social_prep.default_steps(a)["social-index"]() == 0
        assert (seen["index"].out, seen["index"].gallery_dir, seen["index"].full) == ("i.json", "g", False)
        assert seen["index"].audit == ["D:/bio-cache/photos/adobe_ready_audit.csv"]

    def test_the_cli_command_runs_prep_and_returns_its_code(self, tmp_path, monkeypatch):
        import argparse

        steps = Steps(**{"social-draft": 1})
        monkeypatch.setattr(social_prep, "default_steps", lambda args: steps.runners())
        monkeypatch.setattr(social_prep, "free_gb", lambda drive: 500.0)
        code = social_prep.cmd_social_prep(argparse.Namespace(index="i", gallery_dir="g"))
        assert code == 1
        assert steps.calls == ["social-index", "darkroom-sync", "social-draft"]
        events = [(r["event"], r["step"], r["ok"]) for r in Audit().records()]
        assert events[-1] == ("prep_step", "social-draft", False)
