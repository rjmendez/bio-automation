"""Ledger integrity (every corruption type with a clean twin, the transition table, the pre_send block, the torn-tail repair) and the backups
(content, manifest, retention, the daily hook, restore dry run / apply / hash mismatch)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_social import NOW, RecordingSend, drafted, events, rules_with

from bio_automation import cli, settings, social, social_hooks, social_state
from bio_automation.social import Ledger, SocialError
from bio_automation.social_rules import Audit
from bio_automation.social_state import IntegrityError, check_integrity


def rec(event, draft_id="d1", n=0, **fields):
    return {"at": (NOW + dt.timedelta(minutes=n)).isoformat(timespec="seconds"), "event": event, "draft_id": draft_id, **fields}


def write_lines(path: Path, records, *, tail: str = "\n") -> Path:
    """The ledger as the given records (dicts become JSON lines, strings go in as they are), ending with `tail`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r, sort_keys=True) for r in records]
    path.write_bytes(("\n".join(lines) + tail).encode("utf-8"))
    return path


def lifecycle(draft_id="d1", obs=1, n=0):
    return [rec("drafted", draft_id, n, obs_id=obs, month="2026-10"), rec("approved", draft_id, n + 1, obs_id=obs, hash="h"),
            rec("posting", draft_id, n + 2, obs_id=obs), rec("posted", draft_id, n + 3, obs_id=obs, uri="at://x/y/z")]


def kinds(report):
    return [(p.line, p.kind) for p in report.problems]


class TestCleanLedger:
    def test_a_full_life_with_a_redraft_and_a_note_is_clean(self, tmp_path):
        recs = [rec("drafted", "a", 0, obs_id=1), rec("withdrawn", "a", 1, reason="small"), rec("drafted", "a", 2, obs_id=1),
                rec("approved", "a", 3), rec("posting", "a", 4), rec("error", "a", 5, error="login failed"),
                rec("approved", "a", 6, restored=True), rec("posting", "a", 7), rec("posted", "a", 8, uri="at://u"),
                rec("drafted", "b", 9, obs_id=2), rec("rejected", "b", 10)]
        report = check_integrity(write_lines(tmp_path / "l.jsonl", recs))
        assert (report.ok, report.lines, report.problems) == (True, 11, [])

    def test_a_missing_ledger_is_clean_and_empty(self, tmp_path):
        report = check_integrity(tmp_path / "none.jsonl")
        assert (report.ok, report.lines) == (True, 0)

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "l.jsonl"
        path.write_text(json.dumps(rec("drafted")) + "\n\n" + json.dumps(rec("approved", n=1)) + "\n", encoding="utf-8")
        report = check_integrity(path)
        assert (report.ok, report.lines) == (True, 2)

    def test_the_ledger_class_writes_what_the_checker_accepts(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2)
        social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=rules_with(), audit=Audit(tmp_path / "a.jsonl"))
        assert check_integrity(ledger.path).lines == 5
        assert check_integrity(ledger.path).problems == []


class TestCorruption:
    def test_invalid_json_in_the_middle_is_bad_json_with_its_line(self, tmp_path):
        good = rec("drafted")
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [good, '{"at": "x", "event": ', rec("approved", n=1)]))
        assert kinds(report) == [(2, "bad_json")]

    def test_a_line_that_is_json_but_not_an_object(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted"), "[1, 2]"]))
        assert kinds(report) == [(2, "not_object")]

    @pytest.mark.parametrize("field", ["at", "event", "draft_id"])
    def test_a_missing_required_field(self, tmp_path, field):
        broken = rec("drafted")
        del broken[field]
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [broken]))
        assert kinds(report) == [(1, "missing_field")]
        assert report.problems[0].message == f"missing or empty {field}"

    def test_an_empty_draft_id_counts_as_missing(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted", "")]))
        assert kinds(report) == [(1, "missing_field")]

    def test_a_timestamp_that_is_not_iso(self, tmp_path):
        bad = rec("drafted")
        bad["at"] = "yesterday"
        assert kinds(check_integrity(write_lines(tmp_path / "l.jsonl", [bad]))) == [(1, "bad_time")]

    def test_an_unknown_event(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted"), rec("publishd", n=1)]))
        assert kinds(report) == [(2, "unknown_event")]
        assert "publishd" in report.problems[0].message

    def test_bytes_that_are_not_utf8(self, tmp_path):
        path = tmp_path / "l.jsonl"
        path.write_bytes(b'{"at": "\xff"}\n')
        assert kinds(check_integrity(path)) == [(0, "bad_encoding")]

    def test_a_draft_that_starts_with_something_other_than_drafted(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("approved")]))
        assert kinds(report) == [(1, "bad_transition")]
        assert report.problems[0].message == "d1 starts with 'approved', not drafted"

    def test_a_note_before_the_first_state_is_fine(self, tmp_path):
        assert check_integrity(write_lines(tmp_path / "l.jsonl", [rec("error", "d1", 0, error="x"), rec("drafted", "d1", 1)])).ok


class TestTornTail:
    def test_a_last_line_cut_short_is_torn_tail_not_bad_json(self, tmp_path):
        path = write_lines(tmp_path / "l.jsonl", [rec("drafted")], tail='\n{"at": "2026-10-21T09:01:00-04:00", "event": "appr')
        report = check_integrity(path)
        assert kinds(report) == [(2, "torn_tail")]
        assert report.torn_tail

    def test_a_complete_last_line_with_its_newline_is_not_torn(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted")]))
        assert (report.torn_tail, report.ok) == (False, True)

    def test_a_bad_last_line_that_ends_with_a_newline_is_bad_json_not_torn(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted"), '{"at": "x'], tail="\n"))
        assert kinds(report) == [(2, "bad_json")]

    def test_a_valid_last_record_without_newline_is_flagged_so_the_next_append_does_not_join_it(self, tmp_path):
        report = check_integrity(write_lines(tmp_path / "l.jsonl", [rec("drafted"), rec("approved", n=1)], tail=""))
        assert kinds(report) == [(2, "missing_newline")]


class TestTransitions:
    PATHS = {"drafted": ["drafted"], "approved": ["drafted", "approved"], "rejected": ["drafted", "rejected"],
             "posting": ["drafted", "approved", "posting"], "posted": ["drafted", "approved", "posting", "posted"],
             "failed": ["drafted", "approved", "posting", "failed"], "withdrawn": ["drafted", "withdrawn"]}
    ALLOWED = {("drafted", "approved"), ("drafted", "rejected"), ("drafted", "withdrawn"), ("drafted", "drafted"),
               ("approved", "approved"), ("approved", "posting"), ("approved", "rejected"), ("approved", "withdrawn"),
               ("posting", "posted"), ("posting", "failed"),
               ("failed", "approved"), ("failed", "rejected"), ("failed", "withdrawn"), ("failed", "drafted"),
               ("withdrawn", "drafted")}

    def ledger_after(self, tmp_path, before, after):
        recs = [rec(ev, "d1", i, obs_id=1) for i, ev in enumerate(self.PATHS[before] + [after])]
        return check_integrity(write_lines(tmp_path / f"{before}-{after}.jsonl", recs))

    @pytest.mark.parametrize("before", list(PATHS))
    @pytest.mark.parametrize("after", ["drafted", "approved", "rejected", "posting", "posted", "failed", "withdrawn"])
    def test_every_pair_is_allowed_or_refused_as_the_table_says(self, tmp_path, before, after):
        report = self.ledger_after(tmp_path, before, after)
        if (before, after) in self.ALLOWED:
            assert report.problems == []
        else:
            assert kinds(report) == [(len(self.PATHS[before]) + 1, "bad_transition")]
            assert report.problems[0].message == f"d1: {before} -> {after} is not allowed"

    def test_an_approval_given_back_after_a_send_that_sent_nothing_is_allowed_only_when_marked_restored(self, tmp_path):
        base = [rec("drafted", n=0), rec("approved", n=1), rec("posting", n=2)]
        assert check_integrity(write_lines(tmp_path / "ok.jsonl", base + [rec("approved", n=3, restored=True)])).ok
        report = check_integrity(write_lines(tmp_path / "bad.jsonl", base + [rec("approved", n=3)]))
        assert kinds(report) == [(4, "bad_transition")]

    def test_a_note_between_two_states_does_not_change_the_state(self, tmp_path):
        recs = [rec("drafted", n=0), rec("error", n=1, error="x"), rec("posting", n=2)]
        assert kinds(check_integrity(write_lines(tmp_path / "l.jsonl", recs))) == [(3, "bad_transition")]

    def test_drafts_are_tracked_separately(self, tmp_path):
        recs = [rec("drafted", "a", 0, obs_id=1), rec("drafted", "b", 1, obs_id=2), rec("approved", "a", 2), rec("posting", "a", 3), rec("posted", "a", 4)]
        assert check_integrity(write_lines(tmp_path / "l.jsonl", recs)).ok


class TestDuplicatePosts:
    def test_one_observation_in_two_posted_drafts(self, tmp_path):
        recs = lifecycle("single", obs=7, n=0) + lifecycle("other", obs=7, n=10)
        report = check_integrity(write_lines(tmp_path / "l.jsonl", recs))
        assert kinds(report) == [(0, "duplicate_post")]
        assert report.problems[0].message == "observation 7 is in 2 posted drafts: single, other"

    def test_two_posted_drafts_with_different_observations_are_clean(self, tmp_path):
        assert check_integrity(write_lines(tmp_path / "l.jsonl", lifecycle("a", obs=7) + lifecycle("b", obs=8, n=10))).ok

    def test_a_theme_group_overlapping_a_single_is_found_through_obs_ids(self, tmp_path):
        group = [rec("drafted", "group", 10, obs_ids=[5, 6, 7], obs_id=5), rec("approved", "group", 11), rec("posting", "group", 12),
                 rec("posted", "group", 13, uri="at://g")]
        report = check_integrity(write_lines(tmp_path / "l.jsonl", lifecycle("single", obs=7) + group))
        assert [p.message for p in report.problems] == ["observation 7 is in 2 posted drafts: single, group"]

    def test_an_observation_in_a_posted_draft_and_a_merely_drafted_one_is_clean(self, tmp_path):
        recs = lifecycle("a", obs=7) + [rec("drafted", "b", 10, obs_id=7)]
        assert check_integrity(write_lines(tmp_path / "l.jsonl", recs)).ok

    def test_a_redraft_under_the_same_id_with_new_observations_frees_the_old_ones(self, tmp_path):
        recs = (lifecycle("a", obs=7) + [rec("drafted", "x", 10, obs_id=9), rec("withdrawn", "x", 11), rec("drafted", "x", 12, obs_id=8)]
                + [rec("approved", "x", 13), rec("posting", "x", 14), rec("posted", "x", 15, uri="at://x")])
        assert check_integrity(write_lines(tmp_path / "l.jsonl", recs)).ok


class TestReportText:
    def test_summary_names_the_first_three_and_counts_the_rest(self, tmp_path):
        recs = [rec("drafted", f"d{i}", i, obs_id=i) if i == 0 else rec("approved", f"d{i}", i) for i in range(6)]
        report = check_integrity(write_lines(tmp_path / "l.jsonl", recs))
        assert len(report.problems) == 5
        assert report.summary() == ("bad_transition at line 2: d1 starts with 'approved', not drafted; "
                                    "bad_transition at line 3: d2 starts with 'approved', not drafted; "
                                    "bad_transition at line 4: d3 starts with 'approved', not drafted (and 2 more)")


def ledger_in(tmp_path, records, **kw):
    return Ledger(write_lines(tmp_path / "gate.jsonl", records, **kw))


class TestPreSendGate:
    def ctx(self, ledger, **kw):
        return {"draft_id": "d1", "ledger": ledger, "manual": False, **kw}

    def test_a_clean_ledger_passes(self, tmp_path):
        assert social_state.ledger_gate(self.ctx(ledger_in(tmp_path, [rec("drafted")]))) is None
        assert logged() == []

    def test_a_corrupt_ledger_blocks_and_audits_the_first_problems(self, tmp_path):
        ledger = ledger_in(tmp_path, [rec("drafted"), "garbage", "more garbage", rec("approved", n=1)])
        verdict = social_state.ledger_gate(self.ctx(ledger, manual=True))
        assert verdict.action == "block"
        assert verdict.reason == (f"the ledger {ledger.path} is corrupt: bad_json at line 2: not valid JSON (Expecting value); "
                                  "bad_json at line 3: not valid JSON (Expecting value); run `bio social-integrity`")
        (entry,) = logged()
        assert (entry["event"], entry["count"], entry["draft_id"], entry["manual"]) == ("ledger_corrupt", 2, "d1", True)
        assert entry["problems"] == ["bad_json at line 2: not valid JSON (Expecting value)", "bad_json at line 3: not valid JSON (Expecting value)"]

    def test_registered_at_priority_2_for_pre_send_and_90_for_maintenance_under_the_rule(self):
        social_state.register_hooks()
        assert [(h.name, h.rule, h.priority) for h in social_hooks.hooks("pre_send")] == [("ledger_integrity", "ledger_integrity", 2)]
        assert [(h.name, h.rule, h.priority) for h in social_hooks.hooks("maintenance")] == [("daily_backup", "ledger_integrity", 90)]

    def test_a_scheduled_post_is_blocked_and_nothing_is_sent_while_the_ledger_is_corrupt(self, tmp_path):
        social_state.register_hooks()
        ledger, drafts = drafted(tmp_path, 2)
        with open(ledger.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec("rubbish", "zz", 5)) + "\n")
        send = RecordingSend()
        result = social.post_next(ledger, drafts, now=NOW, live=True, send=send, rules=rules_with(), audit=Audit())
        assert (result["status"], len(send.calls)) == ("blocked", 0)
        assert result["reason"].startswith(f"the ledger {ledger.path} is corrupt: unknown_event at line 3: event 'rubbish'")
        assert events(ledger, "2026-10-obs1") == ["drafted"]

    def test_a_manual_post_is_blocked_too(self, tmp_path):
        social_state.register_hooks()
        ledger, drafts = drafted(tmp_path, 2)
        with open(ledger.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec("rubbish", "zz", 5)) + "\n")
        send = RecordingSend()
        result = social.post_next(ledger, drafts, now=NOW, live=True, send=send, rules=rules_with(), audit=Audit(), draft_id="2026-10-obs2", manual=True)
        assert (result["status"], len(send.calls)) == ("blocked", 0)

    def test_a_clean_ledger_posts_normally(self, tmp_path):
        social_state.register_hooks()
        ledger, drafts = drafted(tmp_path, 2)
        send = RecordingSend()
        result = social.post_next(ledger, drafts, now=NOW, live=True, send=send, rules=rules_with(), audit=Audit())
        assert (result["status"], result["draft_id"], len(send.calls)) == ("posted", "2026-10-obs1", 1)

    def test_with_the_rule_off_a_corrupt_ledger_does_not_block(self, tmp_path):
        social_state.register_hooks()
        ledger = ledger_in(tmp_path, [rec("drafted"), "garbage"])
        ctx = {"draft_id": "d1", "draft": {}, "draft_dir": tmp_path, "ledger": ledger, "now": NOW, "live": True, "manual": False, "fetch": None}
        off = social_hooks.run("pre_send", {**ctx, "rules": rules_with(ledger_integrity=False)}, rules=rules_with(ledger_integrity=False), audit=Audit())
        assert off == []
        on = social_hooks.run("pre_send", {**ctx, "rules": rules_with()}, rules=rules_with(), audit=Audit())
        assert [v.action for _, v in on] == ["block"]


class TestRepairTail:
    TORN = '{"at": "2026-10-21T09:01:00-04:00", "event": "appr'

    def torn_ledger(self, tmp_path):
        return write_lines(settings.SOCIAL_LEDGER, [rec("drafted", obs_id=1), rec("approved", n=1)], tail="\n" + self.TORN)

    def test_the_torn_line_moves_aside_and_every_other_byte_stays(self, tmp_path):
        path = self.torn_ledger(tmp_path)
        before = path.read_bytes()
        keep = before[: before.rfind(b"\n") + 1]
        torn = social_state.repair_tail(path, now=NOW)
        assert torn == path.with_name("ledger.jsonl.torn-20261021-090000")
        assert torn.read_text(encoding="utf-8") == self.TORN
        assert path.read_bytes() == keep
        assert check_integrity(path).ok

    def test_a_backup_of_the_torn_ledger_is_made_first(self, tmp_path):
        path = self.torn_ledger(tmp_path)
        before = path.read_bytes()
        social_state.repair_tail(path, now=NOW)
        (folder,) = social_state.backups()
        assert (folder / "ledger.jsonl").read_bytes() == before

    def test_a_missing_newline_gets_its_newline_and_nothing_is_moved(self, tmp_path):
        path = write_lines(settings.SOCIAL_LEDGER, [rec("drafted")], tail="")
        assert social_state.repair_tail(path, now=NOW) is None
        assert path.read_bytes() == (json.dumps(rec("drafted"), sort_keys=True) + "\n").encode()

    def test_a_clean_ledger_is_left_alone_and_no_backup_is_made(self, tmp_path):
        path = write_lines(settings.SOCIAL_LEDGER, [rec("drafted")])
        before = path.read_bytes()
        assert social_state.repair_tail(path, now=NOW) is None
        assert (path.read_bytes(), social_state.backups()) == (before, [])

    def test_other_corruption_is_refused_and_the_file_is_untouched(self, tmp_path):
        path = write_lines(settings.SOCIAL_LEDGER, [rec("drafted"), "garbage", rec("approved", n=1)], tail="\n" + self.TORN)
        before = path.read_bytes()
        with pytest.raises(IntegrityError, match="problems other than a torn last line"):
            social_state.repair_tail(path, now=NOW)
        assert path.read_bytes() == before

    def test_the_command_reports_repairs_and_exits_zero(self, tmp_path, capsys, monkeypatch):
        self.torn_ledger(tmp_path)
        monkeypatch.setattr(sys, "argv", ["bio", "social-integrity"])
        assert cli.main() == 1
        out = capsys.readouterr().out
        assert out.splitlines()[:2] == [f"ledger {settings.SOCIAL_LEDGER}: 3 records, 1 problems", "  torn_tail at line 3: the last line stops mid-record (Unterminated string starting at); a write was cut short"]
        monkeypatch.setattr(sys, "argv", ["bio", "social-integrity", "--repair-tail"])
        assert cli.main() == 0
        out = capsys.readouterr().out
        assert out.splitlines()[-1] == "after repair: 0 problems"
        assert check_integrity(settings.SOCIAL_LEDGER).ok


# --------------------------------------------------------------------------------------------------- backups

T0 = dt.datetime(2026, 10, 21, 8, 0, 5, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
CONTENT = {"ledger": b'{"at": "2026-10-21T08:00:00-04:00", "draft_id": "d1", "event": "drafted"}\n', "audit": b'{"event": "draft_run"}\n',
           "rules": b"[rules.daily_cap]\nenabled = true\n", "themes": b"[themes]\n", "prints": b"obs_id,url\n1,https://p/1\n",
           "manual": b"obs_id,url\n2,https://p/2\n", "tax": b'{"1": "Araneae"}'}


def seed_files():
    """Write one file for each of the seven backed-up settings and return {backup name: path}."""
    spec = {"ledger.jsonl": (settings.SOCIAL_LEDGER, "ledger"), "audit.jsonl": (settings.SOCIAL_AUDIT_LOG, "audit"),
            "rules.toml": (settings.SOCIAL_RULES_FILE, "rules"), "themes.toml": (settings.SOCIAL_THEMES_FILE, "themes"),
            "prints.csv": (settings.SOCIAL_PRINTS_FILE, "prints"), "prints_manual.csv": (settings.SOCIAL_PRINTS_MANUAL_FILE, "manual"),
            "obs_taxonomy.json": (settings.OBS_TAXONOMY_FILE, "tax")}
    out = {}
    for name, (path, key) in spec.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(CONTENT[key])
        out[name] = Path(path)
    return out


def logged():
    """The audit records written during the test (the seeded draft_run line of seed_files() is not one of them)."""
    return [r for r in Audit.records(Audit()) if r["event"] != "draft_run"]


class TestBackup:
    def test_every_existing_file_is_copied_with_its_hash_and_size_in_the_manifest(self):
        live = seed_files()
        before = {name: path.read_bytes() for name, path in live.items()}              # the backup's own audit line grows audit.jsonl afterwards
        folder = social_state.make_backup(now=T0)
        assert folder == settings.SOCIAL_BACKUP_DIR / "20261021-080005"
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["created"] == "2026-10-21T08:00:05-04:00"
        assert manifest["ledger_ok"] is True
        assert sorted(manifest["files"]) == sorted(live)
        for name, path in live.items():
            data = before[name]
            assert (folder / name).read_bytes() == data
            assert manifest["files"][name] == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": str(path)}

    def test_files_that_do_not_exist_are_left_out(self):
        live = seed_files()
        live["themes.toml"].unlink()
        live["prints.csv"].unlink()
        folder = social_state.make_backup(now=T0)
        assert sorted(json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))["files"]) == [
            "audit.jsonl", "ledger.jsonl", "obs_taxonomy.json", "prints_manual.csv", "rules.toml"]

    def test_the_audit_log_gets_backup_made_with_the_files_and_the_bytes(self):
        seed_files()
        social_state.make_backup(now=T0)
        made = [r for r in logged() if r["event"] == "backup_made"]
        assert len(made) == 1
        assert (made[0]["name"], made[0]["bytes"], made[0]["removed"], made[0]["ledger_ok"]) == ("20261021-080005", sum(len(v) for v in CONTENT.values()), [], True)
        assert made[0]["files"] == ["audit.jsonl", "ledger.jsonl", "obs_taxonomy.json", "prints.csv", "prints_manual.csv", "rules.toml", "themes.toml"]

    def test_no_temp_folder_is_left_behind(self):
        seed_files()
        social_state.make_backup(now=T0)
        assert [p.name for p in settings.SOCIAL_BACKUP_DIR.iterdir()] == ["20261021-080005"]

    def test_two_backups_in_the_same_second_get_distinct_folders(self):
        seed_files()
        first, second = social_state.make_backup(now=T0), social_state.make_backup(now=T0)
        assert (first.name, second.name) == ("20261021-080005", "20261021-080005-2")

    def test_nothing_to_back_up_fails_with_a_message_and_backup_failed(self):
        with pytest.raises(SocialError, match="none of the files to back up exist"):
            social_state.make_backup(now=T0)
        failed = [r for r in logged() if r["event"] == "backup_failed"]
        assert [r["error"] for r in failed] == ["none of the files to back up exist"]
        assert [p.name for p in settings.SOCIAL_BACKUP_DIR.iterdir()] == []

    def test_an_unwritable_backup_folder_is_backup_failed_and_a_clear_error(self, tmp_path):
        seed_files()
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where the folder should be", encoding="utf-8")
        with pytest.raises(SocialError, match="backup into .*blocker failed"):
            social_state.make_backup(blocker, now=T0)
        assert [r["event"] for r in logged()] == ["backup_failed"]

    def test_a_corrupt_ledger_is_backed_up_faithfully_and_marked(self):
        live = seed_files()
        live["ledger.jsonl"].write_bytes(b"garbage\n")
        folder = social_state.make_backup(now=T0)
        assert (folder / "ledger.jsonl").read_bytes() == b"garbage\n"
        assert json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))["ledger_ok"] is False


class TestRetention:
    def make(self, n, start=T0):
        return [social_state.make_backup(now=start + dt.timedelta(days=i), keep=None).name for i in range(n)]

    def test_only_the_newest_n_stay(self):
        seed_files()
        names = self.make(5)
        social_state.make_backup(now=T0 + dt.timedelta(days=5), keep=3)
        assert [f.name for f in social_state.backups()] == ["20261024-080005", "20261025-080005", "20261026-080005"]
        assert names[0] not in [f.name for f in social_state.backups()]
        made = [r for r in logged() if r["event"] == "backup_made"][-1]
        assert made["removed"] == ["20261021-080005", "20261022-080005", "20261023-080005"]

    def test_fewer_than_n_are_all_kept(self):
        seed_files()
        self.make(2)
        social_state.make_backup(now=T0 + dt.timedelta(days=2), keep=3)
        assert len(social_state.backups()) == 3

    def test_the_default_keeps_fourteen(self):
        seed_files()
        self.make(14)
        social_state.make_backup(now=T0 + dt.timedelta(days=14))
        assert [f.name for f in social_state.backups()][0] == "20261022-080005"
        assert len(social_state.backups()) == 14

    def test_the_newest_backup_with_a_clean_ledger_survives_a_run_of_corrupt_ones(self):
        live = seed_files()
        good = social_state.make_backup(now=T0, keep=None)
        live["ledger.jsonl"].write_bytes(b"garbage\n")
        for i in range(1, 4):
            social_state.make_backup(now=T0 + dt.timedelta(days=i), keep=2)
        assert [f.name for f in social_state.backups()] == [good.name, "20261023-080005", "20261024-080005"]

    def test_keep_below_one_is_refused_and_nothing_is_made(self):
        seed_files()
        with pytest.raises(SocialError, match="--keep must be at least 1, got 0"):
            social_state.make_backup(now=T0, keep=0)
        assert social_state.backups() == []

    def test_folders_that_are_not_backups_are_never_deleted(self):
        seed_files()
        stranger = settings.SOCIAL_BACKUP_DIR / "notes"
        stranger.mkdir(parents=True)
        self.make(3)
        social_state.make_backup(now=T0 + dt.timedelta(days=3), keep=1)
        assert (stranger.is_dir(), len(social_state.backups())) == (True, 1)


class TestDailyBackupHook:
    def ctx(self, day):
        return {"drafts_dir": Path("."), "ledger": Ledger(settings.SOCIAL_LEDGER), "rules": rules_with(), "today": day}

    def test_the_first_run_of_a_day_makes_a_backup_and_the_second_does_not(self):
        seed_files()
        day = dt.date.today()
        social_state.daily_backup(self.ctx(day))
        assert len(social_state.backups()) == 1
        social_state.daily_backup(self.ctx(day))
        assert len(social_state.backups()) == 1
        assert [r["event"] for r in logged()] == ["backup_made"]

    def test_a_backup_from_yesterday_does_not_count(self):
        seed_files()
        social_state.make_backup(now=dt.datetime.now(settings.LOCAL_TZ) - dt.timedelta(days=1))
        social_state.daily_backup(self.ctx(dt.date.today()))
        assert len(social_state.backups()) == 2

    def test_the_hook_backs_up_the_ledger_of_the_context(self, tmp_path):
        seed_files()
        other = write_lines(tmp_path / "elsewhere.jsonl", [rec("drafted", "zz")])
        ctx = {**self.ctx(dt.date.today()), "ledger": Ledger(other)}
        social_state.daily_backup(ctx)
        (folder,) = social_state.backups()
        assert (folder / "ledger.jsonl").read_bytes() == other.read_bytes()

    def test_a_failing_backup_is_audited_and_does_not_raise(self):
        social_state.daily_backup(self.ctx(dt.date.today()))              # no files exist at all
        assert [r["event"] for r in logged()] == ["backup_failed"]

    def test_it_runs_through_the_maintenance_stage_only_while_the_rule_is_on(self):
        seed_files()
        social_state.register_hooks()
        ctx = self.ctx(dt.date.today())
        social_hooks.run("maintenance", ctx, rules=rules_with(ledger_integrity=False), audit=Audit())
        assert social_state.backups() == []
        social_hooks.run("maintenance", ctx, rules=rules_with(), audit=Audit())
        assert len(social_state.backups()) == 1


class TestRestore:
    def setup_backup(self):
        live = seed_files()
        folder = social_state.make_backup(now=T0)
        live["ledger.jsonl"].write_bytes(CONTENT["ledger"] + b'{"newer": 1}\n')
        live["rules.toml"].unlink()
        return live, folder

    def test_dry_run_reports_each_file_and_changes_nothing(self):
        live, folder = self.setup_backup()
        before = {n: (p.read_bytes() if p.exists() else None) for n, p in live.items()}
        plan = social_state.restore(folder.name)
        assert dict(plan) == {                                  # audit.jsonl differs: the backup itself logged backup_made after copying it
            "ledger.jsonl": "would change", "audit.jsonl": "would change", "rules.toml": "would create", "themes.toml": "same",
                              "prints.csv": "same", "prints_manual.csv": "same", "obs_taxonomy.json": "same"}
        assert {n: (p.read_bytes() if p.exists() else None) for n, p in live.items()} == before
        assert len(social_state.backups()) == 1

    def test_apply_makes_a_pre_restore_backup_then_brings_the_files_back(self):
        live, folder = self.setup_backup()
        newer = live["ledger.jsonl"].read_bytes()
        result = social_state.restore(folder.name, apply=True, now=T0 + dt.timedelta(hours=1))
        assert dict(result)["ledger.jsonl"] == "restored" and dict(result)["rules.toml"] == "restored"
        assert live["ledger.jsonl"].read_bytes() == CONTENT["ledger"]
        assert live["rules.toml"].read_bytes() == CONTENT["rules"]
        pre = social_state.backups()[-1]
        assert pre.name == "20261021-090005"
        assert (pre / "ledger.jsonl").read_bytes() == newer
        (restored,) = [r for r in logged() if r["event"] == "restored"]
        assert (restored["name"], restored["pre_restore_backup"], len(restored["files"])) == (folder.name, pre.name, 7)
        assert list(live["ledger.jsonl"].parent.glob("*.restore-tmp")) == []

    def test_a_hash_mismatch_refuses_the_whole_restore_and_touches_nothing(self):
        live, folder = self.setup_backup()
        (folder / "audit.jsonl").write_bytes(b"tampered\n")
        current = live["ledger.jsonl"].read_bytes()
        with pytest.raises(IntegrityError, match=r"audit\.jsonl does not match its recorded sha256 .*nothing was restored"):
            social_state.restore(folder.name, apply=True, now=T0 + dt.timedelta(hours=1))
        assert live["ledger.jsonl"].read_bytes() == current
        assert len(social_state.backups()) == 1                  # no pre-restore backup was made either

    def test_a_hash_mismatch_also_refuses_a_dry_run(self):
        _, folder = self.setup_backup()
        (folder / "ledger.jsonl").write_bytes(b"tampered\n")
        with pytest.raises(IntegrityError, match=r"ledger\.jsonl does not match"):
            social_state.restore(folder.name)

    def test_a_name_that_is_a_path_or_unknown_is_refused(self):
        self.setup_backup()
        with pytest.raises(SocialError, match="is not a backup name"):
            social_state.restore("../x")
        with pytest.raises(SocialError, match="no backup '19990101-000000'"):
            social_state.restore("19990101-000000")

    def test_a_backup_without_a_manifest_is_refused(self):
        _, folder = self.setup_backup()
        (folder / "MANIFEST.json").unlink()
        with pytest.raises(IntegrityError, match="has no readable MANIFEST.json"):
            social_state.restore(folder.name)

    def test_a_file_the_system_will_not_replace_gives_a_clear_message(self, monkeypatch):
        live, folder = self.setup_backup()
        real = social_state.os.replace

        def locked(src, dst):
            if Path(dst).name == "ledger.jsonl":
                raise PermissionError(13, "The process cannot access the file")
            return real(src, dst)

        monkeypatch.setattr(social_state.os, "replace", locked)
        with pytest.raises(SocialError, match=r"cannot replace .*ledger\.jsonl: another program has it open or it is read-only"):
            social_state.restore(folder.name, apply=True, now=T0 + dt.timedelta(hours=1))
        assert list(live["ledger.jsonl"].parent.glob("*.restore-tmp")) == []

    def test_restore_waits_for_no_one_a_running_poster_refuses_it(self):
        live, folder = self.setup_backup()
        with social.ledger_lock(live["ledger.jsonl"]):
            with pytest.raises(SocialError, match="another social-post run is posting"):
                social_state.restore(folder.name, apply=True, now=T0 + dt.timedelta(hours=1))


class TestCommands:
    def run(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["bio", *argv])
        code = cli.main()
        return code, capsys.readouterr()

    def test_backup_prints_the_folder_and_keeps_n(self, monkeypatch, capsys):
        seed_files()
        code, out = self.run(monkeypatch, capsys, "social-backup", "--keep", "1")
        assert code == 0
        assert out.out.splitlines()[0] == f"backup {social_state.backups()[0]}"
        self.run(monkeypatch, capsys, "social-backup", "--keep", "1")
        assert len(social_state.backups()) == 1

    def test_restore_list_then_dry_run_then_apply_needs_yes(self, monkeypatch, capsys):
        live = seed_files()
        folder = social_state.make_backup(now=T0)
        live["audit.jsonl"].write_bytes(b"changed\n")
        code, out = self.run(monkeypatch, capsys, "social-restore", "--list")
        assert code == 0 and out.out.startswith(f"{folder.name}  7 files  ledger ok  ")
        code, out = self.run(monkeypatch, capsys, "social-restore", "--from", folder.name)
        assert (code, "  audit.jsonl: would change" in out.out, out.out.splitlines()[-1]) == (
            0, True, "dry run: nothing changed (add --apply --yes to restore)")
        assert live["audit.jsonl"].read_bytes() == b"changed\n"
        code, out = self.run(monkeypatch, capsys, "social-restore", "--from", folder.name, "--apply")
        assert (code, out.err) == (1, "bio social-restore: --apply replaces the live files; add --yes to confirm\n")
        assert live["audit.jsonl"].read_bytes() == b"changed\n"
        code, out = self.run(monkeypatch, capsys, "social-restore", "--from", folder.name, "--apply", "--yes")
        assert code == 0
        restored_audit = live["audit.jsonl"].read_bytes()
        assert restored_audit.startswith(CONTENT["audit"])                       # the log is back; only the line saying so was added after it
        assert [json.loads(l)["event"] for l in restored_audit[len(CONTENT["audit"]):].splitlines()] == ["restored"]

    def test_a_restore_error_is_one_line_not_a_traceback(self, monkeypatch, capsys):
        seed_files()
        code, out = self.run(monkeypatch, capsys, "social-restore", "--from", "19990101-000000")
        assert (code, out.err.startswith("bio social-restore: no backup '19990101-000000'")) == (1, True)


class TestUnparsableLedgerStopsThePost:
    def test_a_ledger_line_that_is_not_json_stops_the_run_with_the_line_number_and_sends_nothing(self, tmp_path):
        social_state.register_hooks()
        ledger, drafts = drafted(tmp_path, 2)
        with open(ledger.path, "a", encoding="utf-8") as f:
            f.write("{torn\n")
        send = RecordingSend()
        with pytest.raises(SocialError, match=r"the ledger .*l\.jsonl is corrupt at line 3 \(Expecting property name enclosed in double quotes\)"):
            social.post_next(ledger, drafts, now=NOW, live=True, send=send, rules=rules_with(), audit=Audit())
        assert send.calls == []
