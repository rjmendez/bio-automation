"""Owner controls: the pause switch and the circuit breaker."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from test_social import NOW, URI, RecordingSend, drafted
from test_social import rules_with as _rules_with

from bio_automation import cli, settings, social, social_controls, social_hooks
from bio_automation.social import SocialError
from bio_automation.social_rules import Audit


# The other modules' pre_send/post_send hooks are registered process-wide once any test loads them; these tests are about the controls only,
# so the rules of the other owners are off here.
OTHER_HOOKS_OFF = dict(live_recheck=False, ledger_integrity=False, image_floor=False, verify_posts=False, alerts=False)


def rules_with(**states):
    return _rules_with(**{**OTHER_HOOKS_OFF, **states})


def audit_path() -> Path:
    return Path(settings.SOCIAL_AUDIT_LOG)


def write_audit(*events: tuple) -> None:
    """Append audit records: (event, run, fields)."""
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for i, (event, run, fields) in enumerate(events):
            rec = {"at": f"2026-10-2{i % 10}T08:00:00-04:00", "run": run, "event": event, **fields}
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def fail(run, draft, outcome="nothing sent"):
    return ("post_failed", run, {"draft_id": draft, "outcome": outcome, "error": "boom"})


def events(name=None):
    recs = Audit(audit_path()).records()
    return [r for r in recs if name is None or r["event"] == name]


def run_cli(monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["bio", *argv])
    return cli.main()


def pre_send(manual=False, rules=None, draft_id="d1", now=NOW, live=True):
    social_controls.register_hooks()
    ctx = {"draft_id": draft_id, "draft": {}, "draft_dir": Path("."), "ledger": social.Ledger(), "now": now, "live": live, "manual": manual,
           "fetch": None, "rules": rules or rules_with()}
    results = social_hooks.run("pre_send", ctx, rules=ctx["rules"], audit=Audit(audit_path(), run="r1", now=lambda: NOW))
    return social_hooks.first_non_pass(results)


class TestPauseFile:
    def test_pause_writes_the_file_with_time_and_reason_and_audits(self, monkeypatch, capsys):
        assert run_cli(monkeypatch, "social-pause", "--reason", "away on a trip") == 0
        doc = json.loads(Path(settings.SOCIAL_PAUSE_FILE).read_text(encoding="utf-8"))
        assert doc["reason"] == "away on a trip"
        assert dt.datetime.fromisoformat(doc["at"]).tzinfo is not None
        assert capsys.readouterr().out == f"paused: {settings.SOCIAL_PAUSE_FILE} (away on a trip)\n"
        [rec] = events("paused")
        assert (rec["action"], rec["reason"], rec["forced"]) == ("set", "away on a trip", False)

    def test_a_second_pause_is_refused_and_leaves_the_first_alone_until_forced(self, monkeypatch, capsys):
        assert run_cli(monkeypatch, "social-pause", "--reason", "first") == 0
        capsys.readouterr()
        assert run_cli(monkeypatch, "social-pause", "--reason", "second") == 1
        err = capsys.readouterr().err
        assert err.startswith("bio social-pause: already paused (first, since ")
        assert json.loads(Path(settings.SOCIAL_PAUSE_FILE).read_text(encoding="utf-8"))["reason"] == "first"
        assert run_cli(monkeypatch, "social-pause", "--reason", "second", "--force") == 0           # twin: --force replaces it
        assert json.loads(Path(settings.SOCIAL_PAUSE_FILE).read_text(encoding="utf-8"))["reason"] == "second"
        assert [r["forced"] for r in events("paused")] == [False, True]

    def test_resume_removes_the_file_and_audits(self, monkeypatch, capsys):
        run_cli(monkeypatch, "social-pause")
        capsys.readouterr()
        assert run_cli(monkeypatch, "social-resume") == 0
        assert not Path(settings.SOCIAL_PAUSE_FILE).exists()
        assert capsys.readouterr().out == f"resumed: {settings.SOCIAL_PAUSE_FILE} removed\n"
        assert [r["action"] for r in events("paused")] == ["set", "cleared"]

    def test_resume_without_a_pause_says_so_and_audits_nothing(self, monkeypatch, capsys):
        assert run_cli(monkeypatch, "social-resume") == 0
        assert capsys.readouterr().out == "not paused\n"
        assert events("paused") == []


class TestPauseHook:
    def test_a_pause_blocks_a_scheduled_post_and_audits_paused_once(self, monkeypatch):
        run_cli(monkeypatch, "social-pause", "--reason", "trip")
        hit = pre_send()
        assert hit is not None
        hook, verdict = hit
        assert (hook.name, verdict.action) == ("pause_switch", "block")
        assert verdict.reason.startswith("posting is paused since ") and verdict.reason.endswith("(trip); `bio social-resume` lifts it")
        blocked = [r for r in events("paused") if r["action"] == "blocked"]
        assert [(r["draft_id"], r["reason"], r["unreadable"]) for r in blocked] == [("d1", "trip", False)]
        assert [r["name"] for r in events("hook")] == ["pause_switch"]

    def test_without_a_pause_file_a_scheduled_post_passes(self):
        assert pre_send() is None
        assert events("paused") == []

    def test_a_manual_post_is_not_blocked_by_the_pause(self, monkeypatch):
        run_cli(monkeypatch, "social-pause")
        assert pre_send(manual=True) is None
        assert pre_send(manual=False) is not None                       # twin: the same pause blocks the scheduled run

    def test_with_the_rule_off_the_pause_does_nothing(self, monkeypatch):
        run_cli(monkeypatch, "social-pause")
        assert pre_send(rules=rules_with(pause_switch=False)) is None
        assert pre_send(rules=rules_with(pause_switch=True)) is not None

    @pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b"\xff\xfe\x00bad"])
    def test_a_corrupt_pause_file_still_pauses_with_a_clear_reason(self, content):
        Path(settings.SOCIAL_PAUSE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.SOCIAL_PAUSE_FILE).write_bytes(content)
        _, verdict = pre_send()
        assert verdict.action == "block"
        assert "is unreadable (" in verdict.reason and "remove it with `bio social-resume`" in verdict.reason
        assert [r["unreadable"] for r in events("paused")] == [True]

    def test_an_empty_json_object_is_a_pause_without_a_reason(self):
        Path(settings.SOCIAL_PAUSE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(settings.SOCIAL_PAUSE_FILE).write_text("{}", encoding="utf-8")
        _, verdict = pre_send()
        assert verdict.reason == "posting is paused; `bio social-resume` lifts it"

    def test_the_pause_runs_before_the_breaker(self, monkeypatch):
        run_cli(monkeypatch, "social-pause")
        write_audit(*[fail("r", f"d{i}") for i in range(3)])
        hook, _ = pre_send()
        assert hook.name == "pause_switch"


class TestBreakerCount:
    def count(self, limit=3):
        return social_controls.breaker_state(limit=limit)

    def test_no_audit_log_is_a_closed_breaker_at_zero(self):
        assert self.count() == {"count": 0, "limit": 3, "failures": [], "open": False, "cooldown_h": 6.0, "next_probe_at": None, "probe_due": False}

    def test_n_minus_one_failures_stay_closed_and_n_open(self):
        write_audit(fail("r1", "a"), fail("r2", "b"))
        assert (self.count()["count"], self.count()["open"]) == (2, False)
        write_audit(fail("r3", "c"))
        assert (self.count()["count"], self.count()["open"]) == (3, True)

    def test_the_limit_is_the_rules_number(self):
        write_audit(fail("r1", "a"), fail("r2", "b"))
        assert self.count(limit=2)["open"] is True
        assert self.count(limit=3)["open"] is False

    def test_a_posted_record_resets_the_count(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), ("posted", "r3", {"draft_id": "c"}), fail("r4", "d"))
        assert self.count()["count"] == 1

    def test_failures_before_a_posted_record_do_not_count_toward_n(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"), ("posted", "r4", {"draft_id": "d"}), fail("r5", "e"), fail("r6", "f"))
        assert (self.count()["count"], self.count()["open"]) == (2, False)

    def test_a_breaker_reset_record_resets_the_count(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"), ("breaker_reset", "r4", {}))
        assert (self.count()["count"], self.count()["open"]) == (0, False)
        write_audit(fail("r5", "d"))
        assert self.count()["count"] == 1

    def test_other_events_between_failures_do_not_break_the_streak(self):
        write_audit(fail("r1", "a"), ("post_decision", "r2", {"draft_id": "b", "manual": False}), ("post_skipped", "r2", {}),
                    fail("r2", "b"), ("hook", "r3", {"name": "x"}), fail("r3", "c"))
        assert self.count()["count"] == 3

    def test_a_failed_manual_post_is_not_counted_but_a_scheduled_one_is(self):
        write_audit(("post_decision", "r1", {"draft_id": "a", "manual": True}), fail("r1", "a"),
                    ("post_decision", "r2", {"draft_id": "b", "manual": False}), fail("r2", "b"))
        state = self.count()
        assert (state["count"], [f["draft_id"] for f in state["failures"]]) == (1, ["b"])

    def test_failures_are_listed_newest_first_with_their_details_capped_at_five(self):
        write_audit(*[fail(f"r{i}", f"d{i}", outcome="unknown") for i in range(7)])
        state = self.count()
        assert state["count"] == 7
        assert [f["draft_id"] for f in state["failures"]] == ["d6", "d5", "d4", "d3", "d2"]
        assert state["failures"][0]["outcome"] == "unknown" and state["failures"][0]["error"] == "boom"

    def test_a_torn_line_is_skipped_not_taken_for_a_boundary(self):
        write_audit(fail("r1", "a"))
        with open(audit_path(), "a", encoding="utf-8") as f:
            f.write('{"event": "posted", "tor\n')
        write_audit(fail("r2", "b"))
        assert self.count()["count"] == 2

    def test_only_the_tail_is_read_when_the_boundary_is_near_the_end(self, monkeypatch):
        monkeypatch.setattr(social_controls, "CHUNK", 64)
        write_audit(fail("old", "old"), *[("hook", "rx", {"name": "pad" * 20}) for _ in range(30)],
                    ("posted", "r1", {"draft_id": "p"}), fail("r2", "a"), fail("r3", "b"))
        read = []
        original = social_controls._reverse_lines

        def counting(path):
            for line in original(path):
                read.append(line)
                yield line

        monkeypatch.setattr(social_controls, "_reverse_lines", counting)
        assert self.count()["count"] == 2
        assert len(read) == 3                                           # fail, fail, posted: the long padding and the old failure stay unread

    def test_a_streak_across_many_chunks_is_counted_whole(self, monkeypatch):
        monkeypatch.setattr(social_controls, "CHUNK", 64)
        write_audit(("posted", "r0", {"draft_id": "p"}), *[fail(f"r{i}", f"d{i}") for i in range(12)])
        assert self.count()["count"] == 12


class TestBreakerHook:
    def test_at_n_failures_a_scheduled_post_is_blocked_and_circuit_open_audited(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c", outcome="unknown"))
        hook, verdict = pre_send()
        assert (hook.name, verdict.action) == ("circuit_breaker", "block")
        assert verdict.reason == "circuit breaker open: 3 scheduled posts failed in a row (limit 3); the next probe is allowed after 2026-10-22T14:00:00-04:00; `bio social-breaker --reset` closes it"
        [rec] = events("circuit_open")
        assert (rec["draft_id"], rec["count"], rec["limit"], [f["draft_id"] for f in rec["failures"]]) == ("d1", 3, 3, ["c", "b", "a"])

    def test_at_n_minus_one_failures_the_post_goes_through(self):
        write_audit(fail("r1", "a"), fail("r2", "b"))
        assert pre_send() is None
        assert events("circuit_open") == []

    def test_a_manual_post_is_not_blocked_by_an_open_breaker(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"))
        assert pre_send(manual=True) is None
        assert pre_send(manual=False) is not None

    def test_the_rules_number_sets_the_limit_and_off_disables_it(self):
        write_audit(fail("r1", "a"), fail("r2", "b"))
        assert pre_send(rules=rules_with(circuit_breaker=2)) is not None
        assert pre_send(rules=rules_with(circuit_breaker=False)) is None
        assert pre_send(rules=rules_with(circuit_breaker=3)) is None

    def test_a_reset_record_lets_posts_through_again(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"))
        assert pre_send() is not None
        write_audit(("breaker_reset", "r4", {}))
        assert pre_send() is None


class TestBreakerCommand:
    def test_prints_the_state_and_the_last_failures(self, monkeypatch, capsys):
        write_audit(fail("r1", "a"), fail("r2", "b", outcome="unknown"))
        assert run_cli(monkeypatch, "social-breaker") == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "circuit breaker: closed; 2 failed scheduled posts in a row (limit 3)"
        assert out[1].split() == ["2026-10-21T08:00:00-04:00", "b", "unknown", "boom"]
        assert out[2].split()[1:] == ["a", "nothing", "sent", "boom"]
        assert events("breaker_reset") == []

    def test_reports_open_at_n(self, monkeypatch, capsys):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"))
        run_cli(monkeypatch, "social-breaker")
        assert capsys.readouterr().out.splitlines()[0] == "circuit breaker: OPEN; 3 failed scheduled posts in a row (limit 3)"

    def test_reset_appends_a_breaker_reset_record_and_closes_it(self, monkeypatch, capsys):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"))
        assert run_cli(monkeypatch, "social-breaker", "--reset") == 0
        assert "breaker reset" in capsys.readouterr().out
        [rec] = events("breaker_reset")
        assert (rec["count"], rec["limit"]) == (3, 3)
        run_cli(monkeypatch, "social-breaker")
        assert capsys.readouterr().out.splitlines()[0] == "circuit breaker: closed; 0 failed scheduled posts in a row (limit 3)"

    def test_says_when_the_rule_is_off(self, monkeypatch, capsys):
        monkeypatch.setenv("BIO_SOCIAL_RULE_CIRCUIT_BREAKER", "off")
        run_cli(monkeypatch, "social-breaker")
        assert capsys.readouterr().out == ("circuit breaker: closed; 0 failed scheduled posts in a row (limit 3); "
                                           "the rule circuit_breaker is OFF, so nothing is blocked\n")


class TestInThePostingFlow:
    """The real post_next with the hooks registered: the controls stop the scheduled run, not the manual one."""

    def post(self, tmp_path, *, manual=False, send=None):
        ledger, drafts = drafted(tmp_path, 2)
        social_controls.register_hooks()
        send = send or RecordingSend()
        audit = Audit(audit_path(), run="flow", now=lambda: NOW)
        result = social.post_next(ledger, drafts, now=NOW, live=True, send=send, manual=manual, rules=rules_with(), audit=audit)
        return result, send

    def test_a_paused_scheduled_run_sends_nothing(self, tmp_path, monkeypatch):
        run_cli(monkeypatch, "social-pause", "--reason", "trip")
        result, send = self.post(tmp_path)
        assert result["status"] == "blocked" and result["reason"].endswith("(trip); `bio social-resume` lifts it")
        assert send.calls == []

    def test_the_same_pause_lets_a_manual_run_send(self, tmp_path, monkeypatch):
        run_cli(monkeypatch, "social-pause")
        result, send = self.post(tmp_path, manual=True)
        assert (result["status"], result["uri"], len(send.calls)) == ("posted", URI, 1)

    def test_failures_open_the_breaker_and_a_manual_success_closes_it(self, tmp_path):
        failing = RecordingSend(fail=social.NothingSent("network down"))
        for n in range(3):
            with pytest.raises(SocialError, match="nothing was posted"):
                self.post(tmp_path / f"t{n}", send=failing)
        assert social_controls.breaker_state(limit=3)["open"] is True
        blocked, send = self.post(tmp_path / "blocked")
        assert blocked["status"] == "blocked" and send.calls == []
        manual, send = self.post(tmp_path / "manual", manual=True)
        assert manual["status"] == "posted" and len(send.calls) == 1
        assert social_controls.breaker_state(limit=3) == {"count": 0, "limit": 3, "failures": [], "open": False, "cooldown_h": 6.0, "next_probe_at": None,
                                                          "probe_due": False}
        again, send = self.post(tmp_path / "again")
        assert again["status"] == "posted"


class TestBreakerCooldown:
    """Open at 3 failures (the last one 2026-10-22 08:00 -04:00); the cool-down is settings.SOCIAL_BREAKER_COOLDOWN_H = 6 h."""

    LAST = dt.datetime(2026, 10, 22, 8, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))

    def three_failures(self):
        write_audit(fail("r1", "a"), fail("r2", "b"), fail("r3", "c"))

    def test_the_default_cool_down_is_six_hours(self):
        assert settings.SOCIAL_BREAKER_COOLDOWN_H == 6.0

    def test_inside_the_cool_down_the_post_is_blocked_and_the_next_probe_time_is_named(self):
        self.three_failures()
        hook, verdict = pre_send(now=self.LAST + dt.timedelta(hours=5, minutes=59))
        assert (hook.name, verdict.action) == ("circuit_breaker", "block")
        assert verdict.reason == ("circuit breaker open: 3 scheduled posts failed in a row (limit 3); the next probe is allowed after "
                                  "2026-10-22T14:00:00-04:00; `bio social-breaker --reset` closes it")
        assert events("breaker_probe") == []

    def test_after_the_cool_down_one_live_probe_goes_through_and_is_audited(self):
        self.three_failures()
        assert pre_send(now=self.LAST + dt.timedelta(hours=6)) is None
        [rec] = events("breaker_probe")
        assert (rec["draft_id"], rec["count"], rec["limit"], rec["cooldown_h"]) == ("d1", 3, 3, 6.0)

    def test_a_dry_run_after_the_cool_down_passes_without_spending_the_probe(self):
        self.three_failures()
        assert pre_send(now=self.LAST + dt.timedelta(hours=7), live=False) is None
        assert events("breaker_probe") == []

    def test_a_manual_post_is_never_stopped_and_never_a_probe(self):
        self.three_failures()
        assert pre_send(manual=True, now=self.LAST + dt.timedelta(hours=7)) is None
        assert events("breaker_probe") == []

    def test_a_probe_that_did_not_end_in_a_post_restarts_the_cool_down(self):
        self.three_failures()
        write_audit(("breaker_probe", "r4", {"draft_id": "d", "at": "2026-10-23T08:00:00-04:00"}))
        state = social_controls.breaker_state(limit=3, now=dt.datetime(2026, 10, 23, 9, 0, tzinfo=self.LAST.tzinfo))
        assert (state["open"], state["probe_due"], state["next_probe_at"]) == (True, False, "2026-10-23T14:00:00-04:00")
        hook, verdict = pre_send(now=dt.datetime(2026, 10, 23, 13, 0, tzinfo=self.LAST.tzinfo))
        assert verdict.action == "block"
        assert pre_send(now=dt.datetime(2026, 10, 23, 14, 0, tzinfo=self.LAST.tzinfo)) is None

    def test_a_failed_probe_reopens_the_breaker_with_a_fresh_cool_down(self):
        self.three_failures()
        write_audit(("breaker_probe", "r4", {"draft_id": "d", "at": "2026-10-23T08:00:00-04:00"}),
                    ("post_failed", "r4", {"draft_id": "d", "outcome": "nothing sent", "error": "boom", "at": "2026-10-23T08:01:00-04:00"}))
        state = social_controls.breaker_state(limit=3, now=dt.datetime(2026, 10, 23, 14, 0, tzinfo=self.LAST.tzinfo))
        assert (state["count"], state["open"], state["probe_due"], state["next_probe_at"]) == (4, True, False, "2026-10-23T14:01:00-04:00")

    def test_a_posted_record_after_the_probe_closes_the_breaker(self):
        self.three_failures()
        write_audit(("breaker_probe", "r4", {"draft_id": "d"}), ("posted", "r4", {"draft_id": "d"}))
        state = social_controls.breaker_state(limit=3, now=self.LAST + dt.timedelta(days=3))
        assert (state["count"], state["open"], state["probe_due"], state["next_probe_at"]) == (0, False, False, None)
        assert pre_send(now=self.LAST + dt.timedelta(days=3)) is None

    def test_the_cool_down_comes_from_the_setting(self, monkeypatch):
        self.three_failures()
        monkeypatch.setattr(settings, "SOCIAL_BREAKER_COOLDOWN_H", 1.0)
        state = social_controls.breaker_state(limit=3, now=self.LAST + dt.timedelta(minutes=61))
        assert (state["cooldown_h"], state["next_probe_at"], state["probe_due"]) == (1.0, "2026-10-22T09:00:00-04:00", True)

    def test_a_closed_breaker_has_no_probe_time(self):
        write_audit(fail("r1", "a"))
        state = social_controls.breaker_state(limit=3, now=self.LAST)
        assert (state["open"], state["next_probe_at"], state["probe_due"]) == (False, None, False)

    def test_the_breaker_command_shows_the_next_probe_time(self, monkeypatch, capsys):
        self.three_failures()
        assert run_cli(monkeypatch, "social-breaker") == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "circuit breaker: OPEN; 3 failed scheduled posts in a row (limit 3)"
        assert lines[1] == "  next probe: 2026-10-22T14:00:00-04:00 (cool-down 6 h)" + ("; due now: the next scheduled post goes out as a probe"
                                                                                   if dt.datetime.now(settings.LOCAL_TZ) >= dt.datetime(2026, 10, 22, 14, 0, tzinfo=self.LAST.tzinfo) else "")

    def test_in_the_posting_flow_a_probe_that_succeeds_closes_the_breaker(self, tmp_path, monkeypatch):
        failing = RecordingSend(fail=social.NothingSent("network down"))
        flow = TestInThePostingFlow()
        for n in range(3):
            with pytest.raises(SocialError, match="nothing was posted"):
                flow.post(tmp_path / f"t{n}", send=failing)
        assert social_controls.breaker_state(limit=3, now=NOW)["probe_due"] is False
        blocked, send = flow.post(tmp_path / "blocked")
        assert (blocked["status"], send.calls) == ("blocked", [])
        monkeypatch.setattr(settings, "SOCIAL_BREAKER_COOLDOWN_H", 0.0)        # the flow's clock is frozen at NOW: a zero cool-down is "elapsed"
        probe, send = flow.post(tmp_path / "probe")
        assert (probe["status"], len(send.calls), len(events("breaker_probe"))) == ("posted", 1, 1)
        assert social_controls.breaker_state(limit=3, now=NOW)["open"] is False
