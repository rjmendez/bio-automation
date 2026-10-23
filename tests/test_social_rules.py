"""The posting rules: every one optional, its state from default / settings / file / environment, and every decision in the audit log."""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import urllib.error
from pathlib import Path

import pytest
from PIL import Image
from test_social import NOW, RecordingDescribe, RecordingFetch, RecordingSend, drafted, events, jpeg, labels, no_model, observation, rules_with, write_draft

from bio_automation import settings, social, themes
from bio_automation.social import Ledger, SkipObservation, SocialError
from bio_automation.social_rules import RULES, Audit, RulesError, load_rules, rules_toml

NOWHERE = Path(os.devnull)


class TestRuleStates:
    def test_the_defaults_are_every_rule_on_except_the_two_month_rules(self):
        r = load_rules(NOWHERE, env={})
        assert r.snapshot() == {
            "daily_cap": {"on": True, "value": 3}, "manual_exempt_from_cap": {"on": True}, "auto_approve": {"on": True},
            "month_candidates": {"on": False}, "month_posting": {"on": False}, "themes": {"on": True}, "theme_day_only": {"on": True},
            "theme_species_first": {"on": True}, "lower_quality_photos": {"on": True, "value": 1000}, "prints_first": {"on": True}, "year_rotation": {"on": True}, "skip_casual": {"on": True},
            "strip_review_claim": {"on": True}}
        assert [r.state(x.name).source for x in RULES] == ["default"] * len(RULES)

    def test_the_older_settings_still_set_the_cap_auto_approve_and_themes(self, monkeypatch):
        monkeypatch.setattr(settings, "SOCIAL_POSTS_PER_DAY", 5)
        monkeypatch.setattr(settings, "SOCIAL_AUTO_APPROVE", False)
        monkeypatch.setattr(settings, "SOCIAL_THEMES", False)
        r = load_rules(NOWHERE, env={})
        assert [(r.on(n), r.value(n), r.state(n).source) for n in ("daily_cap", "auto_approve", "themes")] == [
            (True, 5, "settings"), (False, None, "settings"), (False, None, "settings")]
        assert r.state("prints_first").source == "default"

    def write(self, tmp_path, body):
        p = tmp_path / "rules.toml"
        p.write_text(body, encoding="utf-8")
        return p

    def test_the_file_beats_the_settings_and_the_environment_beats_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "SOCIAL_AUTO_APPROVE", False)
        p = self.write(tmp_path, "[rules.auto_approve]\nenabled = true\n[rules.daily_cap]\nenabled = true\nvalue = 8\n[rules.prints_first]\nenabled = false\n")
        r = load_rules(p, env={})
        assert [(n, r.on(n), r.value(n), r.state(n).source) for n in ("auto_approve", "daily_cap", "prints_first")] == [
            ("auto_approve", True, None, "file"), ("daily_cap", True, 8, "file"), ("prints_first", False, None, "file")]
        env = {"BIO_SOCIAL_RULE_DAILY_CAP": "12", "BIO_SOCIAL_RULE_PRINTS_FIRST": "on", "BIO_SOCIAL_RULE_SKIP_CASUAL": "off"}
        r2 = load_rules(p, env=env)
        assert [(n, r2.on(n), r2.value(n), r2.state(n).source) for n in ("daily_cap", "prints_first", "skip_casual")] == [
            ("daily_cap", True, 12, "env"), ("prints_first", True, None, "env"), ("skip_casual", False, None, "env")]

    def test_a_number_switches_a_rule_on_with_that_value_and_off_switches_it_off_keeping_its_number(self):
        assert load_rules(NOWHERE, env={"BIO_SOCIAL_RULE_DAILY_CAP": "off"}).on("daily_cap") is False
        assert load_rules(NOWHERE, env={"BIO_SOCIAL_RULE_DAILY_CAP": "off"}).value("daily_cap") == 3
        assert load_rules(NOWHERE, env={"BIO_SOCIAL_RULE_MONTH_POSTING": "1"}).on("month_posting") is True

    @pytest.mark.parametrize("env,match", [
        ({"BIO_SOCIAL_RULE_DAILY_CAP": "lots"}, r"^BIO_SOCIAL_RULE_DAILY_CAP: expected on, off or a number, got 'lots'$"),
        ({"BIO_SOCIAL_RULE_THEMES": "5"}, r"^BIO_SOCIAL_RULE_THEMES: rule 'themes' has no value, use on or off$")])
    def test_a_bad_environment_value_is_refused_naming_the_variable(self, env, match):
        with pytest.raises(RulesError, match=match):
            load_rules(NOWHERE, env=env)

    @pytest.mark.parametrize("body,match", [
        ("[rules.bogus]\nenabled = true\n", r"rules\.toml: unknown rule 'bogus' \(rules: daily_cap, "),
        ("[rules.themes]\nenabled = \"yes\"\n", r"rule 'themes': enabled must be true or false"),
        ("[rules.daily_cap]\nvalue = -1\n", r"rule 'daily_cap': value must be a whole number"),
        ("[rules.themes]\nvalue = 3\n", r"rule 'themes' has no value"),
        ("[rules\n", r"rules\.toml: ")])
    def test_a_bad_file_is_refused_naming_the_rule(self, tmp_path, body, match):
        with pytest.raises(RulesError, match=match):
            load_rules(self.write(tmp_path, body), env={})

    def test_the_written_defaults_read_back_to_the_same_switches(self, tmp_path):
        text = rules_toml()
        assert "[rules.daily_cap]\nenabled = true\nvalue = 3" in text and "[rules.month_posting]\nenabled = false" in text
        back = load_rules(self.write(tmp_path, text), env={})
        assert back.snapshot() == load_rules(NOWHERE, env={}).snapshot()
        assert {back.state(r.name).source for r in RULES} == {"file"}


class TestAudit:
    def test_each_record_has_time_run_and_event_and_a_run_shares_one_id(self, tmp_path):
        a = Audit(tmp_path / "a.jsonl", run="r1", now=lambda: dt.datetime(2026, 10, 22, 20, 0, 5, tzinfo=dt.timezone.utc))
        a.log("post_decision", draft_id="d1", reasons=["x"])
        a.log("posted", draft_id="d1")
        assert a.records() == [
            {"at": "2026-10-22T20:00:05+00:00", "run": "r1", "event": "post_decision", "draft_id": "d1", "reasons": ["x"]},
            {"at": "2026-10-22T20:00:05+00:00", "run": "r1", "event": "posted", "draft_id": "d1"}]
        assert Audit(tmp_path / "a.jsonl").records() == a.records() and Audit(tmp_path / "none.jsonl").records() == []


class RecordingAuditAtSend:
    """A send that records what the audit log held at the moment of sending."""

    def __init__(self, audit):
        self.audit, self.held, self.calls = audit, None, []

    def __call__(self, request):
        self.held = [r["event"] for r in self.audit.records()]
        self.calls.append(request)
        return "at://did:plc:me/app.bsky.feed.post/3x", "bafy"


class TestPostAudit:
    def test_a_scheduled_post_is_audited_with_the_rules_and_the_reasons_before_it_is_sent(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2)
        audit = Audit(tmp_path / "audit.jsonl", run="r1")
        send = RecordingAuditAtSend(audit)
        r = social.post_next(ledger, drafts, now=NOW, live=True, send=send, audit=audit)
        assert r["status"] == "posted"
        assert send.held == ["approved", "post_decision"]
        recs = audit.records()
        assert [x["event"] for x in recs] == ["approved", "post_decision", "posted"]
        assert (recs[0]["draft_id"], recs[0]["by"], recs[0]["review_claim_removed"]) == ("2026-10-obs1", "auto", False)
        d = recs[1]
        assert (d["live"], d["draft_id"], d["manual"], d["themed"], d["obs_ids"], d["approved_by_this_run"]) == (True, "2026-10-obs1", False, False, [1], True)
        assert d["reasons"] == ["daily_cap: 0 of 3 scheduled posts used today", "month_posting: off, drafts of any month are eligible",
                                "auto_approve: nothing approved, the oldest drafted draft 2026-10-obs1 is approved by this run"]
        assert d["rules"] == load_rules(NOWHERE, env={}).snapshot()
        assert (recs[2]["draft_id"], recs[2]["uri"]) == ("2026-10-obs1", "at://did:plc:me/app.bsky.feed.post/3x")

    def test_a_dry_run_is_audited_too_and_says_it_was_not_live(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        audit = Audit(tmp_path / "audit.jsonl")
        social.post_next(ledger, drafts, now=NOW, audit=audit)
        assert [(x["event"], x["live"]) for x in audit.records()] == [("post_decision", False)]

    def test_a_skipped_run_says_why(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 3)
        audit = Audit(tmp_path / "audit.jsonl")
        for _ in range(3):
            social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), audit=audit)
        social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), audit=audit)
        skipped = [x for x in audit.records() if x["event"] == "post_skipped"][0]
        assert (skipped["status"], skipped["cap"], skipped["posted_today"], skipped["manual"]) == (
            "done for today", 3, ["2026-10-obs1", "2026-10-obs2", "2026-10-obs3"], False)
        ledger2, drafts2 = drafted(tmp_path / "b", 1)
        social.post_next(ledger2, drafts2, now=NOW, auto_approve=False, audit=audit)
        last = audit.records()[-1]
        assert (last["event"], last["status"], last["auto_approve"], last["drafted"]) == ("post_skipped", "nothing approved", False, 1)

    def test_a_decision_that_cannot_be_recorded_is_not_acted_on(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        (tmp_path / "audit.jsonl").mkdir()                                  # a folder where the log file should be
        send = RecordingSend()
        with pytest.raises(SocialError, match=r"^the audit log .*audit\.jsonl cannot be written .*; the draft was not approved$"):
            social.post_next(ledger, drafts, now=NOW, live=True, send=send, audit=Audit(tmp_path / "audit.jsonl"))
        assert send.calls == [] and events(ledger, "2026-10-obs1") == ["drafted"]                    # not even approved
        social.approve(ledger, drafts, "2026-10-obs1", audit=Audit(tmp_path / "other.jsonl"))         # approved elsewhere: the decision itself fails next
        with pytest.raises(SocialError, match=r"^the audit log .*audit\.jsonl cannot be written .*; nothing was sent$"):
            social.post_next(ledger, drafts, now=NOW, live=True, send=send, audit=Audit(tmp_path / "audit.jsonl"))
        assert send.calls == [] and events(ledger, "2026-10-obs1") == ["drafted", "approved"]

    def test_a_post_that_went_out_is_never_undone_by_a_failing_audit_and_the_result_says_so(self, tmp_path):
        class AuditFailingAfterSend(Audit):
            def log(self, event, **fields):
                if event == "posted":
                    raise OSError("disk full")
                return super().log(event, **fields)

        ledger, drafts = drafted(tmp_path, 1)
        r = social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), audit=AuditFailingAfterSend(tmp_path / "audit.jsonl"))
        assert r["status"] == "posted" and r["audit"] == "the posted record could not be written to the audit log"
        assert events(ledger, "2026-10-obs1") == ["drafted", "approved", "posting", "posted"]

    def test_a_failed_send_is_audited_as_nothing_sent_or_unknown(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        audit = Audit(tmp_path / "audit.jsonl")
        with pytest.raises(SocialError, match="nothing was posted"):
            social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(fail=social.NothingSent("no session")), audit=audit)
        assert [(x["event"], x.get("outcome")) for x in audit.records()][-1] == ("post_failed", "nothing sent")


class TestRulesChangeBehaviour:
    def test_daily_cap_off_means_no_limit_and_its_number_is_the_limit_when_on(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 6)
        for _ in range(5):
            assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=rules_with(daily_cap=False))["status"] == "posted"
        l2, d2 = drafted(tmp_path / "b", 6)
        got = [social.post_next(l2, d2, now=NOW, live=True, send=RecordingSend(), rules=rules_with(daily_cap=2))["status"] for _ in range(3)]
        assert got == ["posted", "posted", "done for today"]

    def test_manual_posts_count_toward_the_cap_when_manual_exempt_from_cap_is_off(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 3)
        strict = rules_with(daily_cap=1, manual_exempt_from_cap=False)
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=strict, manual=True)["status"] == "posted"
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=strict, manual=True)["status"] == "done for today"
        exempt = rules_with(daily_cap=1)
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=exempt, manual=True)["status"] == "posted"

    def test_auto_approve_off_posts_nothing_until_the_owner_approves(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        off = rules_with(auto_approve=False)
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=off)["status"] == "nothing approved"
        social.approve(ledger, drafts, "2026-10-obs1", rules=off)
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=off)["status"] == "posted"

    def test_theme_day_only_off_lets_a_themed_draft_wait_in_the_ordinary_queue(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        ledger.append("drafted", "2026-10-30-fungi-friday", obs_id=9, obs_ids=[9, 10], month="2026-10", theme="fungi-friday", for_date="2026-10-30")
        d = write_draft(drafts, "2026-10-30-fungi-friday", 9, "2026-10")
        assert d.is_dir()
        on = social.post_next(ledger, drafts, now=NOW, rules=rules_with(auto_approve=False))
        assert on == {"status": "nothing approved", "month": "2026-10"} or on["draft_id"] != "2026-10-30-fungi-friday"
        off = social.post_next(ledger, drafts, now=NOW, rules=rules_with(theme_day_only=False))
        assert off["draft_id"] == "2026-10-obs1"                                  # the older ordinary draft first
        social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend(), rules=rules_with(theme_day_only=False))
        again = social.post_next(ledger, drafts, now=NOW, rules=rules_with(theme_day_only=False))
        assert again["draft_id"] == "2026-10-30-fungi-friday"                     # then the themed one, any day
        assert social.post_next(ledger, drafts, now=NOW, rules=rules_with())["status"] == "nothing approved"     # twin: day-only keeps it for the 30th

    def test_todays_themed_draft_goes_before_an_older_ordinary_one_only_while_theme_day_only_is_on(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")
        ledger.append("drafted", "2026-10-21-fungi-friday", obs_id=9, obs_ids=[9, 10], month="2026-10", theme="fungi-friday", for_date="2026-10-21")
        write_draft(drafts, "2026-10-21-fungi-friday", 9, "2026-10")
        assert social.post_next(ledger, drafts, now=NOW, rules=rules_with())["draft_id"] == "2026-10-21-fungi-friday"
        assert social.post_next(ledger, drafts, now=NOW, rules=rules_with(theme_day_only=False))["draft_id"] == "2026-10-obs1"

    def test_year_rotation_off_drafts_from_one_shuffled_pool_instead_of_alternating_the_years(self, tmp_path):
        g, index = tmp_path / "gallery", {"files": {}}
        for o in range(1, 13):
            year = 2024 + o % 2
            rel = f"{year}/{year}-10/x{o}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = o

        def order(rules, sub):
            fetch = RecordingFetch({o: observation(o) for o in range(1, 13)})
            social.draft_month(index, Ledger(tmp_path / sub / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / sub / "d",
                               keep_pending=12, model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model,
                               style_source=lambda: None, prints={}, rules=rules)
            return [("2025" if o % 2 else "2024") for o in fetch.calls]

        rotating = order(rules_with(), "a")
        assert rotating == ["2025", "2024"] * 6
        assert order(rules_with(year_rotation=False), "b") != rotating

    def test_strip_review_claim_off_keeps_the_line_and_says_so_in_the_audit(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2, legacy=True)
        audit = Audit(tmp_path / "audit.jsonl")
        social.approve(ledger, drafts, "2026-10-obs1", rules=rules_with(strip_review_claim=False), audit=audit)
        assert "reviewed" in (drafts / "2026-10-obs1" / "draft.json").read_text(encoding="utf-8")
        social.approve(ledger, drafts, "2026-10-obs2", audit=audit)
        assert "reviewed" not in (drafts / "2026-10-obs2" / "draft.json").read_text(encoding="utf-8")
        assert [x["review_claim_removed"] for x in audit.records()] == [False, True]

    def draft_two(self, tmp_path, rules):
        g, index = tmp_path / "gallery", {"files": {}}
        for obs, rel in ((1, "2024/2024-10/a.jpg"), (2, "2025/2025-10/b.jpg")):
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = obs
        audit = Audit(tmp_path / "audit.jsonl")
        report = social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "drafts",
                                    keep_pending=5, model="vis:1", fetch=RecordingFetch({1: observation(1), 2: observation(2, grade="casual")}),
                                    read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None,
                                    prints={}, rules=rules, audit=audit)
        return report, audit

    def test_skip_casual_off_drafts_a_casual_observation_too(self, tmp_path):
        on, _ = self.draft_two(tmp_path / "a", rules_with())
        off, _ = self.draft_two(tmp_path / "b", rules_with(skip_casual=False))
        assert (on["drafted"], on["skipped"]) == (["2026-10-obs1"], {2: "observation 2 is casual"})
        assert (sorted(off["drafted"]), off["skipped"]) == (["2026-10-obs1", "2026-10-obs2"], {})

    def test_drafting_is_audited_the_run_each_draft_and_each_skip(self, tmp_path):
        _, audit = self.draft_two(tmp_path, rules_with())
        recs = audit.records()
        assert [x["event"] for x in recs] == ["draft_run", "draft_skipped", "drafted"] or [x["event"] for x in recs] == ["draft_run", "drafted", "draft_skipped"]
        run = recs[0]
        assert (run["month"], run["pending"], run["keep_pending"], run["need"], run["candidates"], run["eligible"], run["with_print"]) == (
            "2026-10", 0, 5, 5, 2, 2, 0)
        assert run["rules"] == rules_with().snapshot()
        assert {x["event"]: x for x in recs}["draft_skipped"]["reason"] == "observation 2 is casual"

    def test_a_full_queue_is_audited_as_such(self, tmp_path):
        g, index = tmp_path / "gallery", {"files": {"2024/2024-10/a.jpg": 1}}
        ledger = Ledger(tmp_path / "l.jsonl")
        ledger.append("drafted", "2026-10-obs9", obs_id=9, month="2026-10")
        audit = Audit(tmp_path / "audit.jsonl")
        social.draft_month(index, ledger, today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "d", keep_pending=1, prints={},
                           pick_model=no_model, rules=rules_with(), audit=audit)
        assert [(x["event"], x["outcome"], x["pending"]) for x in audit.records()] == [("draft_run", "queue full", 1)]

    def test_year_rotation_off_is_one_shuffled_pool_and_on_alternates_the_years(self):
        cands = {o: [f"{2024 + (o % 2)}/{2024 + (o % 2)}-10/x{o}.jpg"] for o in range(1, 13)}
        rotating = social.order_candidates(cands, set(), "seed")
        years = [cands[o][0][:4] for o in rotating]
        assert years == ["2025", "2024"] * 6
        plain = social.order_candidates(cands, set(), "seed", rotate_years=False)
        assert sorted(plain) == sorted(rotating) and [cands[o][0][:4] for o in plain] != years
        assert plain == social.order_candidates(cands, set(), "seed", rotate_years=False)

    def test_themes_off_drafts_nothing_loads_no_model_and_says_so(self, tmp_path):
        audit = Audit(tmp_path / "audit.jsonl")
        report = social.draft_themes({"files": {}}, Ledger(tmp_path / "l.jsonl"), {}, themes.DEFAULT_THEMES, today=dt.date(2026, 10, 21), days_ahead=7,
                                     gallery_dir=tmp_path, drafts_dir=tmp_path / "d", pick_model=no_model, rules=rules_with(themes=False), audit=audit)
        assert report == {"drafted": [], "skipped": {}, "short": {}}
        assert [(x["event"], x["outcome"]) for x in audit.records()] == [("theme_run", "rule themes is off")]


class TestCli:
    def isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "SOCIAL_RULES_FILE", tmp_path / "rules.toml")
        monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "audit.jsonl")

    def test_social_rules_lists_every_rule_with_its_state_and_where_it_came_from(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.isolate(tmp_path, monkeypatch)
        (tmp_path / "rules.toml").write_text("[rules.prints_first]\nenabled = false\n", encoding="utf-8")
        monkeypatch.setenv("BIO_SOCIAL_RULE_DAILY_CAP", "6")
        monkeypatch.setattr("sys.argv", ["bio", "social-rules"])
        assert cli.main() == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"posting rules (file {tmp_path / 'rules.toml'}; BIO_SOCIAL_RULE_<NAME>=on|off|<number> overrides):"
        assert len(out) == 1 + len(RULES)
        assert out[1].startswith("  daily_cap                ON =6   [env     ] At most <value> scheduled posts a day")
        by = {line.split()[0]: line for line in out[1:]}
        assert by["prints_first"].startswith("  prints_first             off     [file    ] ")
        assert by["month_posting"].startswith("  month_posting            off     [default ] ")
        assert by["lower_quality_photos"].startswith("  lower_quality_photos     ON =1000 [default ] Observations without a gallery photo are eligible too")

    def test_social_rules_write_defaults_once(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.isolate(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.argv", ["bio", "social-rules", "--write-defaults"])
        assert cli.main() == 0
        assert (tmp_path / "rules.toml").read_text(encoding="utf-8") == rules_toml()
        capsys.readouterr()
        assert cli.main() == 1
        assert capsys.readouterr().err == f"bio social-rules: {tmp_path / 'rules.toml'} exists: edit it, or move it away first\n"

    def test_a_broken_rules_file_is_a_one_line_error(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.isolate(tmp_path, monkeypatch)
        (tmp_path / "rules.toml").write_text("[rules.bogus]\nenabled = true\n", encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["bio", "social-rules"])
        assert cli.main() == 1
        assert capsys.readouterr().err.startswith(f"bio social-rules: {tmp_path / 'rules.toml'}: unknown rule 'bogus'")

    def test_social_audit_shows_the_last_records_and_filters_by_run_and_event(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.isolate(tmp_path, monkeypatch)
        a, b = Audit(tmp_path / "audit.jsonl", run="r1"), Audit(tmp_path / "audit.jsonl", run="r2")
        a.log("drafted", draft_id="d1")
        a.log("post_decision", draft_id="d1")
        b.log("post_decision", draft_id="d2")
        monkeypatch.setattr("sys.argv", ["bio", "social-audit", "--run", "r1"])
        assert cli.main() == 0
        assert [json.loads(x)["event"] for x in capsys.readouterr().out.splitlines()] == ["drafted", "post_decision"]
        monkeypatch.setattr("sys.argv", ["bio", "social-audit", "--event", "post_decision", "--tail", "1"])
        assert cli.main() == 0
        assert [json.loads(x)["draft_id"] for x in capsys.readouterr().out.splitlines()] == ["d2"]


def snapshot(tmp_path, observations):
    """A tiny iNat snapshot: observations = [(obs id, observed_on, [(photo id, width, height, license)])]."""
    rows = [{"id": o, "observed_on": d, "photos": [{"id": pid, "dims": {"width": w, "height": h}, "license": lic} for pid, w, h, lic in photos]}
            for o, d, photos in observations]
    p = tmp_path / "inat_observations.json"
    p.write_text(json.dumps({"login": "me", "count": len(rows), "observations": rows}), encoding="utf-8")
    return p


def jpeg_bytes(size):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 90, 10)).save(buf, "JPEG")
    return buf.getvalue()


def open_data(pid, size):
    return social.OPEN_DATA_URL.format(id=pid, size=size)


class Photos:
    """Stands in for the download: {url: bytes, or an int HTTP status, or an exception to raise}; records every url asked for."""

    def __init__(self, by_url):
        self.by_url, self.calls = by_url, []

    def __call__(self, url):
        self.calls.append(url)
        v = self.by_url[url]
        if isinstance(v, int):
            raise urllib.error.HTTPError(url, v, "status", {}, None)
        if isinstance(v, Exception):
            raise v
        return v


class Sleeps:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


CC = "cc-by-nc"


class TestInatPhotoCandidates:
    def test_each_observation_gets_its_biggest_licensed_photo_when_it_is_big_enough(self, tmp_path):
        snap = snapshot(tmp_path, [
            (10, "2020-05-04", [(111, 2048, 1365, CC), (112, 1024, 683, CC), (113, 4000, 3000, None)]),     # the unlicensed one is not in open data
            (11, "2021-01-02", [(114, 2048, 1365, CC)]),                                                      # the gallery has this one
            (12, "2022-02-03", [(115, 800, 600, CC)]),                                                        # too small
            (13, None, [(116, 1500, 1000, CC)]), (14, "2023-03-04", [])])
        assert social.inat_photo_candidates(snap, min_edge=1000, exclude={11}) == {
            10: {"photo_id": 111, "edge": 2048, "observed_on": "2020-05-04"}, 13: {"photo_id": 116, "edge": 1500, "observed_on": None}}
        assert sorted(social.inat_photo_candidates(snap, min_edge=500, exclude={11})) == [10, 12, 13]

    def test_no_snapshot_gives_nothing_and_a_snapshot_gives_something(self, tmp_path):
        snap = snapshot(tmp_path, [(10, "2020-05-04", [(111, 2048, 1365, CC)])])
        assert social.inat_photo_candidates(tmp_path / "missing.json", min_edge=1000) == {}
        assert social.inat_photo_candidates(snap, min_edge=1000) != {}


class TestFetchBestPhoto:
    def test_the_original_is_downloaded_once_kept_and_the_downloads_are_paced(self, tmp_path):
        big = jpeg_bytes((2048, 1365))
        get, sleeps = Photos({open_data(5, "original"): big}), Sleeps()
        got = social.fetch_best_photo(5, tmp_path / "best", min_edge=1000, get=get, sleep=sleeps, pause_s=1.5)
        assert got == tmp_path / "best" / "5.jpg" and got.read_bytes() == big
        assert (get.calls, sleeps.calls) == ([open_data(5, "original")], [1.5])
        again = social.fetch_best_photo(5, tmp_path / "best", min_edge=1000, get=get, sleep=sleeps)
        assert again == got and len(get.calls) == 1 and sleeps.calls == [1.5]                                  # kept: no second download

    def test_when_the_original_is_missing_the_large_copy_is_used_if_it_is_big_enough(self, tmp_path):
        large = jpeg_bytes((1024, 683))
        get = Photos({open_data(5, "original"): 404, open_data(5, "large"): large})
        assert social.fetch_best_photo(5, tmp_path, min_edge=1000, get=get, sleep=Sleeps()).read_bytes() == large
        assert get.calls == [open_data(5, "original"), open_data(5, "large")]

    def test_a_photo_with_no_copy_big_enough_is_skipped_and_nothing_is_kept(self, tmp_path):
        get = Photos({open_data(5, "original"): jpeg_bytes((700, 500)), open_data(5, "large"): 403})
        with pytest.raises(SkipObservation, match=r"^iNat photo 5: no copy of at least 1000 px in iNat's open data$"):
            social.fetch_best_photo(5, tmp_path, min_edge=1000, get=get, sleep=Sleeps())
        assert list(tmp_path.iterdir()) == []
        assert social.fetch_best_photo(5, tmp_path, min_edge=600, get=Photos({open_data(5, "original"): jpeg_bytes((700, 500))}), sleep=Sleeps()).is_file()

    def test_a_kept_photo_under_the_minimum_is_fetched_again_and_an_unreadable_download_falls_back(self, tmp_path):
        jpeg(tmp_path / "5.jpg", size=(500, 333))
        get = Photos({open_data(5, "original"): b"not an image", open_data(5, "large"): jpeg_bytes((1100, 700))})
        got = social.fetch_best_photo(5, tmp_path, min_edge=1000, get=get, sleep=Sleeps())
        assert Image.open(got).size == (1100, 700) and len(get.calls) == 2

    def test_a_server_error_or_a_dead_network_skips_the_observation_naming_the_cause(self, tmp_path):
        with pytest.raises(SkipObservation, match=r"^iNat photo 5: HTTP 500 from https://inaturalist-open-data"):
            social.fetch_best_photo(5, tmp_path, min_edge=1000, get=Photos({open_data(5, "original"): 500}), sleep=Sleeps())
        with pytest.raises(SkipObservation, match=r"^iNat photo 5: URLError: .*no route"):
            social.fetch_best_photo(5, tmp_path, min_edge=1000, get=Photos({open_data(5, "original"): urllib.error.URLError("no route")}), sleep=Sleeps())


class TestResolvePhotos:
    def test_tokens_become_downloaded_paths_and_other_paths_stay(self):
        calls = []
        got = social.resolve_photos(["2025/x.jpg", "inat-photo:701", "D:/a/b.jpg"], lambda pid: calls.append(pid) or Path(f"best/{pid}.jpg"))
        assert got == ["2025/x.jpg", str(Path("best/701.jpg")), "D:/a/b.jpg"] and calls == [701]
        assert social.resolve_photos(["2025/x.jpg"], None) == ["2025/x.jpg"]

    def test_a_token_without_a_fetcher_is_an_error(self):
        with pytest.raises(SocialError, match=r"^an iNat photo has to be fetched but no fetch_photo was given$"):
            social.resolve_photos(["inat-photo:701"], None)


class TestDraftsWithBestPhotos:
    def world(self, tmp_path):
        g, index = tmp_path / "gallery", {"files": {}}
        for o in (1, 2):
            rel = f"2025/2025-10/x{o}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = o
        snap = snapshot(tmp_path, [(2, "2025-10-02", [(201, 2048, 1365, CC)]), (7, "2019-03-04", [(701, 2048, 1365, CC)]),
                                   (8, "2019-10-09", [(801, 2048, 1365, CC)]), (9, "2019-10-10", [(901, 800, 600, CC)])])
        return g, index, social.inat_photo_candidates(snap, min_edge=1000, exclude={1, 2})

    def photo_fetcher(self, tmp_path, fail=()):
        calls = []

        def fetch(pid):
            calls.append(pid)
            if pid in fail:
                raise SkipObservation(f"iNat photo {pid}: no copy of at least 1000 px in iNat's open data")
            return jpeg(tmp_path / "best" / f"{pid}.jpg", size=(1500, 1000))
        fetch.calls = calls
        return fetch

    def draft(self, tmp_path, rules, fail=(), **kw):
        g, index, local = self.world(tmp_path)
        fetch = RecordingFetch({o: observation(o) for o in (1, 2, 7, 8, 9)})
        photos = self.photo_fetcher(tmp_path, fail)
        audit = Audit(tmp_path / "audit.jsonl")
        report = social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "drafts",
                                    keep_pending=10, model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model,
                                    style_source=lambda: None, prints={}, rules=rules, audit=audit, local=local, fetch_photo=photos, **kw)
        return report, fetch, photos, audit

    def test_an_observation_the_gallery_does_not_have_is_drafted_with_its_fetched_original_when_the_rule_is_on(self, tmp_path):
        report, fetch, photos, audit = self.draft(tmp_path / "a", rules_with())
        assert sorted(report["drafted"]) == ["2026-10-obs1", "2026-10-obs2", "2026-10-obs7", "2026-10-obs8"]
        assert sorted(photos.calls) == [701, 801]                                                          # only the drafted ones are downloaded, the gallery's never
        draft = social.load_draft(tmp_path / "a" / "drafts" / "2026-10-obs7")
        assert [i["source"] for i in draft["images"]] == [str(tmp_path / "a" / "best" / "701.jpg")] and draft["images"][0]["width"] == 1500
        assert [x for x in audit.records() if x["event"] == "draft_run"][0]["local_only"] == 2
        assert 9 not in fetch.calls                                                                         # its only photo is 800 px
        off, _, photos2, _ = self.draft(tmp_path / "b", rules_with(lower_quality_photos=False))
        assert sorted(off["drafted"]) == ["2026-10-obs1", "2026-10-obs2"] and photos2.calls == []

    def test_an_observation_the_gallery_has_keeps_its_gallery_photo_even_when_the_candidates_list_it(self, tmp_path):
        g, index, _ = self.world(tmp_path)
        everything = social.inat_photo_candidates(tmp_path / "inat_observations.json", min_edge=1000)         # no exclude: obs 2 is in the list
        assert 2 in everything
        photos = self.photo_fetcher(tmp_path)
        social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "drafts", keep_pending=10,
                           model="vis:1", fetch=RecordingFetch({o: observation(o) for o in (1, 2, 7, 8, 9)}), read_label=labels,
                           describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None, prints={}, rules=rules_with(),
                           audit=Audit(tmp_path / "audit.jsonl"), local=everything, fetch_photo=photos)
        assert 201 not in photos.calls
        assert [i["source"] for i in social.load_draft(tmp_path / "drafts" / "2026-10-obs2")["images"]] == ["2025/2025-10/x2.jpg"]

    def test_a_photo_that_cannot_be_fetched_skips_its_observation_and_the_audit_says_why(self, tmp_path):
        report, _, _, audit = self.draft(tmp_path, rules_with(), fail=(801,))
        assert sorted(report["drafted"]) == ["2026-10-obs1", "2026-10-obs2", "2026-10-obs7"]
        assert report["skipped"] == {8: "iNat photo 801: no copy of at least 1000 px in iNat's open data"}
        assert [(x["obs_id"], x["reason"]) for x in audit.records() if x["event"] == "draft_skipped"] == [
            (8, "iNat photo 801: no copy of at least 1000 px in iNat's open data")]

    def test_with_month_candidates_on_only_this_months_observations_qualify(self, tmp_path):
        report, _, photos, _ = self.draft(tmp_path, rules_with(month_candidates=True))
        assert sorted(report["drafted"]) == ["2026-10-obs1", "2026-10-obs2", "2026-10-obs8"] and photos.calls == [801]

    def test_the_minimum_edge_is_the_rules_number(self, tmp_path):
        snap = snapshot(tmp_path, [(9, "2019-10-10", [(901, 800, 600, CC)])])
        assert social.inat_photo_candidates(snap, min_edge=rules_with().value("lower_quality_photos")) == {}
        assert list(social.inat_photo_candidates(snap, min_edge=rules_with(lower_quality_photos=700).value("lower_quality_photos"))) == [9]

    def test_observations_without_a_gallery_photo_take_their_turn_by_the_year_they_were_seen(self, tmp_path):
        g, index, local = self.world(tmp_path)
        local = {o: v for o, v in local.items() if o in (7, 8)}
        fetch = RecordingFetch({o: observation(o) for o in (1, 2, 7, 8)})
        social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "d", keep_pending=4,
                           model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None,
                           prints={}, rules=rules_with(), audit=Audit(tmp_path / "a.jsonl"), local=local, fetch_photo=self.photo_fetcher(tmp_path))
        assert ["gallery" if o in (1, 2) else "iNat" for o in fetch.calls] == ["gallery", "iNat", "gallery", "iNat"]                  # 2025, 2019, 2025, 2019

    def test_a_themed_group_can_use_an_observation_the_gallery_does_not_have(self, tmp_path):
        g, index, local = self.world(tmp_path)
        taxonomy = {o: themes_entry(f"Amanita sp{o}") for o in (1, 2, 7, 8)}
        theme = themes.Theme("m", "Mushroom Monday", 0, ("MushroomMonday",), "", {"order": ["Agaricales"]}, {})

        def run(rules, sub):
            fetch = RecordingFetch({o: observation(o) for o in (1, 2, 7, 8)})
            ledger = Ledger(tmp_path / sub / "l.jsonl")
            social.draft_themes(index, ledger, taxonomy, [theme], today=dt.date(2026, 10, 26), days_ahead=1, gallery_dir=g, drafts_dir=tmp_path / sub / "d",
                                model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(), pick_model=lambda: "vis:1",
                                style_source=lambda: None, prints={}, rules=rules, audit=Audit(tmp_path / sub / "a.jsonl"), local=local,
                                fetch_photo=self.photo_fetcher(tmp_path / sub))
            return [r["obs_ids"] for r in ledger.records() if r.get("theme")]

        assert [sorted(x) for x in run(rules_with(), "on")] == [[1, 2, 7, 8]]
        assert [sorted(x) for x in run(rules_with(lower_quality_photos=False), "off")] == [[1, 2]]


class TestWithdrawSmall:
    def drafts(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 4)
        sizes = {"2026-10-obs1": (500, 333), "2026-10-obs2": (2667, 4000), "2026-10-obs3": (4000, 2667), "2026-10-obs4": (500, 333)}
        for did, (w, h) in sizes.items():
            p = drafts / did / "draft.json"
            doc = json.loads(p.read_text(encoding="utf-8"))
            for img in doc["images"]:
                img["width"], img["height"] = w, h
            if did == "2026-10-obs3":                                                                         # one big and one small image
                doc["images"][1]["width"], doc["images"][1]["height"] = 400, 300
            p.write_text(json.dumps(doc), encoding="utf-8")
        social.approve(ledger, drafts, "2026-10-obs3")
        ledger.append("posted", "2026-10-obs4", obs_id=4, month="2026-10")                                      # posted long ago: never touched
        return ledger, drafts

    def test_pending_drafts_with_a_small_photo_are_withdrawn_with_the_reason_and_the_big_ones_stay(self, tmp_path):
        ledger, drafts = self.drafts(tmp_path)
        audit = Audit(tmp_path / "audit.jsonl")
        assert social.withdraw_small(ledger, drafts, min_edge=1000, audit=audit) == ["2026-10-obs1", "2026-10-obs3"]
        states = {k: v["state"] for k, v in ledger.drafts().items()}
        assert states == {"2026-10-obs1": "withdrawn", "2026-10-obs2": "drafted", "2026-10-obs3": "withdrawn", "2026-10-obs4": "posted"}
        reasons = {r["draft_id"]: r["reason"] for r in ledger.records() if r["event"] == "withdrawn"}
        assert reasons == {"2026-10-obs1": "photo(s) under 1000 px: 1.jpg is 500, 2.jpg is 500", "2026-10-obs3": "photo(s) under 1000 px: 2.jpg is 400"}
        assert [(x["event"], x["draft_id"]) for x in audit.records()] == [("withdrawn", "2026-10-obs1"), ("withdrawn", "2026-10-obs3")]
        assert social.withdraw_small(ledger, drafts, min_edge=1000, audit=audit) == []                              # nothing left to withdraw

    def test_a_withdrawn_draft_is_never_posted_and_its_observation_can_be_drafted_again(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        social.withdraw_small(ledger, drafts, min_edge=1000, audit=Audit(tmp_path / "audit.jsonl"))
        assert social.post_next(ledger, drafts, now=NOW, live=True, send=RecordingSend()) == {"status": "nothing approved", "month": "2026-10"}
        g = tmp_path / "gallery"
        index = {"files": {}}
        rel = "2025/2025-10/x1.jpg"
        jpeg(g / rel, size=(300, 200))
        index["files"][rel] = 1
        report = social.draft_month(index, ledger, today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "new", keep_pending=5, model="vis:1",
                                    fetch=RecordingFetch({1: observation(1)}), read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model,
                                    style_source=lambda: None, prints={}, rules=rules_with(), audit=Audit(tmp_path / "a2.jsonl"))
        assert report["drafted"] == ["2026-10-obs1"]

    def test_a_withdrawn_themed_draft_frees_its_day_and_its_observations_for_a_new_one(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 0)
        ledger.append("drafted", "2026-10-26-mushroom-monday", obs_id=1, obs_ids=[1, 2], month="2026-10", theme="mushroom-monday", for_date="2026-10-26")
        write_draft(drafts, "2026-10-26-mushroom-monday", 1, "2026-10")
        assert social.withdraw_small(ledger, drafts, min_edge=1000, audit=Audit(tmp_path / "audit.jsonl")) == ["2026-10-26-mushroom-monday"]
        g = tmp_path / "gallery"
        index = {"files": {}}
        for o in (1, 2, 3, 4):
            rel = f"2025/2025-10/x{o}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = o
        theme = themes.Theme("mushroom-monday", "Mushroom Monday", 0, ("MushroomMonday",), "", {"order": ["Agaricales"]}, {})
        report = social.draft_themes(index, ledger, {o: themes_entry(f"Amanita sp{o}") for o in (1, 2, 3, 4)}, [theme], today=dt.date(2026, 10, 22), days_ahead=7,
                                     gallery_dir=g, drafts_dir=tmp_path / "new", model="vis:1", fetch=RecordingFetch({o: observation(o) for o in (1, 2, 3, 4)}),
                                     read_label=labels, describe_fn=RecordingDescribe(), pick_model=lambda: "vis:1", style_source=lambda: None, prints={},
                                     rules=rules_with(), audit=Audit(tmp_path / "a2.jsonl"))
        assert report["drafted"] == ["2026-10-26-mushroom-monday"]


def themes_entry(species):
    lineage = [{"rank": "kingdom", "name": "Fungi", "common_name": None}, {"rank": "order", "name": "Agaricales", "common_name": None}]
    return {"taxon": {"id": 1, "name": species, "common_name": None, "rank": "species", "iconic": "Fungi", "lineage": lineage},
            "grade": "research", "observed_on": "2025-10-01", "geoprivacy": None, "fetched": 1.0}
