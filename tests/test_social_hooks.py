"""The hook framework: ordering, rule gating, the exception policy per stage, extension loading, and the five places the flow runs hooks."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest
from test_social import NOW, RecordingDescribe, RecordingFetch, RecordingSend, drafted, events, jpeg, labels, no_model, observation, rules_with
from test_social_themes import MON, MONDAY, World

from bio_automation import cli, social, social_hooks
from bio_automation.social import Ledger, SkipObservation, SocialError
from bio_automation.social_hooks import Verdict
from bio_automation.social_rules import Audit


def audit_at(tmp_path):
    return Audit(tmp_path / "audit.jsonl", run="r1", now=lambda: NOW)


def run(stage, ctx, tmp_path, rules=None):
    audit = audit_at(tmp_path)
    return social_hooks.run(stage, ctx, rules=rules or rules_with(), audit=audit), audit.records()


class Recorder:
    """A hook body that records the contexts it saw and answers with a fixed verdict (or None)."""

    def __init__(self, verdict=None):
        self.verdict, self.seen = verdict, []

    def __call__(self, ctx):
        self.seen.append(ctx)
        return self.verdict


def boom(ctx):
    raise RuntimeError("kaput")


class TestRegistry:
    def test_hooks_run_lower_priority_first_then_in_registration_order(self):
        for name, prio in (("c", 70), ("a", 10), ("b", 50), ("b2", 50)):
            social_hooks.register(name, "order", Recorder(), priority=prio)
        assert [h.name for h in social_hooks.hooks("order")] == ["a", "b", "b2", "c"]

    def test_a_second_registration_of_the_same_name_and_stage_replaces_the_first(self):
        first, second = Recorder(), Recorder()
        social_hooks.register("x", "draft", first)
        social_hooks.register("x", "draft", second)
        assert [h.fn for h in social_hooks.hooks("draft")] == [second]
        social_hooks.register("x", "order", first)
        assert [(h.stage, h.fn) for h in social_hooks.hooks("order")] == [("order", first)]      # twin: same name, other stage, is kept

    def test_an_unknown_stage_is_refused(self):
        with pytest.raises(ValueError, match="unknown hook stage 'later'"):
            social_hooks.register("x", "later", Recorder())

    def test_the_extension_list_is_the_agreed_one(self):
        assert social_hooks.EXTENSIONS == ("social_gates", "social_controls", "social_verify", "social_alerts", "social_prep",
                                           "social_selfcheck", "social_state", "social_taste", "social_engagement")


class TestRun:
    def test_priority_order_is_the_order_the_hooks_run_in(self, tmp_path):
        order = []
        for name, prio in (("late", 80), ("early", 5)):
            social_hooks.register(name, "maintenance", lambda ctx, n=name: order.append(n), priority=prio)
        results, _ = run("maintenance", {}, tmp_path)
        assert (order, [h.name for h, _ in results]) == (["early", "late"], ["early", "late"])

    def test_a_hook_whose_rule_is_off_is_skipped_and_with_the_rule_on_it_runs(self, tmp_path):
        rec = Recorder()
        social_hooks.register("gated", "order", rec, rule="skip_casual")
        results, _ = run("order", {"k": 1}, tmp_path, rules_with(skip_casual=False))
        assert (results, rec.seen) == ([], [])
        results, _ = run("order", {"k": 1}, tmp_path, rules_with())
        assert ([(h.name, v) for h, v in results], rec.seen) == ([("gated", Verdict("pass"))], [{"k": 1}])

    def test_a_hook_without_a_rule_always_runs(self, tmp_path):
        rec = Recorder()
        social_hooks.register("free", "order", rec)
        run("order", {}, tmp_path, rules_with(skip_casual=False, themes=False))
        assert len(rec.seen) == 1

    def test_a_non_pass_verdict_is_audited_with_name_stage_action_reason_and_the_draft_id(self, tmp_path):
        social_hooks.register("floor", "pre_send", Recorder(Verdict("withdraw", "too small")))
        _, recs = run("pre_send", {"draft_id": "2026-10-obs1", "obs_id": 1}, tmp_path)
        assert recs == [{"at": NOW.isoformat(timespec="seconds"), "run": "r1", "event": "hook", "name": "floor", "stage": "pre_send",
                         "action": "withdraw", "reason": "too small", "draft_id": "2026-10-obs1", "obs_id": 1}]

    def test_a_pass_verdict_writes_no_audit_record(self, tmp_path):
        social_hooks.register("quiet", "pre_send", Recorder(Verdict("pass")))
        social_hooks.register("none", "pre_send", Recorder(None))
        results, recs = run("pre_send", {"draft_id": "d"}, tmp_path)
        assert ([v.action for _, v in results], recs) == (["pass", "pass"], [])

    @pytest.mark.parametrize("stage", ["draft", "pre_send"])
    def test_draft_and_pre_send_stop_at_the_first_non_pass_verdict(self, tmp_path, stage):
        later = Recorder()
        social_hooks.register("a", stage, Recorder(Verdict("reject", "no")), priority=1)
        social_hooks.register("b", stage, later, priority=2)
        results, recs = run(stage, {"draft_id": "d"}, tmp_path)
        assert ([h.name for h, _ in results], later.seen, [r["name"] for r in recs]) == (["a"], [], ["a"])

    @pytest.mark.parametrize("stage", ["order", "post_send", "maintenance"])
    def test_the_other_stages_run_every_hook_even_after_a_non_pass_verdict(self, tmp_path, stage):
        later = Recorder()
        social_hooks.register("a", stage, Recorder(Verdict("reject", "no")), priority=1)
        social_hooks.register("b", stage, later, priority=2)
        results, recs = run(stage, {"draft_id": "d"}, tmp_path)
        assert ([h.name for h, _ in results], len(later.seen), [r["name"] for r in recs]) == (["a", "b"], 1, ["a"])

    def test_first_non_pass_returns_the_first_one_and_none_when_all_pass(self):
        a, b, c = (social_hooks.Hook(n, "draft", None, Recorder(), 50) for n in "abc")
        assert social_hooks.first_non_pass([(a, Verdict("pass")), (b, Verdict("reject", "r1")), (c, Verdict("block", "r2"))]) == (b, Verdict("reject", "r1"))
        assert social_hooks.first_non_pass([(a, Verdict("pass"))]) is None

    @pytest.mark.parametrize("stage", ["draft", "order", "post_send", "maintenance"])
    def test_an_exception_fails_open_in_every_stage_but_pre_send_and_is_audited(self, tmp_path, stage):
        social_hooks.register("broken", stage, boom, priority=1)
        after = Recorder()
        social_hooks.register("after", stage, after, priority=2)
        results, recs = run(stage, {"obs_id": 7}, tmp_path)
        assert [(h.name, v.action) for h, v in results] == [("broken", "pass"), ("after", "pass")]
        assert len(after.seen) == 1
        assert recs == [{"at": NOW.isoformat(timespec="seconds"), "run": "r1", "event": "hook_error", "name": "broken", "stage": stage,
                         "error": "RuntimeError: kaput", "outcome": "pass", "obs_id": 7}]

    def test_an_exception_in_pre_send_fails_closed_as_a_block_and_stops_the_stage(self, tmp_path):
        social_hooks.register("broken", "pre_send", boom, priority=1)
        after = Recorder()
        social_hooks.register("after", "pre_send", after, priority=2)
        results, recs = run("pre_send", {"draft_id": "d1"}, tmp_path)
        assert [(h.name, v.action) for h, v in results] == [("broken", "block")]
        assert after.seen == []
        assert [(r["event"], r["outcome"] if r["event"] == "hook_error" else r["action"]) for r in recs] == [("hook_error", "block"), ("hook", "block")]
        assert recs[0]["error"] == "RuntimeError: kaput" and recs[0]["draft_id"] == "d1"

    def test_an_unknown_verdict_action_is_an_error_not_a_pass_through(self, tmp_path):
        social_hooks.register("odd", "pre_send", Recorder(Verdict("allow", "x")))
        results, recs = run("pre_send", {"draft_id": "d"}, tmp_path)
        assert [v.action for _, v in results] == ["block"]
        assert recs[0]["error"] == "ValueError: unknown verdict action 'allow' (actions: pass, reject, withdraw, block)"

    def test_a_hook_naming_an_unknown_rule_is_an_error_handled_per_stage(self, tmp_path):
        social_hooks.register("typo", "pre_send", Recorder(), rule="no_such_rule")
        social_hooks.register("typo", "order", Recorder(), rule="no_such_rule")
        assert [v.action for _, v in run("pre_send", {"draft_id": "d"}, tmp_path)[0]] == ["block"]
        assert [v.action for _, v in run("order", {}, tmp_path)[0]] == ["pass"]

    def test_an_unwritable_audit_log_does_not_change_the_verdict(self, tmp_path):
        class Unwritable:
            def log(self, event, **fields):
                raise OSError("disk full")

        social_hooks.register("floor", "pre_send", Recorder(Verdict("withdraw", "small")))
        results = social_hooks.run("pre_send", {"draft_id": "d"}, rules=rules_with(), audit=Unwritable())
        assert [v for _, v in results] == [Verdict("withdraw", "small")]


@pytest.fixture
def fakepkg(tmp_path, monkeypatch):
    """A package `fakeext` on sys.path whose modules stand in for the extensions; social_hooks is pointed at it."""
    pkg = tmp_path / "pylib" / "fakeext"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "pylib"))
    monkeypatch.setattr(social_hooks, "PACKAGE", "fakeext")
    yield pkg
    for name in [n for n in sys.modules if n == "fakeext" or n.startswith("fakeext.")]:
        del sys.modules[name]


class TestExtensions:
    def test_a_missing_extension_module_is_tolerated_and_the_present_ones_load(self, fakepkg, monkeypatch):
        (fakepkg / "ext_one.py").write_text("X = 1\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_one", "ext_absent"))
        assert [m.__name__ for m in social_hooks.extensions()] == ["fakeext.ext_one"]

    def test_register_hooks_is_called_once_even_when_load_all_runs_twice(self, fakepkg, monkeypatch):
        (fakepkg / "ext_one.py").write_text(
            "from bio_automation import social_hooks\nCALLS = []\n"
            "def register_hooks():\n    CALLS.append(1)\n    social_hooks.register('one', 'order', lambda ctx: None)\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_one",))
        social_hooks.load_all()
        social_hooks.load_all()
        assert [h.name for h in social_hooks.hooks("order")] == ["one"]
        assert sys.modules["fakeext.ext_one"].CALLS == [1]

    def test_reset_forgets_hooks_and_lets_the_extension_register_again(self, fakepkg, monkeypatch):
        (fakepkg / "ext_one.py").write_text(
            "from bio_automation import social_hooks\ndef register_hooks():\n    social_hooks.register('one', 'order', lambda ctx: None)\n",
            encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_one",))
        social_hooks.load_all()
        social_hooks.reset()
        assert social_hooks.hooks("order") == []
        social_hooks.load_all()
        assert [h.name for h in social_hooks.hooks("order")] == ["one"]

    def test_an_extension_without_register_hooks_loads_fine(self, fakepkg, monkeypatch):
        (fakepkg / "ext_plain.py").write_text("COMMANDS = {}\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_plain",))
        assert [m.__name__ for m in social_hooks.load_all()] == ["fakeext.ext_plain"]

    def test_a_broken_extension_propagates_its_error(self, fakepkg, monkeypatch):
        (fakepkg / "ext_bad.py").write_text("raise SyntaxError('bad extension')\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_bad",))
        with pytest.raises(SyntaxError, match="bad extension"):
            social_hooks.extensions()

    def test_a_missing_dependency_inside_an_extension_propagates_not_tolerated(self, fakepkg, monkeypatch):
        (fakepkg / "ext_dep.py").write_text("import not_installed_anywhere_xyz\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_dep",))
        with pytest.raises(ModuleNotFoundError, match="not_installed_anywhere_xyz"):
            social_hooks.extensions()

    def test_a_failing_register_hooks_propagates_and_is_retried_by_the_next_load(self, fakepkg, monkeypatch):
        (fakepkg / "ext_r.py").write_text("def register_hooks():\n    raise RuntimeError('cannot register')\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_r",))
        for _ in range(2):
            with pytest.raises(RuntimeError, match="cannot register"):
                social_hooks.load_all()

    def test_the_cli_adds_extension_parsers_and_dispatches_extension_commands(self, fakepkg, monkeypatch, capsys):
        (fakepkg / "ext_cmd.py").write_text(
            "class ExtError(Exception):\n    pass\n"
            "def add_parsers(sub):\n    p = sub.add_parser('ext-hello')\n    p.add_argument('--bad', action='store_true')\n"
            "def hello(args):\n    if args.bad:\n        raise ExtError('it went wrong')\n    print('hello from the extension')\n    return 0\n"
            "COMMANDS = {'ext-hello': hello}\nERRORS = (ExtError,)\n", encoding="utf-8")
        monkeypatch.setattr(social_hooks, "EXTENSIONS", ("ext_cmd",))
        monkeypatch.setattr("sys.argv", ["bio", "ext-hello"])
        assert cli.main() == 0
        assert capsys.readouterr().out == "hello from the extension\n"
        monkeypatch.setattr("sys.argv", ["bio", "ext-hello", "--bad"])
        assert cli.main() == 1
        assert capsys.readouterr().err == "bio ext-hello: it went wrong\n"


class TestDraftStage:
    def setup(self, tmp_path):
        g = tmp_path / "gallery"
        rels = ["2024/2024-10/2024-10-01_DSC_0.jpg", "2024/2024-10/2024-10-01_DSC_1.jpg"]
        for r in rels:
            jpeg(g / r, size=(300, 200))
        return g, rels

    def make(self, tmp_path, rules=None):
        g, rels = self.setup(tmp_path)
        return social.make_draft(42, rels, month="2026-10", gallery_dir=g, drafts_dir=tmp_path / "drafts", model="vis:1",
                                 fetch=RecordingFetch({42: observation()}), read_label=labels, describe_fn=RecordingDescribe(),
                                 now=NOW, rules=rules or rules_with(), audit=audit_at(tmp_path))

    def test_a_reject_skips_the_observation_and_leaves_no_folder(self, tmp_path):
        social_hooks.register("alt-check", "draft", lambda ctx: Verdict("reject", f"alt repeats a private place: {ctx['alt']}"))
        with pytest.raises(SkipObservation, match="observation 42: alt repeats a private place: alt for title 2024-10-01_DSC_0"):
            self.make(tmp_path)
        assert list((tmp_path / "drafts").iterdir()) == []

    def test_a_pass_writes_the_draft_and_the_hook_saw_image_alt_label_and_month(self, tmp_path):
        rec = Recorder(Verdict("pass"))
        social_hooks.register("seen", "draft", rec)
        draft = self.make(tmp_path)
        assert draft["draft_id"] == "2026-10-obs42"
        assert (tmp_path / "drafts" / "2026-10-obs42" / "draft.json").is_file()
        first = rec.seen[0]
        assert (first["obs_id"], first["image"], first["alt"], first["label"]["title"], first["month"]) == (
            42, tmp_path / "drafts" / "2026-10-obs42" / "1.jpg", "alt for title 2024-10-01_DSC_0", "title 2024-10-01_DSC_0", "2026-10")
        assert first["obs"]["observation_id"] == 42 and len(rec.seen) == 2

    def test_a_reject_on_the_second_image_removes_the_whole_folder(self, tmp_path):
        social_hooks.register("second", "draft", lambda ctx: Verdict("reject", "bad second") if ctx["image"].name == "2.jpg" else None)
        with pytest.raises(SkipObservation, match="observation 42: bad second"):
            self.make(tmp_path)
        assert list((tmp_path / "drafts").iterdir()) == []

    def test_a_reject_by_a_hook_whose_rule_is_off_does_not_happen(self, tmp_path):
        social_hooks.register("alt-check", "draft", lambda ctx: Verdict("reject", "x"), rule="skip_casual")
        assert self.make(tmp_path, rules_with(skip_casual=False))["draft_id"] == "2026-10-obs42"

    def test_draft_month_reports_the_hook_reject_as_a_skip_and_audits_it(self, tmp_path):
        g, rels = self.setup(tmp_path)
        index = {"files": {rels[0]: 1, rels[1]: 2}}
        social_hooks.register("no-obs-1", "draft", lambda ctx: Verdict("reject", "private place") if ctx["obs_id"] == 1 else None)
        audit = audit_at(tmp_path)
        report = social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "drafts",
                                    keep_pending=5, model="vis:1", fetch=RecordingFetch({1: observation(1), 2: observation(2)}), read_label=labels,
                                    describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None, prints={},
                                    rules=rules_with(), audit=audit)
        assert (report["drafted"], report["skipped"]) == (["2026-10-obs2"], {1: "observation 1: private place"})
        assert [(r["event"], r.get("name")) for r in audit.records() if r["event"] in ("hook", "draft_skipped")] == [("hook", "no-obs-1"), ("draft_skipped", None)]


class TestOrderStage:
    def test_single_candidates_are_reordered_in_place_and_drafted_in_that_order(self, tmp_path):
        g = tmp_path / "gallery"
        index = {"files": {}}
        for o in (1, 2, 3):
            rel = f"2024/2024-10/2024-10-0{o}_DSC_{o}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = o
        seen = []

        def put_three_first(ctx):
            seen.append((ctx["kind"], ctx["taxonomy"], ctx["today"], sorted(ctx["candidates"])))
            ctx["candidates"].sort(key=lambda o: o != 3)

        social_hooks.register("three-first", "order", put_three_first)
        report = social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "d",
                                    keep_pending=1, model="vis:1", fetch=RecordingFetch({o: observation(o) for o in (1, 2, 3)}), read_label=labels,
                                    describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None, prints={},
                                    rules=rules_with(), audit=audit_at(tmp_path))
        assert report["drafted"] == ["2026-10-obs3"]
        assert seen == [("single", None, dt.date(2026, 10, 21), [1, 2, 3])]

    def test_without_the_hook_the_first_candidate_is_not_three(self, tmp_path):
        g = tmp_path / "gallery"
        index = {"files": {}}
        for o in (1, 2, 3):
            rel = f"2024/2024-10/2024-10-0{o}_DSC_{o}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = o
        report = social.draft_month(index, Ledger(tmp_path / "l.jsonl"), today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "d",
                                    keep_pending=1, model="vis:1", fetch=RecordingFetch({o: observation(o) for o in (1, 2, 3)}), read_label=labels,
                                    describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None, prints={},
                                    rules=rules_with(), audit=audit_at(tmp_path))
        assert report["drafted"] != ["2026-10-obs3"]

    def draft_theme_day(self, tmp_path):
        w = World(tmp_path, n=8)
        return w, w.draft([MONDAY], MON, 1, rules=rules_with(), audit=audit_at(tmp_path), pick_model=lambda: "vis:1")

    def test_theme_candidates_are_reordered_and_the_hook_gets_kind_and_taxonomy(self, tmp_path):
        seen = []

        def reverse(ctx):
            seen.append((ctx["kind"], sorted(ctx["taxonomy"]), ctx["today"], list(ctx["candidates"])))
            ctx["candidates"].reverse()

        social_hooks.register("reverse", "order", reverse)
        w, report = self.draft_theme_day(tmp_path)
        got = json.loads((w.drafts / report["drafted"][0] / "draft.json").read_text(encoding="utf-8"))["obs_ids"]
        assert len(seen) == 1 and seen[0][:3] == ("theme:mushroom-monday", list(range(1, 9)), MON)
        assert len(seen[0][3]) == 7 and got == seen[0][3][::-1][:4]

    def test_without_the_hook_the_group_keeps_the_picked_order(self, tmp_path):
        seen = []
        social_hooks.register("look", "order", lambda ctx: seen.append(list(ctx["candidates"])))
        w, report = self.draft_theme_day(tmp_path)
        assert json.loads((w.drafts / report["drafted"][0] / "draft.json").read_text(encoding="utf-8"))["obs_ids"] == seen[0][:4]

    def test_a_hook_cannot_add_candidates_to_a_theme_group(self, tmp_path):
        social_hooks.register("add", "order", lambda ctx: ctx["candidates"].insert(0, 99))
        w, report = self.draft_theme_day(tmp_path)
        ids = json.loads((w.drafts / report["drafted"][0] / "draft.json").read_text(encoding="utf-8"))["obs_ids"]
        assert 99 not in ids and set(ids) <= set(range(1, 9))

    def test_a_hook_that_empties_the_group_makes_the_day_short(self, tmp_path):
        social_hooks.register("none", "order", lambda ctx: ctx["candidates"].clear())
        _, report = self.draft_theme_day(tmp_path)
        assert (report["drafted"], list(report["short"])) == ([], [MON.isoformat()])


def post(ledger, drafts, tmp_path, *, live=True, send=None, **kw):
    return social.post_next(ledger, drafts, now=NOW, live=live, send=send or RecordingSend(), rules=kw.pop("rules", rules_with()),
                            audit=audit_at(tmp_path), **kw)


class TestPreSendStage:
    def test_a_withdraw_records_it_and_the_next_draft_is_posted_in_the_same_run(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 3)
        social_hooks.register("floor", "pre_send", lambda ctx: Verdict("withdraw", "photo too small") if ctx["draft_id"] == "2026-10-obs1" else None)
        send = RecordingSend()
        result = post(ledger, drafts, tmp_path, send=send)
        assert (result["status"], len(send.calls)) == ("posted", 1)
        assert events(ledger, "2026-10-obs1") == ["drafted", "withdrawn"]
        assert [r for r in ledger.records() if r["event"] == "withdrawn"][0]["reason"] == "photo too small"
        assert events(ledger, "2026-10-obs2") == ["drafted", "approved", "posting", "posted"]
        assert events(ledger, "2026-10-obs3") == ["drafted"]
        recs = Audit(tmp_path / "audit.jsonl").records()
        assert [r["draft_id"] for r in recs if r["event"] == "withdrawn"] == ["2026-10-obs1"]
        assert [r["draft_id"] for r in recs if r["event"] == "post_decision"] == ["2026-10-obs2"]

    def test_with_no_withdrawing_hook_the_first_draft_posts(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 3)
        post(ledger, drafts, tmp_path)
        assert events(ledger, "2026-10-obs1") == ["drafted", "approved", "posting", "posted"]

    def test_reject_is_treated_like_withdraw(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2)
        social_hooks.register("r", "pre_send", lambda ctx: Verdict("reject", "no") if ctx["draft_id"] == "2026-10-obs1" else None)
        assert post(ledger, drafts, tmp_path)["status"] == "posted"
        assert events(ledger, "2026-10-obs1") == ["drafted", "withdrawn"]

    def test_at_most_three_drafts_are_taken_back_in_one_run(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 5)
        social_hooks.register("all", "pre_send", lambda ctx: Verdict("withdraw", "nope"))
        send = RecordingSend()
        result = post(ledger, drafts, tmp_path, send=send)
        assert (result, send.calls) == ({"status": "withdrawn", "withdrawn": ["2026-10-obs1", "2026-10-obs2", "2026-10-obs3"]}, [])
        assert [d for d, v in ledger.drafts().items() if v["state"] == "withdrawn"] == ["2026-10-obs1", "2026-10-obs2", "2026-10-obs3"]

    def test_a_block_sends_nothing_leaves_the_draft_untouched_and_audits_post_skipped(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2)
        social_hooks.register("pause", "pre_send", lambda ctx: Verdict("block", "paused by the owner"))
        send = RecordingSend()
        result = post(ledger, drafts, tmp_path, send=send)
        assert (result, send.calls) == ({"status": "blocked", "reason": "paused by the owner", "draft_id": "2026-10-obs1"}, [])
        assert [r["event"] for r in ledger.records()] == ["drafted", "drafted"]
        skipped = [r for r in Audit(tmp_path / "audit.jsonl").records() if r["event"] == "post_skipped"]
        assert [(r["status"], r["hook"], r["reason"]) for r in skipped] == [("blocked", "pause", "paused by the owner")]

    def test_a_crashing_pre_send_hook_blocks_the_post(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        social_hooks.register("broken", "pre_send", boom)
        send = RecordingSend()
        result = post(ledger, drafts, tmp_path, send=send)
        assert (result["status"], send.calls) == ("blocked", [])
        assert "RuntimeError: kaput" in result["reason"]

    def test_a_dry_run_runs_pre_send_hooks_but_writes_no_ledger_record(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 3)
        before = ledger.records()
        social_hooks.register("floor", "pre_send", lambda ctx: Verdict("withdraw", "small") if ctx["draft_id"] == "2026-10-obs1" else None)
        result = post(ledger, drafts, tmp_path, live=False)
        assert (result["status"], result["draft_id"]) == ("dry-run", "2026-10-obs2")
        assert ledger.records() == before
        social_hooks.reset()
        social_hooks.register("pause", "pre_send", lambda ctx: Verdict("block", "paused"))
        assert post(ledger, drafts, tmp_path, live=False)["status"] == "blocked"
        assert ledger.records() == before

    def test_the_hook_sees_the_documented_context(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        rec = Recorder()
        social_hooks.register("seen", "pre_send", rec)
        post(ledger, drafts, tmp_path, manual=True)
        ctx = rec.seen[0]
        assert (ctx["draft_id"], ctx["draft"]["obs_id"], ctx["draft_dir"], ctx["now"], ctx["live"], ctx["manual"], ctx["ledger"]) == (
            "2026-10-obs1", 1, drafts / "2026-10-obs1", NOW, True, True, ledger)
        assert ctx["fetch"] is social._fetch and ctx["rules"].on("daily_cap")

    def test_an_owner_chosen_draft_that_is_withdrawn_is_not_replaced_by_another(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 2)
        social_hooks.register("floor", "pre_send", lambda ctx: Verdict("withdraw", "small"))
        send = RecordingSend()
        result = post(ledger, drafts, tmp_path, send=send, draft_id="2026-10-obs2")
        assert (result, send.calls) == ({"status": "withdrawn", "draft_id": "2026-10-obs2", "reason": "small"}, [])
        assert events(ledger, "2026-10-obs1") == ["drafted"]

    def test_a_withdrawal_with_nothing_else_eligible_reports_what_was_withdrawn(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        social_hooks.register("floor", "pre_send", lambda ctx: Verdict("withdraw", "small"))
        result = post(ledger, drafts, tmp_path)
        assert result == {"status": "nothing approved", "month": "2026-10", "withdrawn": ["2026-10-obs1"]}


class TestPostSendStage:
    def test_post_send_runs_after_the_posted_record_with_uri_cid_and_request(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        seen = []

        def look(ctx):
            seen.append((ctx["draft_id"], ctx["uri"], ctx["cid"], ctx["request"]["langs"], len(ctx["request"]["images"]),
                         events(ctx["ledger"], ctx["draft_id"]), ctx["manual"]))

        social_hooks.register("look", "post_send", look)
        result = post(ledger, drafts, tmp_path)
        assert seen == [("2026-10-obs1", result["uri"], "bafyabc", ["en"], 2, ["drafted", "approved", "posting", "posted"], False)]

    def test_post_send_does_not_run_when_nothing_was_sent(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        rec = Recorder()
        social_hooks.register("look", "post_send", rec)
        post(ledger, drafts, tmp_path, live=False)
        assert rec.seen == []

    def test_a_raising_post_send_hook_never_breaks_the_post_and_is_audited(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        social_hooks.register("broken", "post_send", boom)
        result = post(ledger, drafts, tmp_path)
        assert result["status"] == "posted"
        assert events(ledger, "2026-10-obs1")[-1] == "posted"
        errs = [r for r in Audit(tmp_path / "audit.jsonl").records() if r["event"] == "hook_error"]
        assert [(r["name"], r["stage"], r["draft_id"]) for r in errs] == [("broken", "post_send", "2026-10-obs1")]


class TestMaintenanceStage:
    def test_social_draft_runs_the_maintenance_stage_with_drafts_dir_ledger_rules_and_today(self, tmp_path, monkeypatch, capsys):
        from bio_automation import settings

        monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
        monkeypatch.setattr(settings, "SOCIAL_LEDGER", tmp_path / "ledger.jsonl")
        monkeypatch.setattr(settings, "INAT_SNAPSHOT", tmp_path / "none.json")
        index = tmp_path / "index.json"
        index.write_text(json.dumps({"files": {}}), encoding="utf-8")
        seen = []
        social_hooks.register("m", "maintenance", lambda ctx: seen.append((ctx["drafts_dir"], isinstance(ctx["ledger"], Ledger), ctx["today"],
                                                                         ctx["rules"].on("themes"))))
        monkeypatch.setattr("sys.argv", ["bio", "social-draft", "--index", str(index), "--date", "2026-10-21", "--no-themes", "--keep-pending", "0"])
        assert cli.main() == 0
        capsys.readouterr()
        assert seen == [(tmp_path / "social" / "drafts", True, dt.date(2026, 10, 21), True)]
