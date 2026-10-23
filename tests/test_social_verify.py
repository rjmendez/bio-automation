"""Post verification: the AppView read-back after a live post, every mismatch kind, the retry policy, and the social-verify command."""
from __future__ import annotations

import copy
import json
import urllib.parse

import pytest
from test_social import NOW, URI, RecordingSend, drafted, observation, rules_with

from bio_automation import cli, settings, social, social_hooks, social_verify
from bio_automation.social import Ledger, segments_text
from bio_automation.social_rules import Audit

DID = "did:plc:me"
SEGMENTS = social.text_segments(observation(42))
TEXT = segments_text(SEGMENTS)
ALTS = ["first alt", "second alt"]


def appview_post(text=TEXT, alts=ALTS, segments=SEGMENTS, author=DID):
    """What the AppView answers for a perfect post: facets with byte ranges over each link and tag segment."""
    facets, offset = [], 0
    for seg in segments:
        size = len(seg[1].encode("utf-8"))
        if seg[0] in ("link", "tag"):
            feature = {"$type": "app.bsky.richtext.facet#link", "uri": seg[2]} if seg[0] == "link" else {"$type": "app.bsky.richtext.facet#tag", "tag": seg[2]}
            facets.append({"index": {"byteStart": offset, "byteEnd": offset + size}, "features": [feature]})
        offset += size
    return {"posts": [{"uri": URI, "author": {"did": author},
                       "record": {"text": text, "facets": facets, "embed": {"images": [{"alt": a} for a in alts]}}}]}


class RecordingGet:
    """Stands in for the network: records each URL and answers with the next scripted answer (an Exception instance is raised)."""

    def __init__(self, *answers):
        self.answers, self.urls = list(answers), []

    def __call__(self, url):
        self.urls.append(url)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


class RecordingSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


def verify(get, sleep=None, **kw):
    return social_verify.verify_post(URI, text=TEXT, alts=ALTS, segments=SEGMENTS, get=get, sleep=sleep or RecordingSleep(), **kw)


def tag_of(segments):
    return next(s for s in segments if s[0] == "tag")


class TestCompare:
    def test_a_perfect_match_passes_every_check_on_the_first_attempt(self):
        get = RecordingGet(appview_post())

        result = verify(get)

        assert result == {"ok": True, "checks": ["author", "text", "image_count", "alts", "facets"], "problems": [], "attempts": 1, "unverified": False}
        assert get.urls == [social_verify.APPVIEW_URL + "?uris=" + urllib.parse.quote(URI, safe="")]

    def test_a_missing_record_is_reported_after_all_the_attempts(self):
        result = verify(RecordingGet({"posts": []}))

        assert result == {"ok": False, "checks": [], "problems": ["the AppView does not have the post after 10 attempts"],
                          "attempts": 10, "unverified": True}

    def test_different_text_is_reported_with_both_texts(self):
        result = verify(RecordingGet(appview_post(text=TEXT + "!")))

        assert result["ok"] is False
        assert result["problems"] == [f"text differs: sent {TEXT!r}, read back {TEXT + '!'!r}"]
        assert result["checks"] == ["author", "image_count", "alts", "facets"]

    def test_a_different_image_count_is_reported(self):
        result = verify(RecordingGet(appview_post(alts=["first alt"])))

        assert result["problems"] == ["image count differs: sent 2, read back 1"]
        assert result["checks"] == ["author", "text", "facets"]

    def test_one_different_alt_is_reported_by_its_position_and_the_other_passes(self):
        result = verify(RecordingGet(appview_post(alts=["first alt", "changed"])))

        assert result["problems"] == ["image 2 alt differs: sent 'second alt', read back 'changed'"]
        assert result["checks"] == ["author", "text", "image_count", "facets"]

    def test_swapped_alts_fail_because_the_order_matters(self):
        result = verify(RecordingGet(appview_post(alts=["second alt", "first alt"])))

        assert result["problems"] == ["image 1 alt differs: sent 'first alt', read back 'second alt'",
                                      "image 2 alt differs: sent 'second alt', read back 'first alt'"]

    def test_a_missing_tag_facet_is_reported_and_the_other_facets_pass(self):
        tag = tag_of(SEGMENTS)[2]
        post = appview_post()
        post["posts"][0]["record"]["facets"] = [f for f in post["posts"][0]["record"]["facets"] if f["features"][0].get("tag") != tag]

        result = verify(RecordingGet(post))

        assert result["problems"] == [f"missing tag facet {tag}"]
        assert "facets" not in result["checks"]

    def test_a_missing_link_facet_is_reported(self):
        url = social.INAT_OBS_URL.format(42)
        post = appview_post()
        post["posts"][0]["record"]["facets"] = [f for f in post["posts"][0]["record"]["facets"] if f["features"][0].get("uri") != url]

        assert verify(RecordingGet(post))["problems"] == [f"missing link facet {url}"]

    def test_a_facet_over_the_wrong_bytes_is_reported(self):
        tag = tag_of(SEGMENTS)
        post = appview_post()
        for f in post["posts"][0]["record"]["facets"]:
            if f["features"][0].get("tag") == tag[2]:
                f["index"]["byteStart"] -= 1

        assert verify(RecordingGet(post))["problems"] == [f"tag facet {tag[2]} covers {chr(10) + tag[1]!r} instead of {tag[1]!r}"]

    def test_an_author_that_is_not_the_uri_did_is_reported(self):
        result = verify(RecordingGet(appview_post(author="did:plc:other")))

        assert result["problems"] == ["author is 'did:plc:other', the URI says 'did:plc:me'"]

    def test_every_problem_is_listed_together(self):
        result = verify(RecordingGet(appview_post(text=TEXT + "!", alts=[])))

        assert result["problems"] == [f"text differs: sent {TEXT!r}, read back {TEXT + '!'!r}", "image count differs: sent 2, read back 0"]


class TestRetries:
    def test_not_indexed_yet_then_found_sleeps_two_then_three_seconds(self):
        get, sleep = RecordingGet({"posts": []}, {"posts": []}, appview_post()), RecordingSleep()

        result = verify(get, sleep)

        assert result["ok"] is True and result["attempts"] == 3
        assert len(get.urls) == 3
        assert sleep.calls == [2, 3]

    def test_found_at_once_never_sleeps(self):
        sleep = RecordingSleep()

        verify(RecordingGet(appview_post()), sleep)

        assert sleep.calls == []

    def test_retries_are_exhausted_after_ten_attempts_with_growing_sleeps_of_54_seconds(self):
        get, sleep = RecordingGet({"posts": []}), RecordingSleep()

        result = verify(get, sleep)

        assert result["ok"] is False and result["attempts"] == 10
        assert len(get.urls) == 10
        assert sleep.calls == [2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert sum(sleep.calls) == 54

    def test_a_network_error_is_retried_and_then_succeeds(self):
        get = RecordingGet(OSError("connection reset"), appview_post())

        result = verify(get)

        assert result["ok"] is True and result["attempts"] == 2

    def test_a_network_error_that_never_clears_is_a_problem_naming_the_error(self):
        result = verify(RecordingGet(OSError("connection reset")), attempts=2)

        assert result == {"ok": False, "checks": [], "problems": ["the AppView call failed (OSError: connection reset) after 2 attempts"],
                          "attempts": 2, "unverified": True}

    def test_content_mismatch_is_not_retried(self):
        get = RecordingGet(appview_post(text="x"))

        verify(get)

        assert len(get.urls) == 1


class TestHook:
    def hook_ctx(self, tmp_path):
        request = {"text": TEXT, "langs": ["en"], "images": [{"path": "a", "alt": a, "width": 1, "height": 1} for a in ALTS]}
        return {"draft_id": "d1", "draft": {"text": SEGMENTS}, "uri": URI, "cid": "c", "request": request, "ledger": None, "now": NOW,
                "manual": False, "rules": rules_with()}

    def run_hook(self, tmp_path, get, rules=None):
        audit = Audit(tmp_path / "audit.jsonl", run="r1", now=lambda: NOW)
        hook = social_verify.make_hook(get=get, sleep=RecordingSleep(), audit=audit)
        social_hooks.register("verify_posts", "post_send", hook, rule="verify_posts")
        ctx = self.hook_ctx(tmp_path)
        social_hooks.run("post_send", ctx, rules=rules or rules_with(), audit=audit)
        return [r for r in audit.records() if r["event"].startswith("post_verif") or r["event"] == "hook_error"]

    def test_a_good_post_is_audited_as_verified_with_its_checks(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet(appview_post()))

        assert records == [{"at": NOW.isoformat(timespec="seconds"), "run": "r1", "event": "post_verified", "draft_id": "d1", "uri": URI,
                            "checks": ["author", "text", "image_count", "alts", "facets"], "attempts": 1}]

    def test_a_bad_post_is_audited_with_its_problems(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet(appview_post(alts=["first alt"])))

        assert len(records) == 1
        assert records[0]["event"] == "post_verify_failed"
        assert records[0]["problems"] == ["image count differs: sent 2, read back 1"]

    def test_a_dead_network_raises_nothing_and_the_audit_says_why(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet(OSError("no route to host")))

        assert len(records) == 1                                  # no hook_error: the hook handled it itself
        assert records[0]["event"] == "post_verify_failed"
        assert records[0]["problems"] == ["the AppView call failed (OSError: no route to host) after 10 attempts"]

    def test_a_garbled_answer_raises_nothing_and_the_audit_says_why(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet({"posts": [{"record": "not a dict", "author": None}]}))

        assert records[0]["event"] == "post_verify_failed"
        assert records[0]["problems"][0].startswith("the AppView answer could not be read (AttributeError")

    def test_with_the_rule_off_nothing_is_read_or_audited(self, tmp_path):
        get = RecordingGet(appview_post())

        records = self.run_hook(tmp_path, get, rules=rules_with(verify_posts=False))

        assert records == []
        assert get.urls == []

    def test_the_default_registration_is_a_post_send_hook_of_rule_verify_posts_at_priority_10(self):
        social_verify.register_hooks()

        (hook,) = social_hooks.hooks("post_send")

        assert (hook.name, hook.rule, hook.priority) == ("verify_posts", "verify_posts", 10)

    def test_a_live_post_through_post_next_is_verified_after_it_is_sent(self, tmp_path):
        ledger, drafts = drafted(tmp_path, 1)
        sent = RecordingSend()
        audit = Audit(tmp_path / "audit.jsonl", run="r1", now=lambda: NOW)
        draft = social.load_draft(drafts / "2026-10-obs1")
        get = RecordingGet(appview_post(text=segments_text(draft["text"]), segments=draft["text"], alts=["first alt", "second alt"]))
        social_hooks.register("verify_posts", "post_send", social_verify.make_hook(get=get, sleep=RecordingSleep(), audit=audit), rule="verify_posts")

        result = social.post_next(ledger, drafts, now=NOW, live=True, send=sent, rules=rules_with(), audit=audit)

        assert result["status"] == "posted"
        events = [r["event"] for r in audit.records()]
        assert events.index("posted") > events.index("post_verified")
        assert get.urls == [social_verify.posts_url(URI)]


class TestCommand:
    def world(self, tmp_path, monkeypatch, *, posted=True):
        ledger, drafts = drafted(tmp_path, 1)
        monkeypatch.setattr(settings, "SOCIAL_LEDGER", ledger.path)
        monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path)
        if posted:
            ledger.append("posted", "2026-10-obs1", obs_id=1, month="2026-10", uri=URI, cid="c", now=NOW)
        return social.load_draft(drafts / "2026-10-obs1")

    def args(self, **kw):
        import argparse

        return argparse.Namespace(**{"uri": None, "draft": None, "last": False, **kw})

    def test_last_prints_ok_and_returns_zero_for_a_matching_post(self, tmp_path, monkeypatch, capsys):
        draft = self.world(tmp_path, monkeypatch)
        monkeypatch.setattr(social_verify, "http_get", RecordingGet(appview_post(text=segments_text(draft["text"]), segments=draft["text"])))

        code = social_verify.cmd_social_verify(self.args(last=True))

        assert code == 0
        assert capsys.readouterr().out == f"OK   2026-10-obs1 {URI}\n  passed: author, text, image_count, alts, facets\n"

    def test_a_mismatching_post_prints_fail_and_the_problem_and_returns_one(self, tmp_path, monkeypatch, capsys):
        draft = self.world(tmp_path, monkeypatch)
        monkeypatch.setattr(social_verify, "http_get", RecordingGet(appview_post(text="other", segments=draft["text"])))

        code = social_verify.cmd_social_verify(self.args(uri=URI))

        out = capsys.readouterr().out
        assert code == 1
        assert out.splitlines()[0] == f"FAIL 2026-10-obs1 {URI}"
        assert out.splitlines()[2] == f"  problem: text differs: sent {segments_text(draft['text'])!r}, read back 'other'"

    def test_a_post_sent_after_the_legacy_review_claim_was_stripped_verifies(self, tmp_path, monkeypatch, capsys):
        draft = self.world(tmp_path, monkeypatch)
        claim = social.REVIEW_CLAIM
        legacy = copy.deepcopy(draft["text"])
        legacy[0][1] = legacy[0][1] + claim.strip("\n") + "\n"
        path = tmp_path / "drafts" / "2026-10-obs1" / "draft.json"
        path.write_text(json.dumps({**draft, "text": legacy}), encoding="utf-8")
        sent = social.strip_review_claim(legacy)
        monkeypatch.setattr(social_verify, "http_get", RecordingGet(appview_post(text=segments_text(sent), segments=sent)))

        assert social_verify.cmd_social_verify(self.args(draft="2026-10-obs1")) == 0

    def test_an_unknown_uri_is_an_error_naming_it(self, tmp_path, monkeypatch):
        self.world(tmp_path, monkeypatch)

        with pytest.raises(social_verify.VerifyError, match="is not a post in the ledger"):
            social_verify.cmd_social_verify(self.args(uri="at://did:plc:me/app.bsky.feed.post/zzz"))

    def test_last_with_nothing_posted_is_an_error(self, tmp_path, monkeypatch):
        self.world(tmp_path, monkeypatch, posted=False)

        with pytest.raises(social_verify.VerifyError, match="the ledger has no posted draft"):
            social_verify.cmd_social_verify(self.args(last=True))

    def test_the_command_is_reachable_through_the_cli_parser(self):
        social_hooks.reset()
        parser = cli.build_parser()

        ns = parser.parse_args(["social-verify", "--last"])

        assert (ns.command, ns.last, ns.uri, ns.draft) == ("social-verify", True, None, None)
        assert social_verify.COMMANDS["social-verify"] is social_verify.cmd_social_verify


class TestWarnVersusError:
    def run_hook(self, tmp_path, get):
        audit = Audit(tmp_path / "audit.jsonl", run="r1", now=lambda: NOW)
        hook = social_verify.make_hook(get=get, sleep=RecordingSleep(), audit=audit)
        ctx = {"draft_id": "d1", "draft": {"text": SEGMENTS}, "uri": URI, "cid": "c", "now": NOW, "manual": False, "rules": rules_with(),
               "request": {"text": TEXT, "langs": ["en"], "images": [{"path": "a", "alt": a, "width": 1, "height": 1} for a in ALTS]}}
        hook(ctx)
        return audit.records()

    def test_a_post_the_appview_does_not_have_yet_is_level_warn_and_keeps_its_problems(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet({"posts": []}))
        assert [(r["event"], r["level"], r["problems"], r["attempts"]) for r in records] == [
            ("post_verify_failed", "warn", ["the AppView does not have the post after 10 attempts"], 10)]

    def test_an_unreachable_appview_is_level_warn(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet(OSError("no route to host")))
        assert [(r["event"], r["level"]) for r in records] == [("post_verify_failed", "warn")]

    def test_a_post_that_was_read_back_and_differs_is_level_error(self, tmp_path):
        records = self.run_hook(tmp_path, RecordingGet(appview_post(text=TEXT + "!")))
        assert [(r["event"], r["level"], r["attempts"]) for r in records] == [("post_verify_failed", "error", 1)]

    def test_a_good_post_has_no_failed_record(self, tmp_path):
        assert [r["event"] for r in self.run_hook(tmp_path, RecordingGet(appview_post()))] == ["post_verified"]

    def test_a_crash_inside_the_check_is_level_error(self, tmp_path):
        hook = social_verify.make_hook(get=RecordingGet({"posts": []}), sleep=RecordingSleep(), audit=Audit(tmp_path / "a.jsonl", run="r1", now=lambda: NOW))
        hook({"draft_id": "d1", "uri": URI})                                          # no "request" key
        recs = Audit(tmp_path / "a.jsonl").records()
        assert [(r["event"], r["level"], r["attempts"]) for r in recs] == [("post_verify_failed", "error", 0)]
