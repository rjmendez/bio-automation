"""social_cleanup: planning and running the removal of the old inat_skeet bot posts. The network (get_json) and the bsky-oauth account
(delete_many) are the stubbed dependencies; plan building, the plan id, the ledger and resume logic run for real."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from bsky_oauth import BatchInterrupted, NothingSent

from bio_automation import cli, social_cleanup
from bio_automation.social_rules import Audit
from bio_automation.social import SocialError

BOT = "Turdus migratorius - American Robin\n\nQuality grade: research\n2025-05-28\nWarning! AI generated alt text."
DID = "did:plc:me"


def rec(n, text, created):
    return {"uri": f"at://{DID}/app.bsky.feed.post/{n}", "value": {"text": text, "createdAt": created}}


RECORDS = [rec(1, BOT, "2025-03-02T00:00:00Z"), rec(2, BOT, "2025-02-20T00:00:00Z"), rec(3, BOT, "2025-04-01T00:00:00Z"),
           rec(4, BOT, "2025-05-01T00:00:00Z"), rec(5, "my own post", "2025-01-01T00:00:00Z")]
ENGAGEMENT = {f"at://{DID}/app.bsky.feed.post/1": (3, 0, 0, 0), f"at://{DID}/app.bsky.feed.post/2": (0, 0, 0, 0),
              f"at://{DID}/app.bsky.feed.post/3": (8, 1, 2, 0)}        # post 4 is not returned by the AppView


def routes(url):
    if url.startswith("https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle"):
        return {"did": DID}
    if url.startswith("https://plc.directory/"):
        return {"service": [{"id": "#atproto_pds", "serviceEndpoint": "https://pds.example"}]}
    if url.startswith("https://pds.example/xrpc/com.atproto.repo.listRecords"):
        return {"records": RECORDS}
    if url.startswith("https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts"):
        asked = [u for k, u in __import__("urllib.parse").parse.parse_qsl(url.split("?", 1)[1]) if k == "uris"]
        return {"posts": [{"uri": u, "likeCount": e[0], "repostCount": e[1], "replyCount": e[2], "quoteCount": e[3]}
                          for u, e in ENGAGEMENT.items() if u in asked]}
    raise AssertionError(url)


class TestPlan:
    def test_bot_posts_without_replies_are_planned_and_the_rest_kept_with_reasons(self):
        plan = social_cleanup.build_plan("me.example", get_json=routes)
        assert [(d["uri"].rsplit("/", 1)[1], d["created_at"][:10], d["engagement"]["likes"]) for d in plan["delete"]] == [
            ("2", "2025-02-20", 0), ("1", "2025-03-02", 3)]
        assert [(k["uri"].rsplit("/", 1)[1], k["reason"]) for k in plan["keep"]] == [
            ("3", "2 replies from people"), ("4", "engagement unknown (not returned by the AppView)")]
        assert plan["delete"][0]["text"] == "Turdus migratorius - American Robin"
        assert plan["plan_id"] == social_cleanup.plan_id([f"at://{DID}/app.bsky.feed.post/1", f"at://{DID}/app.bsky.feed.post/2"])
        assert (plan["did"], plan["pds"], plan["rule"]) == (DID, "https://pds.example", "inat_skeet template posts with no replies")

    def test_plan_id_ignores_order_and_changes_with_content(self):
        assert social_cleanup.plan_id(["b", "a"]) == social_cleanup.plan_id(["a", "b"])
        assert social_cleanup.plan_id(["a", "b"]) != social_cleanup.plan_id(["a", "c"])

    def test_summary(self):
        plan = social_cleanup.build_plan("me.example", get_json=routes)
        assert {k: v for k, v in social_cleanup.summary(plan).items() if k != "plan_id"} == {
            "rule": "inat_skeet template posts with no replies", "delete": 2, "keep": 2,
            "kept_because": {"has replies": 1, "engagement unknown": 1}, "date_range": ["2025-02-20", "2025-03-02"],
            "likes_on_deleted": 3, "samples": ["2025-02-20 Turdus migratorius - American Robin", "2025-03-02 Turdus migratorius - American Robin"]}


class RecordingAccount:
    """Stands in for bsky_oauth.Account.delete_many: records each batch, answers from a script of per-call outcomes."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def delete_many(self, uris, *, pause_ms):
        self.calls.append(list(uris))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return [{"uri": u, **step.get(u, {"status": "deleted"})} for u in uris if u not in step.get("_omit", [])]


def plan_file(tmp_path, n=5):
    uris = [f"at://{DID}/app.bsky.feed.post/{i}" for i in range(1, n + 1)]
    plan = {"kind": "bsky_cleanup_plan", "pds": "https://pds.example", "plan_id": social_cleanup.plan_id(uris),
            "delete": [{"uri": u} for u in uris], "keep": []}
    path = tmp_path / f"cleanup_plan_{plan['plan_id']}.json"
    path.write_text(json.dumps(plan))
    return path, plan["plan_id"], uris


def ledger(path):
    return [json.loads(l) for l in path.with_suffix(".ledger.jsonl").read_text().splitlines()]


def gone(_uri):
    return False


def still_there(_uri):
    return True


@pytest.fixture
def audit(tmp_path):
    return Audit(tmp_path / "audit.jsonl", run="r1")


def audit_events(audit, name):
    return [r for r in audit.records() if r["event"] == name]


class TestRun:
    def test_wrong_confirmation_or_an_edited_plan_is_refused_and_the_right_one_runs(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 2)
        acct = RecordingAccount([{}])
        with pytest.raises(SocialError, match=r"--confirm 'nope' does not match plan " + pid):
            social_cleanup.run_plan(path, "nope", confirm_count=2, account=acct, exists=gone, audit=audit)
        assert acct.calls == []
        edited = json.loads(path.read_text())
        edited["delete"].append({"uri": f"at://{DID}/app.bsky.feed.post/99"})
        bad = tmp_path / "edited.json"
        bad.write_text(json.dumps(edited))
        with pytest.raises(SocialError, match=r"was edited after it was built"):
            social_cleanup.run_plan(bad, pid, confirm_count=2, account=acct, exists=gone, audit=audit)
        assert acct.calls == []
        report = social_cleanup.run_plan(path, pid, confirm_count=2, account=acct, exists=gone, audit=audit)
        assert report == {"plan_id": pid, "dry_run": False, "already_done": 0, "deleted": 2, "failed": 0, "stopped": None, "remaining": 0}
        assert acct.calls == [uris]

    def test_batches_ledger_and_resume(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 5)
        acct = RecordingAccount([{}, {"_omit": [uris[3]], uris[2]: {"status": "rate_limited", "error": "Rate Limit Exceeded"}}])
        first = social_cleanup.run_plan(path, pid, confirm_count=5, account=acct, exists=gone, chunk=2, audit=audit)
        assert acct.calls == [uris[:2], uris[2:4]]
        assert (first["deleted"], first["stopped"], first["remaining"]) == (2, "rate limited: rerun later to continue", 3)
        assert [(r["uri"], r["status"]) for r in ledger(path)] == [(uris[0], "deleted"), (uris[1], "deleted"), (uris[2], "rate_limited")]
        again = RecordingAccount([{}, {}])
        second = social_cleanup.run_plan(path, pid, confirm_count=5, account=again, exists=gone, chunk=2, audit=audit)
        assert again.calls == [uris[2:4], uris[4:]]
        assert (second["already_done"], second["deleted"], second["remaining"], second["stopped"]) == (2, 3, 0, None)

    def test_a_failed_delete_counts_as_done_only_when_the_record_is_gone(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 2)
        failing = {u: {"status": "failed", "error": "Could not locate record"} for u in uris}
        r1 = social_cleanup.run_plan(path, pid, confirm_count=2, account=RecordingAccount([failing]), exists=still_there, audit=audit)
        assert (r1["deleted"], r1["failed"], r1["remaining"]) == (0, 2, 2)
        r2 = social_cleanup.run_plan(path, pid, confirm_count=2, account=RecordingAccount([failing]), exists=gone, audit=audit)
        assert (r2["deleted"], r2["failed"], r2["remaining"]) == (2, 0, 0)

    def test_lost_session_stops_and_an_interrupted_batch_keeps_what_was_done(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        r = social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([NothingSent("refresh refused")]), exists=gone, audit=audit)
        assert (r["stopped"], r["remaining"]) == ("nothing sent: refresh refused", 3)
        partial = BatchInterrupted([{"uri": uris[0], "status": "deleted"}], "socket hang up")
        r = social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([partial]), exists=gone, audit=audit)
        assert r["stopped"].startswith("interrupted: socket hang up") and r["remaining"] == 2
        assert [x["uri"] for x in ledger(path) if x["status"] == "deleted"] == [uris[0]]


class RaisingAccount:
    """Any use of the account is a test failure: a dry run and a refused run must never reach it."""

    def delete_many(self, uris, *, pause_ms):
        raise AssertionError(f"delete_many called with {uris}")


class TestDryRunAndCount:
    def test_without_confirmation_it_is_a_dry_run_that_deletes_nothing_and_writes_no_ledger(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        report = social_cleanup.run_plan(path, account=RaisingAccount(), audit=audit)
        assert report == {"plan_id": pid, "dry_run": True, "plan_size": 3, "already_done": 0, "would_delete": 3, "deleted": 0, "failed": 0,
                          "stopped": None, "remaining": 3}
        assert not path.with_suffix(".ledger.jsonl").exists()
        assert [(r["would_delete"], r["already_done"]) for r in audit_events(audit, "cleanup_dry_run")] == [(3, 0)]
        assert audit_events(audit, "cleanup_delete") == []

    def test_a_dry_run_after_a_partial_run_counts_only_what_is_left(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([{"_omit": uris[1:]}]), exists=gone, audit=audit)
        report = social_cleanup.run_plan(path, account=RaisingAccount(), audit=audit)
        assert (report["dry_run"], report["already_done"], report["would_delete"], report["plan_size"]) == (True, 1, 2, 3)

    @pytest.mark.parametrize("count", [2, 4, 0, -3])
    def test_a_count_that_is_not_the_plan_size_is_refused_before_anything_is_sent(self, tmp_path, audit, count):
        path, pid, _ = plan_file(tmp_path, 3)
        with pytest.raises(SocialError, match=rf"--confirm-count {count} does not match the plan, which deletes 3 posts"):
            social_cleanup.run_plan(path, pid, confirm_count=count, account=RaisingAccount(), exists=gone, audit=audit)
        assert not path.with_suffix(".ledger.jsonl").exists()
        assert audit_events(audit, "cleanup_start") == []

    def test_the_right_count_runs(self, tmp_path, audit):                      # positive twin of the refusals above
        path, pid, uris = plan_file(tmp_path, 3)
        acct = RecordingAccount([{}])
        report = social_cleanup.run_plan(path, pid, confirm_count=3, account=acct, exists=gone, audit=audit)
        assert (report["deleted"], acct.calls) == (3, [uris])

    @pytest.mark.parametrize("with_id,count", [(True, None), (False, 3)])
    def test_only_one_of_the_two_confirmations_is_refused(self, tmp_path, audit, with_id, count):
        path, pid, _ = plan_file(tmp_path, 3)
        with pytest.raises(SocialError, match="needs both --confirm <plan id> and --confirm-count"):
            social_cleanup.run_plan(path, pid if with_id else None, confirm_count=count, account=RaisingAccount(), audit=audit)

    def test_a_uri_of_another_account_or_collection_is_refused_for_a_plan_with_a_did(self, tmp_path, audit):
        uris = [f"at://{DID}/app.bsky.feed.post/1", "at://did:plc:other/app.bsky.feed.post/2"]
        plan = {"kind": "bsky_cleanup_plan", "pds": "https://pds.example", "did": DID, "plan_id": social_cleanup.plan_id(uris),
                "delete": [{"uri": u} for u in uris], "keep": []}
        path = tmp_path / "foreign.json"
        path.write_text(json.dumps(plan))
        with pytest.raises(SocialError, match=r"1 URI\(s\) of the plan are not app.bsky.feed.post records of the plan's account \(first: at://did:plc:other"):
            social_cleanup.run_plan(path, plan["plan_id"], confirm_count=2, account=RaisingAccount(), exists=gone, audit=audit)
        liked = [f"at://{DID}/app.bsky.feed.like/1"]
        plan.update(delete=[{"uri": u} for u in liked], plan_id=social_cleanup.plan_id(liked))
        path.write_text(json.dumps(plan))
        with pytest.raises(SocialError, match="not app.bsky.feed.post records"):
            social_cleanup.run_plan(path, plan["plan_id"], confirm_count=1, account=RaisingAccount(), exists=gone, audit=audit)
        own = [f"at://{DID}/app.bsky.feed.post/1"]                                  # twin: the plan's own post passes the guard
        plan.update(delete=[{"uri": u} for u in own], plan_id=social_cleanup.plan_id(own))
        path.write_text(json.dumps(plan))
        assert social_cleanup.run_plan(path, plan["plan_id"], confirm_count=1, account=RecordingAccount([{}]), exists=gone, audit=audit)["deleted"] == 1


class TestAudit:
    def test_every_deletion_result_is_an_audit_record_with_the_uri_and_no_text(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        acct = RecordingAccount([{uris[1]: {"status": "failed", "error": "Could not locate record"}}])
        social_cleanup.run_plan(path, pid, confirm_count=3, account=acct, exists=still_there, audit=audit)
        assert [(r["uri"], r["status"], r.get("error"), r["plan_id"]) for r in audit_events(audit, "cleanup_delete")] == [
            (uris[0], "deleted", None, pid), (uris[1], "failed", "Could not locate record", pid), (uris[2], "deleted", None, pid)]
        [start] = audit_events(audit, "cleanup_start")
        assert (start["plan_size"], start["todo"], start["already_done"]) == (3, 3, 0)
        [end] = audit_events(audit, "cleanup_end")
        assert (end["deleted"], end["failed"], end["remaining"], end["stopped"]) == (2, 1, 1, None)
        assert [r["event"] for r in audit.records()] == ["cleanup_start", "cleanup_delete", "cleanup_delete", "cleanup_delete", "cleanup_end"]

    def test_a_resumed_run_audits_only_the_new_deletions(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([{"_omit": uris[1:]}]), exists=gone, audit=audit)
        social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([{}]), exists=gone, audit=audit)
        assert [r["uri"] for r in audit_events(audit, "cleanup_delete")] == uris          # each URI once, in order, across both runs
        assert [r["already_done"] for r in audit_events(audit, "cleanup_start")] == [0, 1]

    def test_an_interrupted_run_still_writes_its_end_record(self, tmp_path, audit):
        path, pid, uris = plan_file(tmp_path, 3)
        partial = BatchInterrupted([{"uri": uris[0], "status": "deleted"}], "socket hang up")
        social_cleanup.run_plan(path, pid, confirm_count=3, account=RecordingAccount([partial]), exists=gone, audit=audit)
        [end] = audit_events(audit, "cleanup_end")
        assert (end["deleted"], end["remaining"], end["stopped"].startswith("interrupted: socket hang up")) == (1, 2, True)

    def test_an_unwritable_audit_log_stops_the_run_before_the_first_delete(self, tmp_path):
        path, pid, _ = plan_file(tmp_path, 2)
        blocked = tmp_path / "audit-is-a-dir"
        blocked.mkdir()
        with pytest.raises(OSError, match="audit-is-a-dir"):
            social_cleanup.run_plan(path, pid, confirm_count=2, account=RaisingAccount(), exists=gone, audit=Audit(blocked))
        assert not path.with_suffix(".ledger.jsonl").exists()


class TestCli:
    def run(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["bio", *argv])
        return cli.main()

    def test_no_confirmation_is_a_dry_run_exit_0_and_prints_the_confirm_command(self, tmp_path, monkeypatch, capsys):
        path, pid, _ = plan_file(tmp_path, 3)
        assert self.run(monkeypatch, "social-cleanup-run", str(path)) == 0
        out, err = capsys.readouterr()
        assert json.loads(out)["would_delete"] == 3
        assert err == f"dry run: nothing deleted. To delete: --confirm {pid} --confirm-count 3\n"

    def test_a_wrong_count_exits_1_with_the_message(self, tmp_path, monkeypatch, capsys):
        path, pid, _ = plan_file(tmp_path, 3)
        assert self.run(monkeypatch, "social-cleanup-run", str(path), "--confirm", pid, "--confirm-count", "9") == 1
        assert capsys.readouterr().err.startswith("bio social-cleanup-run: --confirm-count 9 does not match the plan, which deletes 3 posts")


class TestRecordExists:
    def test_record_not_found_is_false_found_is_true_other_errors_raise(self):
        def missing(url):
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"RecordNotFound"}'))

        def broken(url):
            raise urllib.error.HTTPError(url, 500, "Server Error", {}, io.BytesIO(b"oops"))

        calls = []
        assert social_cleanup.record_exists(f"at://{DID}/app.bsky.feed.post/1", "https://pds.example", get_json=missing) is False
        assert social_cleanup.record_exists(f"at://{DID}/app.bsky.feed.post/1", "https://pds.example",
                                            get_json=lambda u: calls.append(u) or {"uri": u}) is True
        assert calls == ["https://pds.example/xrpc/com.atproto.repo.getRecord?repo=did%3Aplc%3Ame&collection=app.bsky.feed.post&rkey=1"]
        with pytest.raises(urllib.error.HTTPError, match="Server Error"):
            social_cleanup.record_exists(f"at://{DID}/app.bsky.feed.post/1", "https://pds.example", get_json=broken)
