"""social_engagement: public-count snapshots of the posted Bluesky posts and their aggregates. The stubs are recording ones: a `get` that
returns counts for the URIs it is asked about (and notes each call) and a `sleep` that notes its pauses."""
from __future__ import annotations

import argparse
import datetime as dt
import json

import pytest

from bio_automation import settings, social_engagement as eng
from bio_automation.social import Ledger

TZ = dt.timezone(dt.timedelta(hours=-4))
NOW = dt.datetime(2026, 10, 23, 12, 0, 0, tzinfo=TZ)


def uri(n):
    return f"at://did:plc:abc/app.bsky.feed.post/{n}"


class RecordingGet:
    """Answers each call with a post for every asked URI that is not in `absent`; likeCount is the URI's number, the rest are fixed."""

    def __init__(self, absent=()):
        self.calls = []
        self.absent = set(absent)

    def __call__(self, url, params):
        self.calls.append((url, list(params)))
        asked = [v for k, v in params]
        return {"posts": [{"uri": u, "cid": "c", "author": {"handle": "someone.example"}, "record": {"text": "other people's words"},
                           "likeCount": int(u.rsplit("/", 1)[1]), "repostCount": 2, "replyCount": 1, "quoteCount": 0}
                          for u in asked if u not in self.absent]}


class RecordingSleep:
    def __init__(self):
        self.pauses = []

    def __call__(self, s):
        self.pauses.append(s)


def make_ledger(tmp_path, posts):
    """posts: [(draft_id, theme | None, obs_ids, posted_at, uri)] -> a Ledger with a drafted + posted record for each."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for draft_id, theme, obs_ids, at, u in posts:
        extra = {"theme": theme, "obs_ids": obs_ids} if theme else {}
        ledger.append("drafted", draft_id, now=at - dt.timedelta(hours=1), obs_id=obs_ids[0], month="2026-10", **extra)
        ledger.append("posted", draft_id, now=at, obs_id=obs_ids[0], uri=u, cid="cid")
    return ledger


def n_posts(tmp_path, n, at=NOW - dt.timedelta(hours=2)):
    return make_ledger(tmp_path, [(f"d{i}", None, [i], at, uri(i)) for i in range(1, n + 1)])


@pytest.mark.parametrize("n, sizes, pauses", [(24, [24], []), (25, [25], []), (26, [25, 1], [1.0])])
def test_batches_hold_at_most_25_uris_and_pause_a_second_between_requests(tmp_path, n, sizes, pauses):
    get, sleep = RecordingGet(), RecordingSleep()
    out = eng.snapshot(days=30, ledger=n_posts(tmp_path, n), log=tmp_path / "e.jsonl", get=get, sleep=sleep, now=NOW)
    assert [len(params) for _, params in get.calls] == sizes
    assert sleep.pauses == pauses
    assert out == {"posts": n, "snapshots": n, "missing": []}
    assert len((tmp_path / "e.jsonl").read_text().splitlines()) == n
    assert all(url == eng.APPVIEW and all(k == "uris" for k, _ in params) for url, params in get.calls)


def test_a_batch_boundary_does_not_lose_or_repeat_a_uri(tmp_path):
    get = RecordingGet()
    eng.snapshot(days=30, ledger=n_posts(tmp_path, 26), log=tmp_path / "e.jsonl", get=get, sleep=RecordingSleep(), now=NOW)
    asked = [v for _, params in get.calls for _, v in params]
    assert asked == [uri(i) for i in range(1, 27)]


def test_snapshot_line_holds_the_counts_and_the_post_facts_and_nothing_of_other_users(tmp_path):
    ledger = make_ledger(tmp_path, [("2026-10-23-fungi-friday", "fungi-friday", [7, 8], dt.datetime(2026, 10, 23, 8, 16, tzinfo=TZ), uri(11))])
    eng.snapshot(days=30, ledger=ledger, log=tmp_path / "e.jsonl", get=RecordingGet(), sleep=RecordingSleep(), now=NOW)
    raw = (tmp_path / "e.jsonl").read_text()
    assert json.loads(raw) == {"at": "2026-10-23T12:00:00-04:00", "uri": uri(11), "draft_id": "2026-10-23-fungi-friday", "theme": "fungi-friday",
                               "weekday": "Fri", "hour": 8, "likeCount": 11, "repostCount": 2, "replyCount": 1, "quoteCount": 0}
    assert "someone.example" not in raw and "other people" not in raw


def test_a_post_the_appview_does_not_return_is_reported_missing_and_gets_no_snapshot(tmp_path):
    out = eng.snapshot(days=30, ledger=n_posts(tmp_path, 3), log=tmp_path / "e.jsonl", get=RecordingGet(absent=[uri(2)]),
                       sleep=RecordingSleep(), now=NOW)
    assert out == {"posts": 3, "snapshots": 2, "missing": [uri(2)]}
    assert [json.loads(x)["uri"] for x in (tmp_path / "e.jsonl").read_text().splitlines()] == [uri(1), uri(3)]


def test_no_posts_means_no_request_and_no_file(tmp_path):
    get = RecordingGet()
    out = eng.snapshot(days=30, ledger=Ledger(tmp_path / "none.jsonl"), log=tmp_path / "e.jsonl", get=get, sleep=RecordingSleep(), now=NOW)
    assert out == {"posts": 0, "snapshots": 0, "missing": []}
    assert get.calls == [] and not (tmp_path / "e.jsonl").exists()


def test_days_window_includes_a_post_exactly_n_days_old_and_excludes_one_a_second_older(tmp_path):
    edge = NOW - dt.timedelta(days=7)
    ledger = make_ledger(tmp_path, [("in", None, [1], edge, uri(1)), ("out", None, [2], edge - dt.timedelta(seconds=1), uri(2)),
                                    ("fresh", None, [3], NOW - dt.timedelta(minutes=5), uri(3))])
    assert [p["draft_id"] for p in eng.posted(ledger, days=7, now=NOW)] == ["in", "fresh"]
    assert [p["draft_id"] for p in eng.posted(ledger, days=8, now=NOW)] == ["out", "in", "fresh"]


def test_a_draft_posted_twice_counts_once_with_its_latest_uri(tmp_path):
    ledger = make_ledger(tmp_path, [("d", None, [1], NOW - dt.timedelta(hours=5), uri(1))])
    ledger.append("posted", "d", now=NOW - dt.timedelta(hours=1), obs_id=1, uri=uri(9), cid="x")
    assert [p["uri"] for p in eng.posted(ledger, days=30, now=NOW)] == [uri(9)]


def test_theme_weekday_and_hour_come_from_the_ledger_and_singles_are_called_single(tmp_path):
    ledger = make_ledger(tmp_path, [("t", "fungi-friday", [5, 6], dt.datetime(2026, 10, 23, 8, 16, tzinfo=TZ), uri(1)),
                                    ("s", None, [9], dt.datetime(2026, 10, 22, 20, 5, tzinfo=TZ), uri(2))])
    got = [(p["theme"], p["weekday"], p["hour"], p["obs_id"]) for p in eng.posted(ledger, days=30, now=NOW)]
    assert got == [("single", "Thu", 20, 9), ("fungi-friday", "Fri", 8, 5)]


# ----------------------------------------------------------------------------------------------- aggregation on a hand-made fixture

def snap(draft_id, theme, weekday, hour, likes, reposts=0, replies=0, quotes=0, u=None, at="2026-10-23T12:00:00-04:00"):
    return {"at": at, "uri": u or uri(draft_id), "draft_id": draft_id, "theme": theme, "weekday": weekday, "hour": hour,
            "likeCount": likes, "repostCount": reposts, "replyCount": replies, "quoteCount": quotes}


FIXTURE = [snap("A", "fungi-friday", "Fri", 8, 10, 2, 1), snap("B", "fungi-friday", "Fri", 8, 4),
           snap("C", "single", "Thu", 12, 6, 1, 2, 1), snap("D", "single", "Wed", 20, 0)]


def test_aggregate_by_theme_numbers():
    got = eng.aggregate(FIXTURE, lambda s: s["theme"])
    assert got == {
        "fungi-friday": {"posts": 2, "mean_likes": 7.0, "median_likes": 7.0, "mean_reposts": 1.0, "mean_replies": 0.5, "mean_score": 8.5,
                         "best": {"draft_id": "A", "score": 13}, "worst": {"draft_id": "B", "score": 4}},
        "single": {"posts": 2, "mean_likes": 3.0, "median_likes": 3.0, "mean_reposts": 0.5, "mean_replies": 1.0, "mean_score": 5.0,
                   "best": {"draft_id": "C", "score": 10}, "worst": {"draft_id": "D", "score": 0}}}


def test_median_differs_from_mean_with_three_posts():
    rows = [snap("a", "t", "Mon", 9, 0), snap("b", "t", "Mon", 9, 1), snap("c", "t", "Mon", 9, 8)]
    got = eng.aggregate(rows, lambda s: s["theme"])["t"]
    assert (got["mean_likes"], got["median_likes"]) == (3.0, 1)


def test_latest_snapshot_of_a_post_wins():
    old = snap("A", "t", "Mon", 9, 1, at="2026-10-20T12:00:00-04:00")
    new = snap("A", "t", "Mon", 9, 5, at="2026-10-23T12:00:00-04:00")
    assert [s["likeCount"] for s in eng.latest_per_post([old, new])] == [5]


def test_report_groups_by_weekday_and_hour():
    report = eng.build_report(FIXTURE)
    assert {k: v["posts"] for k, v in report["by_weekday"].items()} == {"Fri": 2, "Thu": 1, "Wed": 1}
    assert {k: v["posts"] for k, v in report["by_hour"].items()} == {"8": 2, "12": 1, "20": 1}
    assert report["posts"] == 4


def test_suggestion_names_the_best_theme_with_its_sample_sizes():
    assert eng.build_report(FIXTURE)["suggestions"] == [
        "Best theme: fungi-friday averages 8.5 engagements over 2 posts (weakest: single at 5.0 over 2)."]


def test_no_suggestion_from_groups_of_one_post_says_so():
    report = eng.build_report(FIXTURE[:3])          # fungi 2 posts, single 1 post: nothing to compare yet
    assert report["suggestions"] == ["Too few posts to suggest anything yet: a theme, day or hour needs 2 posts, and two of them, to be compared."]


def test_no_suggestion_when_the_groups_tie():
    rows = [snap("a", "x", "Mon", 9, 3), snap("b", "x", "Mon", 9, 3), snap("c", "y", "Mon", 9, 3), snap("d", "y", "Mon", 9, 3)]
    assert eng.build_report(rows)["suggestions"] == [
        "Too few posts to suggest anything yet: a theme, day or hour needs 2 posts, and two of them, to be compared."]


# ----------------------------------------------------------------------------------------------- taxonomy

def lineage(kingdom, cls):
    return {"taxon": {"lineage": [{"name": kingdom, "rank": "kingdom"}, {"name": cls, "rank": "class"}]}}


def taxonomy_ledger(tmp_path):
    at = NOW - dt.timedelta(hours=3)
    return make_ledger(tmp_path, [("A", "fungi-friday", [1, 2], at, uri("A")), ("B", "fungi-friday", [3], at, uri("B")),
                                  ("C", None, [5], at, uri("C")), ("D", None, [6], at, uri("D"))])


def test_kingdom_and_class_aggregates_use_the_lead_observation_and_skip_unknown_ones(tmp_path):
    tax = {"1": lineage("Animalia", "Insecta"), "3": lineage("Fungi", "Agaricomycetes"), "5": lineage("Animalia", "Arachnida")}   # 6 unknown
    report = eng.build_report(FIXTURE, ledger=taxonomy_ledger(tmp_path), taxonomy=tax)
    assert {k: v["posts"] for k, v in report["by_kingdom"].items()} == {"Animalia": 2, "Fungi": 1}
    assert {k: v["posts"] for k, v in report["by_class"].items()} == {"Insecta": 1, "Agaricomycetes": 1, "Arachnida": 1}
    assert report["by_kingdom"]["Animalia"]["mean_likes"] == 8.0            # A (10) and C (6); D has no taxonomy entry and is left out


def test_without_a_taxonomy_the_report_has_no_kingdom_or_class_and_everything_else_is_unchanged():
    report = eng.build_report(FIXTURE, taxonomy=None)
    assert report["by_kingdom"] is None and report["by_class"] is None
    assert report["by_theme"]["fungi-friday"]["posts"] == 2


def test_missing_taxonomy_file_is_an_empty_cache(tmp_path):
    assert eng.load_taxonomy_cache(tmp_path / "nope.json") == {}
    (tmp_path / "t.json").write_text('{"1": {"taxon": {}}}')
    assert eng.load_taxonomy_cache(tmp_path / "t.json") == {"1": {"taxon": {}}}


# ----------------------------------------------------------------------------------------------- digest entry, command, http

def write_log(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_engagement_by_theme_reads_the_log_within_the_window(tmp_path):
    log = tmp_path / "e.jsonl"
    old = snap("Z", "old-theme", "Mon", 9, 99, at="2026-09-01T12:00:00-04:00")
    write_log(log, [*FIXTURE, old])
    got = eng.engagement_by_theme(30, log=log, now=NOW)
    assert sorted(got) == ["fungi-friday", "single"]
    assert got["fungi-friday"]["mean_score"] == 8.5
    assert sorted(eng.engagement_by_theme(60, log=log, now=NOW)) == ["fungi-friday", "old-theme", "single"]


def test_engagement_by_theme_without_a_log_is_empty(tmp_path):
    assert eng.engagement_by_theme(30, log=tmp_path / "none.jsonl", now=NOW) == {}


def test_report_only_prints_the_stored_aggregates_and_makes_no_request(tmp_path, capsys, monkeypatch):
    fetched = []
    monkeypatch.setattr(eng, "snapshot", lambda **kw: fetched.append(kw))
    settings.SOCIAL_ENGAGEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(settings.LOCAL_TZ).isoformat(timespec="seconds")
    write_log(settings.SOCIAL_ENGAGEMENT_LOG, [dict(s, at=now) for s in FIXTURE])
    assert eng.cmd_social_engagement(argparse.Namespace(days=30, refresh=False, report=True)) == 0
    out = capsys.readouterr().out
    assert fetched == []
    assert "Engagement over 4 posts" in out and "fungi-friday" in out and "Best theme: fungi-friday averages 8.5" in out


def test_refresh_with_report_fetches_first(tmp_path, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(eng, "snapshot", lambda **kw: calls.append(kw) or {"posts": 0, "snapshots": 0, "missing": []})
    assert eng.cmd_social_engagement(argparse.Namespace(days=5, refresh=True, report=True)) == 0
    assert calls == [{"days": 5}]
    assert "0 posts in the last 5 days" in capsys.readouterr().out


def test_days_below_one_is_refused():
    with pytest.raises(eng.EngagementError, match="--days must be at least 1, not 0"):
        eng.cmd_social_engagement(argparse.Namespace(days=0, refresh=False, report=False))


def test_http_get_sends_repeated_uris_params_and_no_contact_address(monkeypatch):
    seen = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"posts": []}'

    def urlopen(req, timeout):
        seen.append(req)
        return Resp()

    monkeypatch.setattr(eng.urllib.request, "urlopen", urlopen)
    assert eng.http_get(eng.APPVIEW, [("uris", "at://a/b/1"), ("uris", "at://a/b/2")]) == {"posts": []}
    req = seen[0]
    assert req.full_url == eng.APPVIEW + "?uris=at%3A%2F%2Fa%2Fb%2F1&uris=at%3A%2F%2Fa%2Fb%2F2"
    assert "@" not in req.get_header("User-agent")


def test_http_failure_is_one_clear_error(monkeypatch):
    def urlopen(req, timeout):
        raise eng.urllib.error.URLError("no route")

    monkeypatch.setattr(eng.urllib.request, "urlopen", urlopen)
    with pytest.raises(eng.EngagementError, match="the Bluesky AppView did not answer"):
        eng.http_get(eng.APPVIEW, [("uris", "x")])
