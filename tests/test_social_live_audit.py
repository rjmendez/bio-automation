"""social-audit-live: the privacy gates re-run on the owner's PUBLIC posts. The AppView and the image CDN are the injected dependencies
(get_json / get_bytes); the gates, the ledger and drafts lookup, the audit log and the report all run for real."""
from __future__ import annotations

import datetime as dt
import http.client
import io
import json
import urllib.error
import urllib.parse

import pytest
from PIL import Image

from bio_automation import cli, settings, social_live_audit
from bio_automation.social_alerts import ERROR_EVENTS
from bio_automation.social_rules import Audit

DID = "did:plc:me"
CLEAN_TEXT = "Eastern Phoebe, photographed on 2026-05-02."
CLEAN_ALT = "A small grey-brown bird perched on a branch."
HIDDEN_OBS = {"geoprivacy": "obscured", "place_guess": "Quillwood Hollow, Berks County, Pennsylvania"}
OPEN_OBS = {"geoprivacy": None, "place_guess": "Quillwood Hollow, Berks County, Pennsylvania"}


def uri(n):
    return f"at://{DID}/app.bsky.feed.post/{n}"


PDS = "https://pds.example"
PLC_DOC = {"id": DID, "service": [{"id": "#atproto_pds", "type": "AtprotoPersonalDataServer", "serviceEndpoint": PDS + "/"}]}


def cid(n, i=0):
    return f"bafkrei{n}x{i}"


def blob_url(n, i=0):
    return f"{PDS}/xrpc/com.atproto.sync.getBlob?did={urllib.parse.quote(DID)}&cid={cid(n, i)}"


def post(n, text=CLEAN_TEXT, alts=(CLEAN_ALT,), quote=False, **extra):
    """An AppView post view whose record carries the blob refs. quote=True: a quote post with images (record-with-media views)."""
    images = [{"alt": a, "fullsize": f"https://cdn.example/{n}/{i}.jpg", "thumb": f"https://cdn.example/{n}/{i}_t.jpg"} for i, a in enumerate(alts)]
    refs = [{"alt": a, "image": {"$type": "blob", "ref": {"$link": cid(n, i)}, "mimeType": "image/jpeg"}} for i, a in enumerate(alts)]
    if quote:
        embed = {"$type": "app.bsky.embed.recordWithMedia#view", "record": {}, "media": {"$type": "app.bsky.embed.images#view", "images": images}}
        rec_embed = {"$type": "app.bsky.embed.recordWithMedia", "media": {"$type": "app.bsky.embed.images", "images": refs}}
    else:
        embed = {"images": images} if images else None
        rec_embed = {"$type": "app.bsky.embed.images", "images": refs} if refs else None
    record = {"text": text, **({"embed": rec_embed} if rec_embed else {})}
    return {"post": {"uri": uri(n), "author": {"did": DID}, "record": record, "embed": embed, **extra}}


def jpeg_bytes(color=(120, 140, 90), size=(640, 480), exif_gps=False):
    import numpy as np

    rng = np.random.default_rng(7)
    arr = np.clip(np.array(color, dtype=float) + rng.normal(0, 40, (size[1], size[0], 3)), 0, 255).astype("uint8")
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if exif_gps:
        exif = Image.Exif()
        exif.get_ifd(0x8825)[2] = (12.0, 3.0, 4.0)
        im.save(buf, "JPEG", exif=exif)
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


class Appview:
    """A scripted getAuthorFeed: pages of feed items, recording every URL it was asked for."""

    def __init__(self, pages):
        self.pages, self.urls = list(pages), []

    def __call__(self, url):
        self.urls.append(url)
        if not self.pages:
            raise AssertionError("asked for more pages than scripted: " + url)
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(settings, "BSKY_HANDLE", "example.bsky.social")
    return tmp_path


def write_ledger(world, rows):
    (world / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_draft(world, draft_id, obs):
    d = world / "social" / "drafts" / draft_id
    d.mkdir(parents=True)
    (d / "draft.json").write_text(json.dumps({"obs_id": 1, "observation": obs}), encoding="utf-8")


class FeedAndDid(Appview):
    """The feed page, then the DID document for every PLC lookup."""

    def __call__(self, url):
        if url.startswith(settings.PLC_URL):
            self.urls.append(url)
            return PLC_DOC
        return super().__call__(url)


def run(world, feed, *, images=False, get_bytes=None, limit=50):
    audit = Audit(world / "audit.jsonl", run="r1")
    return social_live_audit.audit_live("me.example", limit=limit, images=images, get_json=FeedAndDid([{"feed": feed}]),
                                        get_bytes=get_bytes or (lambda u: b""), audit=audit)


class TestTextAndAlt:
    def test_a_clean_post_is_clean(self, world):
        r = run(world, [post(1)])
        assert r == {"checked": 1, "problem_posts": 0, "place_unchecked": 1, "problems": [], "gps_verified": 0, "gps_unverified": 0, "images_checked": False}

    def test_coordinates_in_the_text_and_in_an_alt_are_reported_by_uri_and_kind_only(self, world):
        r = run(world, [post(1, text="Found at 12.34567, -65.43217 today"), post(2, alts=("Bird, lat 12.3456 lon -65.4321",)), post(3)])
        assert [(p["uri"], p["kinds"]) for p in r["problems"]] == [
            (uri(1), ["coordinates in post text"]), (uri(2), ["coordinates in alt of image 1"])]
        assert "12.34" not in json.dumps(r)

    def test_a_street_address_is_reported(self, world):
        r = run(world, [post(1, alts=("Taken behind 1234 Old Mill Rd at dusk",))])
        assert r["problems"] == [{"uri": uri(1), "kinds": ["street address in alt of image 1"]}]

    def test_a_missing_alt_text_and_too_many_images_are_reported(self, world):
        r = run(world, [post(1, alts=("", CLEAN_ALT)), post(2, alts=(CLEAN_ALT,) * 5)])
        assert [(p["uri"], p["kinds"]) for p in r["problems"]] == [
            (uri(1), ["image without alt text 1"]), (uri(2), ["more than 4 images"])]

    def test_the_old_bot_template_posts_are_not_flagged_for_a_missing_alt_but_still_for_a_leak(self, world):
        bot = "Turdus migratorius - American Robin" + chr(10) * 2 + "Quality grade: research" + chr(10) + "2025-05-28" + chr(10) + "Warning! AI generated alt text."
        assert run(world, [post(1, text=bot, alts=("", ""))])["problem_posts"] == 0
        r = run(world, [post(1, text=bot, alts=("lat 12.3456 lon -65.4321",))])
        assert r["problems"] == [{"uri": uri(1), "kinds": ["coordinates in alt of image 1"]}]

    def test_a_post_without_images_is_checked_on_its_text_alone(self, world):
        assert run(world, [post(1, alts=())])["problem_posts"] == 0
        assert run(world, [post(1, text="at 12.34567, -65.43217", alts=())])["problems"][0]["kinds"] == ["coordinates in post text"]


class TestHiddenPlace:
    def feed(self):
        return [post(1, alts=("A phoebe at Quillwood Hollow",)), post(2, alts=("A phoebe at Quillwood Hollow",))]

    def test_the_place_of_an_observation_that_hides_its_location_is_a_leak_where_the_draft_is_on_record(self, world):
        write_ledger(world, [{"event": "posted", "uri": uri(1), "draft_id": "d1"}])
        write_draft(world, "d1", HIDDEN_OBS)
        r = run(world, self.feed())
        assert r["problems"] == [{"uri": uri(1), "kinds": ["place finer than the county in alt of image 1"]}]
        assert r["place_unchecked"] == 1                       # post 2 has no draft on record: only coordinates and addresses could be checked
        assert "Quillwood" not in json.dumps(r)

    def test_the_same_words_are_a_leak_for_an_observation_with_a_public_place_too(self, world):    # owner rule 2026-10-24: never finer than the county
        write_ledger(world, [{"event": "posted", "uri": uri(1), "draft_id": "d1"}])
        write_draft(world, "d1", OPEN_OBS)
        r = run(world, self.feed())
        assert r["problems"] == [{"uri": uri(1), "kinds": ["place finer than the county in alt of image 1"]}]
        assert r["place_unchecked"] == 1
        assert "Quillwood" not in json.dumps(r)

    @pytest.mark.parametrize("obs", [HIDDEN_OBS, OPEN_OBS], ids=["hidden", "public"])
    def test_the_county_and_state_are_fine_for_a_hidden_and_for_a_public_place(self, world, obs):    # twin
        feed = [post(1, alts=("A phoebe in Berks County, Pennsylvania",)), post(2, alts=("A phoebe in Berks County, Pennsylvania",))]
        write_ledger(world, [{"event": "posted", "uri": uri(1), "draft_id": "d1"}])
        write_draft(world, "d1", obs)
        r = run(world, feed)
        assert (r["problem_posts"], r["place_unchecked"]) == (0, 1)

    def test_a_themed_draft_checks_every_one_of_its_observations(self, world):
        write_ledger(world, [{"event": "posted", "uri": uri(1), "draft_id": "d1"}])
        d = world / "social" / "drafts" / "d1"
        d.mkdir(parents=True)
        (d / "draft.json").write_text(json.dumps({"observations": {"1": OPEN_OBS, "2": HIDDEN_OBS}}), encoding="utf-8")
        assert run(world, [self.feed()[0]])["problems"][0]["kinds"] == ["place finer than the county in alt of image 1"]

    def test_a_missing_or_corrupt_draft_counts_as_unchecked_not_as_a_problem(self, world):
        write_ledger(world, [{"event": "posted", "uri": uri(1), "draft_id": "gone"}, {"event": "posted", "uri": uri(2), "draft_id": "bad"}])
        d = world / "social" / "drafts" / "bad"
        d.mkdir(parents=True)
        (d / "draft.json").write_text("{not json", encoding="utf-8")
        r = run(world, self.feed())
        assert (r["problem_posts"], r["place_unchecked"]) == (0, 2)


class TestImages:
    def fetch(self, mapping):
        def get(url):
            return mapping[url]
        return get

    def blobs(self, **by_post):
        """get_bytes for the CDN full-size copies (re-encoded: never any EXIF) and the PDS blobs (the originals, which may carry GPS)."""
        data = {}
        for n in (1, 2, 3, 4):
            data[f"https://cdn.example/{n}/0.jpg"] = jpeg_bytes()
            data[blob_url(n)] = jpeg_bytes()
        data.update(by_post)
        return self.fetch(data)

    def test_a_black_picture_and_a_tiny_one_are_reported_from_the_cdn_copy(self, world):
        data = {"https://cdn.example/1/0.jpg": jpeg_bytes(color=(0, 0, 0), size=(640, 480)),
                "https://cdn.example/2/0.jpg": jpeg_bytes(size=(100, 80))}
        r = run(world, [post(1), post(2), post(4)], images=True, get_bytes=self.blobs(**data))
        kinds = {p["uri"]: p["kinds"] for p in r["problems"]}
        assert kinds[uri(2)] == ["image floor: image is too small (image 1)"]
        assert uri(4) not in kinds                                   # twin: a normal photo passes
        assert kinds[uri(1)][0].startswith("image floor: image is ")

    def test_gps_tags_are_found_in_the_original_blob_and_not_in_the_exif_free_cdn_copy(self, world):
        # the CDN copy is re-encoded (no EXIF); only the blob on the PDS carries the GPS the author's file had
        r = run(world, [post(3), post(4)], images=True, get_bytes=self.blobs(**{blob_url(3): jpeg_bytes(exif_gps=True)}))
        assert [(p["uri"], p["kinds"]) for p in r["problems"]] == [(uri(3), ["GPS tags in a public image (image 1)"])]
        assert (r["gps_verified"], r["gps_unverified"]) == (2, 0)

    def test_a_cdn_copy_with_gps_is_not_what_the_gps_check_reads(self, world):
        r = run(world, [post(3)], images=True, get_bytes=self.blobs(**{"https://cdn.example/3/0.jpg": jpeg_bytes(exif_gps=True)}))
        assert r["problems"] == [] and r["gps_verified"] == 1          # the blob was clean; the CDN bytes are not consulted for GPS

    def test_a_blob_that_cannot_be_fetched_is_counted_not_verifiable_and_not_clean(self, world):
        def get(url):
            if "getBlob" in url:
                raise urllib.error.HTTPError(url, 404, "not found", None, None)
            return jpeg_bytes()
        r = run(world, [post(1)], images=True, get_bytes=get)
        assert r["problems"] == [] and (r["gps_verified"], r["gps_unverified"]) == (0, 1)
        recs = [x for x in Audit(world / "audit.jsonl").records() if x["event"] == "live_audit"]
        assert (recs[0]["gps_verified"], recs[0]["gps_unverified"]) == (0, 1)

    def test_the_blob_url_names_the_did_and_cid_on_the_pds_from_the_did_document(self, world):
        asked = []

        def get(url):
            asked.append(url)
            return jpeg_bytes()
        run(world, [post(1)], images=True, get_bytes=get)
        assert asked == ["https://cdn.example/1/0.jpg", blob_url(1)]

    def test_a_quote_post_with_images_is_audited_for_alt_and_gps(self, world):
        feed = [post(1, alts=("Bird at lat 12.3456 lon -65.4321",), quote=True), post(2, alts=("",), quote=True), post(3, quote=True)]
        r = run(world, feed, images=True, get_bytes=self.blobs(**{blob_url(3): jpeg_bytes(exif_gps=True)}))
        assert {p["uri"]: p["kinds"] for p in r["problems"]} == {
            uri(1): ["coordinates in alt of image 1"], uri(2): ["image without alt text 1"], uri(3): ["GPS tags in a public image (image 1)"]}

    def test_a_plain_quote_post_without_images_is_clean(self, world):     # twin
        p = post(1, alts=(), quote=True)
        p["post"]["embed"] = {"$type": "app.bsky.embed.recordWithMedia#view", "record": {}, "media": {"$type": "app.bsky.embed.external#view"}}
        assert run(world, [p], images=True)["problems"] == []

    @pytest.mark.parametrize("error", [http.client.IncompleteRead(b"ab", 10), http.client.RemoteDisconnected("closed"), ConnectionResetError("reset"), ValueError("bad")])
    def test_a_download_that_fails_in_any_way_is_one_could_not_be_fetched_problem_and_never_a_crash(self, world, error):
        def get(url):
            raise error
        assert run(world, [post(1)], images=True, get_bytes=get)["problems"] == [{"uri": uri(1), "kinds": ["image 1 could not be fetched"]}]

    def test_no_images_flag_downloads_nothing(self, world):
        def boom(url):
            raise AssertionError("downloaded " + url)
        assert run(world, [post(1)], images=False, get_bytes=boom)["problem_posts"] == 0

    def test_an_image_that_cannot_be_fetched_is_a_problem_of_that_post(self, world):
        def get(url):
            raise urllib.error.URLError("down")
        assert run(world, [post(1)], images=True, get_bytes=get)["problems"] == [{"uri": uri(1), "kinds": ["image 1 could not be fetched"]}]

    def test_bytes_that_are_not_an_image_are_reported_as_undecodable(self, world):
        r = run(world, [post(1)], images=True, get_bytes=lambda u: b"<html>")
        assert r["problems"] == [{"uri": uri(1), "kinds": ["image floor: image could not be decoded (image 1)"]}]


class TestFeed:
    def test_reposts_are_skipped_and_the_request_is_an_unauthenticated_get_on_the_public_appview(self, world):
        app = Appview([{"feed": [post(1), {**post(2), "reason": {"$type": "app.bsky.feed.defs#reasonRepost"}}]}])
        posts = social_live_audit.fetch_recent("me.example", 50, get_json=app)
        assert [p["uri"] for p in posts] == [uri(1)]
        [url] = app.urls
        parsed = urllib.parse.urlparse(url)
        assert (parsed.scheme, parsed.netloc, parsed.path) == ("https", "api.bsky.app", "/xrpc/app.bsky.feed.getAuthorFeed")
        q = urllib.parse.parse_qs(parsed.query)
        assert (q["actor"], q["filter"]) == (["me.example"], ["posts_no_replies"])

    def test_pages_are_followed_until_the_limit(self, world):
        app = Appview([{"feed": [post(1), post(2)], "cursor": "c1"}, {"feed": [post(3), post(4)], "cursor": "c2"}])
        posts = social_live_audit.fetch_recent("me.example", 3, get_json=app)
        assert [p["uri"] for p in posts] == [uri(1), uri(2), uri(3)]
        assert "cursor=c1" in app.urls[1] and len(app.urls) == 2

    def test_a_network_error_is_audited_and_raised(self, world):
        audit = Audit(world / "audit.jsonl", run="r1")
        with pytest.raises(social_live_audit.LiveAuditError, match=r"the public AppView could not be read \(URLError: <urlopen error down>\)"):
            social_live_audit.audit_live("me.example", get_json=Appview([urllib.error.URLError("down")]), audit=audit)
        assert [(r["event"], r["reason"]) for r in audit.records()] == [
            ("live_audit_failed", "the public AppView could not be read (URLError: <urlopen error down>)")]


class TestAudit:
    def test_a_run_with_problems_logs_live_audit_and_live_audit_problem_with_uris_not_text(self, world):
        audit = Audit(world / "audit.jsonl", run="r1")
        social_live_audit.audit_live("me.example", images=False, audit=audit,
                                     get_json=Appview([{"feed": [post(1, text="at 12.34567, -65.43217"), post(2)]}]))
        recs = audit.records()
        assert [r["event"] for r in recs] == ["live_audit", "live_audit_problem"]
        assert (recs[0]["checked"], recs[0]["problem_posts"]) == (2, 1)
        assert recs[1]["uris"] == [uri(1)]
        assert recs[1]["reason"] == f"1 public post(s) fail the privacy gates: {uri(1)}"
        assert recs[1]["problems"] == [f"{uri(1)}: coordinates in post text"]
        assert "12.34" not in json.dumps(recs)

    def test_a_clean_run_logs_only_live_audit(self, world):
        audit = Audit(world / "audit.jsonl", run="r1")
        social_live_audit.audit_live("me.example", images=False, audit=audit, get_json=Appview([{"feed": [post(1)]}]))
        assert [r["event"] for r in audit.records()] == ["live_audit"]

    def test_the_problem_event_is_an_error_alert_that_names_the_uris(self, world):
        from bio_automation import social_alerts

        assert ERROR_EVENTS["live_audit_problem"] == "error"
        now = dt.datetime(2026, 10, 23, 21, 40, tzinfo=settings.LOCAL_TZ)
        audit = Audit(world / "audit.jsonl", run="r1", now=lambda: now - dt.timedelta(minutes=5))
        social_live_audit.audit_live("me.example", images=False, audit=audit, get_json=Appview([{"feed": [post(1, text="at 12.34567, -65.43217")]}]))
        alerts, _ = social_alerts.scan(now, audit_path=world / "audit.jsonl", ledger_path=world / "ledger.jsonl", slots=[],
                                       log_file=world / "social.log", taxonomy_file=world / "tax.json", disk_free=lambda root: 10**12)
        [alert] = [a for a in alerts if a.kind == "live_audit_problem"]
        assert alert.level == "error" and uri(1) in alert.message and "12.34" not in alert.message


class TestWeekly:
    def test_last_audit_at_is_the_newest_live_audit_record(self, world):
        a = Audit(world / "audit.jsonl", run="r", now=lambda: dt.datetime(2026, 10, 1, 9, 0, tzinfo=settings.LOCAL_TZ))
        a.log("live_audit", checked=1)
        Audit(world / "audit.jsonl", run="r", now=lambda: dt.datetime(2026, 10, 9, 9, 0, tzinfo=settings.LOCAL_TZ)).log("paused", action="set")
        Audit(world / "audit.jsonl", run="r", now=lambda: dt.datetime(2026, 10, 5, 9, 0, tzinfo=settings.LOCAL_TZ)).log("live_audit", checked=1)
        assert social_live_audit.last_audit_at() == dt.datetime(2026, 10, 5, 9, 0, tzinfo=settings.LOCAL_TZ)

    def test_last_audit_at_is_none_without_a_record(self, world):
        assert social_live_audit.last_audit_at() is None

    def cli(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["bio", *argv])
        return cli.main()

    def test_weekly_skips_when_the_last_audit_is_under_seven_days_old_and_runs_otherwise(self, world, monkeypatch, capsys):
        calls = []
        monkeypatch.setattr(social_live_audit, "_get_json", lambda url: calls.append(url) or {"feed": [post(1)]})
        Audit().log("live_audit", checked=1, problem_posts=0)          # now, clean
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 0
        assert capsys.readouterr().out.startswith("skipped: the last clean live audit was ") and calls == []
        old = dt.datetime.now(settings.LOCAL_TZ) - dt.timedelta(days=8)
        (world / "audit.jsonl").write_text(json.dumps({"at": old.isoformat(timespec="seconds"), "run": "x", "event": "live_audit"}) + "\n")
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 0
        assert len(calls) == 1

    def test_exit_codes_clean_problems_and_unreadable_appview(self, world, monkeypatch, capsys):
        feed = {"feed": [post(1)]}
        monkeypatch.setattr(social_live_audit, "_get_json", lambda url: feed)
        assert self.cli(monkeypatch, "social-audit-live", "--no-images") == 0
        assert "checked 1 public posts of example.bsky.social; 1 had no draft on record" in capsys.readouterr().out
        feed["feed"] = [post(1, text="at 12.34567, -65.43217")]
        assert self.cli(monkeypatch, "social-audit-live", "--no-images") == 1
        assert capsys.readouterr().out.splitlines()[2] == f"PROBLEM {uri(1)}: coordinates in post text"

        def down(url):
            raise urllib.error.URLError("down")
        monkeypatch.setattr(social_live_audit, "_get_json", down)
        assert self.cli(monkeypatch, "social-audit-live", "--no-images") == 2
        assert capsys.readouterr().err.startswith("bio social-audit-live: the public AppView could not be read")


class TestWeeklyRepeatsWhileThereAreProblems:
    def cli(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["bio", *argv])
        return cli.main()

    def test_a_run_with_problems_does_not_silence_the_next_evening(self, world, monkeypatch, capsys):
        calls = []
        feed = {"feed": [post(1, text="at 12.34567, -65.43217")]}
        monkeypatch.setattr(social_live_audit, "_get_json", lambda url: calls.append(url) or feed)
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 1             # first run: a problem
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 1             # the next evening: audited again, not skipped
        assert len(calls) == 2
        assert [r["problem_posts"] for r in Audit().records() if r["event"] == "live_audit"] == [1, 1]
        assert len([r for r in Audit().records() if r["event"] == "live_audit_problem"]) == 2            # so the error alert comes back too

    def test_after_a_clean_run_the_weekly_call_skips_again(self, world, monkeypatch, capsys):     # twin
        calls = []
        feed = {"feed": [post(1)]}
        monkeypatch.setattr(social_live_audit, "_get_json", lambda url: calls.append(url) or feed)
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 0
        assert self.cli(monkeypatch, "social-audit-live", "--weekly", "--no-images") == 0
        assert len(calls) == 1

    def test_last_audit_at_can_ignore_runs_that_found_problems(self, world):
        for day, problems in ((1, 0), (5, 2)):
            Audit(world / "audit.jsonl", run="r", now=lambda d=day: dt.datetime(2026, 10, d, 9, 0, tzinfo=settings.LOCAL_TZ)).log("live_audit", problem_posts=problems)
        assert social_live_audit.last_audit_at() == dt.datetime(2026, 10, 5, 9, 0, tzinfo=settings.LOCAL_TZ)
        assert social_live_audit.last_audit_at(clean_only=True) == dt.datetime(2026, 10, 1, 9, 0, tzinfo=settings.LOCAL_TZ)


class TestUnexpectedErrors:
    def cli(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["bio", *argv])
        return cli.main()

    def test_a_feed_that_raises_something_unexpected_exits_2_with_a_failed_record_not_1(self, world, monkeypatch, capsys):
        def odd(url):
            raise RuntimeError("surprise")
        monkeypatch.setattr(social_live_audit, "_get_json", odd)
        assert self.cli(monkeypatch, "social-audit-live", "--no-images") == 2
        assert capsys.readouterr().err.startswith("bio social-audit-live: unexpected error reading the feed (RuntimeError: surprise)")
        assert [r["reason"] for r in Audit().records() if r["event"] == "live_audit_failed"] == ["RuntimeError: surprise"]

    def test_a_torn_feed_response_is_the_could_not_read_the_appview_exit_2(self, world, monkeypatch, capsys):
        def torn(url):
            raise http.client.IncompleteRead(b"x", 5)
        monkeypatch.setattr(social_live_audit, "_get_json", torn)
        assert self.cli(monkeypatch, "social-audit-live", "--no-images") == 2
        assert "the public AppView could not be read (IncompleteRead" in capsys.readouterr().err

    def test_the_report_says_what_the_gps_check_covered(self, world, monkeypatch, capsys):
        feed = {"feed": [post(1)]}
        monkeypatch.setattr(social_live_audit, "_get_json", lambda url: PLC_DOC if url.startswith(settings.PLC_URL) else feed)
        monkeypatch.setattr(social_live_audit, "_get_bytes", lambda url: jpeg_bytes())
        assert self.cli(monkeypatch, "social-audit-live") == 0
        assert capsys.readouterr().out.splitlines()[1] == "GPS tags: 1 original image blob(s) checked"
        monkeypatch.setattr(social_live_audit, "_get_bytes", lambda url: (_ for _ in ()).throw(OSError("no")) if "getBlob" in url else jpeg_bytes())
        assert self.cli(monkeypatch, "social-audit-live") == 0
        assert capsys.readouterr().out.splitlines()[1] == ("GPS tags: 0 original image blob(s) checked; 1 image(s) NOT VERIFIABLE "
                                                            "(blob not fetched from the PDS): no assurance about GPS for those")
