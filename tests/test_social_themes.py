"""Themed Bluesky posts end to end: the text, the group draft, drafting ahead, the ledger and the day it is posted."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pytest
from test_social import PRINT, TZ, RecordingDescribe, RecordingFetch, RecordingSend, jpeg, labels, no_model, observation, rules_with

from bio_automation import social, themes
from bio_automation.social import Ledger, SkipObservation, SocialError
from bio_automation.themes import Theme

MONDAY = Theme("mushroom-monday", "Mushroom Monday", 0, ("MushroomMonday", "mushrooms"), "🍄", {"order": ["Agaricales"]}, {})
TUESDAY = Theme("texture-tuesday", "Texture Tuesday", 1, ("TextureTuesday",), "", {"order": ["Agaricales"]}, {})
WED = dt.date(2026, 10, 21)
MON = dt.date(2026, 10, 26)


def entry(species, order="Agaricales"):
    lineage = [{"rank": "kingdom", "name": "Fungi", "common_name": None}, {"rank": "order", "name": order, "common_name": None}]
    return {"taxon": {"id": 1, "name": species, "common_name": None, "rank": "species", "iconic": "Fungi", "lineage": lineage},
            "grade": "research", "observed_on": "2025-10-01", "geoprivacy": None, "fetched": time.time()}


class World:
    """n observations 1..n: one gallery photo each, a taxonomy entry, an iNat observation; `ledger`, `drafts`, `fetch`, `describe` are stubs/files."""

    def __init__(self, tmp_path, n=8, casual=()):
        self.tmp = tmp_path
        self.gallery = tmp_path / "gallery"
        self.index = {"files": {}}
        for o in range(1, n + 1):
            rel = f"2025/2025-10/2025-10-{o:02d}_DSC_{o}.jpg"
            jpeg(self.gallery / rel, size=(300, 200))
            self.index["files"][rel] = o
        self.taxonomy = {o: entry(f"Amanita sp{o}") for o in range(1, n + 1)}
        self.obs = {o: observation(o, grade="casual" if o in casual else "research", iconic="Fungi", name=f"Amanita sp{o}",
                                   common=f"fly agaric {o}", place="Bucks County, PA, USA") for o in range(1, n + 1)}
        self.fetch = RecordingFetch(self.obs)
        self.describe = RecordingDescribe()
        self.ledger = Ledger(tmp_path / "ledger.jsonl")
        self.drafts = tmp_path / "drafts"

    def draft(self, themes_list, today, days_ahead, **kw):
        return social.draft_themes(self.index, self.ledger, self.taxonomy, themes_list, today=today, days_ahead=days_ahead,
                                   gallery_dir=self.gallery, drafts_dir=self.drafts, model="vis:1", fetch=self.fetch, read_label=labels,
                                   describe_fn=self.describe, pick_model=kw.pop("pick_model", no_model), style_source=lambda: None,
                                   prints=kw.pop("prints", {}), **kw)


def items(*spec):
    return [{"common": c, "name": n, "print": p} for c, n, p in spec]


class TestGroupText:
    def test_title_names_print_and_hashtags_in_that_order_with_links_and_tags_as_facets(self):
        segs = social.group_text_segments(MONDAY, items(("fly agaric", "Amanita muscaria", None), ("Boletes", "Boletales", PRINT)))
        assert social.segments_text(segs) == ("🍄 Mushroom Monday\nFly agaric (Amanita muscaria)\nBoletes (Boletales)\nPrints: " + PRINT +
                                              "\n#MushroomMonday #mushrooms #macrophotography #nature #inaturalist")
        assert [s for s in segs if s[0] == "link"] == [["link", PRINT, PRINT]]
        assert [s[1] for s in segs if s[0] == "tag"] == ["#MushroomMonday", "#mushrooms", "#macrophotography", "#nature", "#inaturalist"]

    def test_over_300_characters_the_scientific_names_go_before_the_print_link(self):
        long = [(f"Common name number {i} of the post" , f"Genus speciesname{i} variety long", PRINT if i == 1 else None) for i in range(1, 5)]
        text = social.segments_text(social.group_text_segments(MONDAY, items(*long)))
        assert len(text) <= 300 and "Genus speciesname" not in text and "Prints: " + PRINT in text
        assert text.startswith("🍄 Mushroom Monday\nCommon name number 1 of the post\nCommon name number 2")

    def test_when_even_that_is_too_long_the_print_goes_then_tags_then_the_names(self):
        big = [("X" * 70, "Y" * 10, PRINT) for _ in range(4)]
        text = social.segments_text(social.group_text_segments(MONDAY, items(*big)))
        assert text == "🍄 Mushroom Monday\n4 finds\n#MushroomMonday"
        mid = [("X" * 58, "Y", PRINT) for _ in range(4)]
        t2 = social.segments_text(social.group_text_segments(MONDAY, items(*mid)))
        assert t2 == "🍄 Mushroom Monday\n" + ("X" * 58 + "\n") * 4 + "#MushroomMonday #mushrooms #macrophotography" and len(t2) == 298
        near = [("X" * 50, "Y", PRINT) for _ in range(4)]
        t3 = social.segments_text(social.group_text_segments(MONDAY, items(*near)))
        assert "Prints:" not in t3 and t3.endswith("#nature #inaturalist") and len(t3) == 287                  # all five tags fit once the print link is gone

    def test_a_title_that_cannot_fit_in_any_form_is_an_error(self):
        huge = MONDAY._replace(name="N" * 400)
        with pytest.raises(SocialError, match=r"^N{400}: the post text is over 300 characters even in its shortest form$"):
            social.group_text_segments(huge, items(("a", "b", None)))


class TestGroupDraft:
    def make(self, w, ids, theme=MONDAY, prints=None):
        rels_of = {o: sorted(r for r, ob in w.index["files"].items() if ob == o) for o in ids}
        return social.make_group_draft(ids, rels_of, theme, MON, gallery_dir=w.gallery, drafts_dir=w.drafts, model="vis:1", fetch=w.fetch,
                                       read_label=labels, describe_fn=w.describe, style=None, prints=prints)

    def test_the_first_photo_of_each_observation_becomes_one_post_with_its_own_alt_text(self, tmp_path):
        w = World(tmp_path)
        draft, skipped = self.make(w, [1, 2, 3, 4, 5, 6])
        assert skipped == {} and draft["draft_id"] == "2026-10-26-mushroom-monday"
        assert (draft["month"], draft["for_date"], draft["theme"], draft["obs_id"], draft["obs_ids"]) == ("2026-10", "2026-10-26", "mushroom-monday", 1, [1, 2, 3, 4])
        assert [(i["file"], i["obs_id"], i["alt"]) for i in draft["images"]] == [
            (f"{n}.jpg", n, f"alt for title 2025-10-{n:02d}_DSC_{n}") for n in range(1, 5)]
        assert social.segments_text(draft["text"]) == ("🍄 Mushroom Monday\nFly agaric 1 (Amanita sp1)\nFly agaric 2 (Amanita sp2)\n"
                                                       "Fly agaric 3 (Amanita sp3)\nFly agaric 4 (Amanita sp4)\n#MushroomMonday #mushrooms #macrophotography #nature #inaturalist")
        out = w.drafts / "2026-10-26-mushroom-monday"
        assert sorted(p.name for p in out.iterdir()) == ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "draft.json"]
        assert json.loads((out / "draft.json").read_text(encoding="utf-8")) == draft
        assert [p.name for p in w.drafts.iterdir()] == ["2026-10-26-mushroom-monday"]                   # no scratch folder left behind
        assert (w.fetch.calls, len(w.describe.calls)) == ([1, 2, 3, 4], 4)

    def test_a_casual_observation_is_skipped_for_the_next_one_in_the_reserve(self, tmp_path):
        w = World(tmp_path, casual=(2,))
        draft, skipped = self.make(w, [1, 2, 3, 4, 5, 6])
        assert skipped == {2: "observation 2 is casual"} and draft["obs_ids"] == [1, 3, 4, 5]
        assert w.fetch.calls == [1, 2, 3, 4, 5]

    def test_an_observation_with_a_print_gets_the_link_in_the_text(self, tmp_path):
        w = World(tmp_path)
        draft, _ = self.make(w, [1, 2], prints={2: PRINT})
        assert "Prints: " + PRINT in social.segments_text(draft["text"])

    def test_fewer_postable_observations_than_the_minimum_raises_and_leaves_no_folder(self, tmp_path):
        w = World(tmp_path, n=3, casual=(2, 3))
        with pytest.raises(SkipObservation, match=r"^Mushroom Monday on 2026-10-26: 1 postable observation\(s\) of the 2 needed$"):
            self.make(w, [1, 2, 3])
        assert not (w.drafts / "2026-10-26-mushroom-monday").exists() and not (w.drafts / "_scratch").exists()


class TestDraftThemes:
    def test_the_next_mondays_group_is_drafted_with_its_ledger_fields_and_a_rerun_does_nothing(self, tmp_path):
        w = World(tmp_path)
        report = w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        assert report == {"drafted": ["2026-10-26-mushroom-monday"], "skipped": {}, "short": {}}
        rec = [r for r in w.ledger.records() if r["event"] == "drafted"][0]
        assert (rec["draft_id"], rec["month"], rec["kind"], rec["theme"], rec["for_date"], rec["images"]) == (
            "2026-10-26-mushroom-monday", "2026-10", "theme", "mushroom-monday", "2026-10-26", 4)
        assert len(rec["obs_ids"]) == 4 and rec["obs_id"] == rec["obs_ids"][0]
        before = w.ledger.records()
        assert w.draft([MONDAY], WED, 7) == {"drafted": [], "skipped": {}, "short": {}}                # no model is loaded for nothing to do
        assert w.ledger.records() == before

    def test_two_weeks_ahead_gives_two_groups_with_no_observation_in_both(self, tmp_path):
        w = World(tmp_path)
        report = w.draft([MONDAY], WED, 14, pick_model=lambda: "vis:1")
        assert report["drafted"] == ["2026-10-26-mushroom-monday", "2026-11-02-mushroom-monday"]
        ids = [r["obs_ids"] for r in w.ledger.records()]
        assert len(set(ids[0]) | set(ids[1])) == 8

    def test_posted_rejected_and_pending_observations_are_left_out(self, tmp_path):
        w = World(tmp_path)
        w.ledger.append("drafted", "2025-10-obs1", obs_id=1, month="2025-10")
        w.ledger.append("posted", "2025-10-obs1", obs_id=1, month="2025-10")
        w.ledger.append("drafted", "2026-10-obs2", obs_id=2, month="2026-10")                          # drafted this month: taken
        report = w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        got = [r for r in w.ledger.records() if r.get("theme")][0]["obs_ids"]
        assert report["drafted"] == ["2026-10-26-mushroom-monday"] and not {1, 2} & set(got)

    def test_a_missed_themed_draft_gives_its_observations_back_and_a_coming_one_keeps_them(self, tmp_path):
        past = World(tmp_path / "a", n=4)
        past.ledger.append("drafted", "2026-10-19-mushroom-monday", obs_id=1, obs_ids=[1, 2, 3, 4], month="2026-10", theme="mushroom-monday",
                           for_date="2026-10-19")
        past.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        assert sorted([r for r in past.ledger.records() if r["draft_id"] == "2026-10-26-mushroom-monday"][0]["obs_ids"]) == [1, 2, 3, 4]
        coming = World(tmp_path / "b", n=4)
        coming.ledger.append("drafted", "2026-10-28-texture-tuesday", obs_id=1, obs_ids=[1, 2, 3], month="2026-10", theme="texture-tuesday",
                             for_date="2026-10-28")
        report = coming.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        assert report == {"drafted": [], "skipped": {}, "short": {"2026-10-26": "Mushroom Monday: fewer than 2 observations left"}}

    def test_too_few_observations_is_reported_and_no_model_is_loaded(self, tmp_path):
        w = World(tmp_path, n=1)
        assert w.draft([MONDAY], WED, 7) == {"drafted": [], "skipped": {}, "short": {"2026-10-26": "Mushroom Monday: fewer than 2 observations left"}}

    def test_a_day_without_a_theme_drafts_nothing(self, tmp_path):
        w = World(tmp_path)
        assert w.draft([TUESDAY], dt.date(2026, 10, 26), 1) == {"drafted": [], "skipped": {}, "short": {}}                # Monday: no Tuesday theme
        assert w.draft([TUESDAY._replace(enabled=False)], dt.date(2026, 10, 27), 1) == {"drafted": [], "skipped": {}, "short": {}}   # a disabled one
        assert w.ledger.records() == []
        assert w.draft([TUESDAY], dt.date(2026, 10, 27), 1, pick_model=lambda: "vis:1")["drafted"] == ["2026-10-27-texture-tuesday"]   # twin: on its day


class TestThemedPosting:
    NOW_MON = dt.datetime(2026, 10, 26, 8, 0, tzinfo=TZ)

    def themed(self, tmp_path, theme=MONDAY, day=MON):
        w = World(tmp_path)
        w.draft([theme], day, 1, pick_model=lambda: "vis:1")
        return w

    def test_todays_themed_draft_goes_first_even_with_an_older_approved_single_and_the_single_goes_next(self, tmp_path):
        w = self.themed(tmp_path)
        single = w.drafts / "2026-10-obs99"
        jpeg(single / "1.jpg", size=(300, 200))
        (single / "draft.json").write_text(json.dumps({"draft_id": "2026-10-obs99", "month": "2026-10", "obs_id": 99, "text": [["text", "single"]],
                                                       "images": [{"file": "1.jpg", "alt": "a", "width": 300, "height": 200}]}), encoding="utf-8")
        w.ledger.append("drafted", "2026-10-obs99", obs_id=99, month="2026-10", now=self.NOW_MON - dt.timedelta(days=2))
        social.approve(w.ledger, w.drafts, "2026-10-obs99")
        send = RecordingSend()
        first = social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send)
        assert (first["draft_id"], first["status"]) == ("2026-10-26-mushroom-monday", "posted")
        assert len(send.calls[0]["images"]) == 4 and send.calls[0]["text"].startswith("🍄 Mushroom Monday\n")
        second = social.post_next(w.ledger, w.drafts, now=self.NOW_MON + dt.timedelta(hours=4), live=True, send=send)
        assert second["draft_id"] == "2026-10-obs99"

    def test_a_coming_days_draft_waits_and_a_past_days_draft_is_never_posted(self, tmp_path):
        w = self.themed(tmp_path)
        send = RecordingSend()
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON - dt.timedelta(days=1), live=True, send=send) == {"status": "nothing approved", "month": "2026-10"}
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON + dt.timedelta(days=1), live=True, send=send) == {"status": "nothing approved", "month": "2026-10"}
        assert send.calls == []
        social.approve(w.ledger, w.drafts, "2026-10-26-mushroom-monday")                                  # approved early: still only for its own day
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON - dt.timedelta(days=1), live=True, send=send, auto_approve=False) == {
            "status": "nothing approved", "month": "2026-10"}
        assert send.calls == []
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send)["status"] == "posted"

    def test_with_auto_approval_off_an_unapproved_themed_draft_waits_and_an_approved_one_posts(self, tmp_path):
        w = self.themed(tmp_path)
        send = RecordingSend()
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send, auto_approve=False)["status"] == "nothing approved"
        social.approve(w.ledger, w.drafts, "2026-10-26-mushroom-monday")
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send, auto_approve=False)["status"] == "posted"

    def test_a_themed_draft_is_the_scheduled_post_of_its_day_and_counts_in_the_cap(self, tmp_path):
        w = self.themed(tmp_path)
        send = RecordingSend()
        social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send, posts_per_day=1)
        assert social.post_next(w.ledger, w.drafts, now=self.NOW_MON, live=True, send=send, posts_per_day=1) == {
            "status": "done for today", "posted_today": ["2026-10-26-mushroom-monday"]}

    def test_themed_drafts_do_not_count_as_pending_singles_and_their_observations_are_not_drafted_as_singles(self, tmp_path):
        w = World(tmp_path, n=6)
        w.draft([MONDAY], MON, 1, pick_model=lambda: "vis:1")
        used = [r for r in w.ledger.records() if r.get("theme")][0]["obs_ids"]
        fetch = RecordingFetch({o: w.obs[o] for o in w.obs})
        report = social.draft_month(w.index, w.ledger, today=MON, gallery_dir=w.gallery, drafts_dir=w.drafts, keep_pending=1, model="vis:1",
                                    fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(), pick_model=no_model, style_source=lambda: None, prints={})
        assert report["pending"] == 0 and len(report["drafted"]) == 1
        assert not set(fetch.calls) & set(used)


class TestCli:
    def setup(self, tmp_path, monkeypatch):
        from bio_automation import settings

        w = World(tmp_path, n=4)
        (tmp_path / "index.json").write_text(json.dumps({"files": w.index["files"], "unresolved": {}}), encoding="utf-8")
        (tmp_path / "tax.json").write_text(json.dumps({str(o): e for o, e in w.taxonomy.items()}), encoding="utf-8")
        (tmp_path / "themes.toml").write_text('[[theme]]\nkey = "mushroom-monday"\nname = "Mushroom Monday"\nweekday = "monday"\n'
                                              'tags = ["MushroomMonday", "mushrooms"]\n[theme.include]\norder = ["Agaricales"]\n', encoding="utf-8")
        monkeypatch.setattr(settings, "OBS_TAXONOMY_FILE", tmp_path / "tax.json")
        monkeypatch.setattr(settings, "INAT_TAXON_CACHE", tmp_path / "taxa.sqlite3")
        monkeypatch.setattr(settings, "SOCIAL_LEDGER", tmp_path / "ledger.jsonl")
        monkeypatch.setattr(settings, "SOCIAL_THEMES_FILE", tmp_path / "themes.toml")
        return settings

    def test_social_themes_shows_the_calendar_with_what_is_left_and_the_plan_for_the_next_days(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.setup(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.argv", ["bio", "social-themes", "--days", "2", "--date", "2026-10-25", "--index", str(tmp_path / "index.json")])
        assert cli.main() == 0
        out = capsys.readouterr().out.splitlines()
        assert out[0] == f"calendar (file {tmp_path / 'themes.toml'}):"
        assert out[1] == "  monday    Mushroom Monday        on      4 observations left  #MushroomMonday #mushrooms"
        assert (out[2], out[3]) == ("", "next 2 days:")
        assert out[4] == "  2026-10-25 sun  usual single observation"
        head, _, names = out[5].partition(": ")
        assert head == "  2026-10-26 mon  Mushroom Monday" and sorted(names.split(", ")) == [f"Amanita sp{o}" for o in (1, 2, 3, 4)]

    def test_write_defaults_writes_the_builtin_calendar_once_and_refuses_to_overwrite(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        settings = self.setup(tmp_path, monkeypatch)
        (tmp_path / "themes.toml").unlink()
        monkeypatch.setattr("sys.argv", ["bio", "social-themes", "--write-defaults"])
        assert cli.main() == 0
        assert (tmp_path / "themes.toml").read_text(encoding="utf-8") == themes.themes_toml()
        assert capsys.readouterr().out == f"wrote {settings.SOCIAL_THEMES_FILE}\n"
        assert cli.main() == 1
        assert capsys.readouterr().err == f"bio social-themes: {tmp_path / 'themes.toml'} exists: edit it, or move it away first\n"

    def test_a_broken_themes_file_is_a_one_line_error(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli

        self.setup(tmp_path, monkeypatch)
        (tmp_path / "themes.toml").write_text('[[theme]]\nkey = "a"\nname = "A"\nweekday = "funday"\n[theme.include]\nclass = ["X"]\n', encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["bio", "social-themes", "--index", str(tmp_path / "index.json")])
        assert cli.main() == 1
        assert capsys.readouterr().err.startswith(f"bio social-themes: {tmp_path / 'themes.toml'}: theme 'a': weekday must be one of monday")


class TestRedraftUnderTheSameId:
    def ledger(self, tmp_path):
        ledger = Ledger(tmp_path / "l.jsonl")
        ledger.append("drafted", "2026-10-26-mushroom-monday", obs_id=1, obs_ids=[1, 2, 3, 4], month="2026-10", theme="mushroom-monday", for_date="2026-10-26")
        ledger.append("approved", "2026-10-26-mushroom-monday", obs_id=1, month="2026-10", hash="old")
        ledger.append("withdrawn", "2026-10-26-mushroom-monday", obs_id=1, month="2026-10", reason="small")
        ledger.append("drafted", "2026-10-26-mushroom-monday", obs_id=5, obs_ids=[5, 6, 7, 8], month="2026-10", theme="mushroom-monday", for_date="2026-10-26")
        return ledger

    def test_a_redraft_replaces_the_observations_and_forgets_the_old_approval(self, tmp_path):
        v = self.ledger(tmp_path).drafts()["2026-10-26-mushroom-monday"]
        assert (v["state"], v["obs_id"], v["obs_ids"], v["for_date"], v["theme"]) == ("drafted", 5, [5, 6, 7, 8], "2026-10-26", "mushroom-monday")
        assert v["approved_hash"] is None and "approved_seq" not in v

    def test_the_withdrawn_observations_are_free_and_the_redrafted_ones_are_taken(self, tmp_path):
        v = self.ledger(tmp_path).drafts()["2026-10-26-mushroom-monday"]
        taken = {o for o in social._obs_of(v)} if social._taken(v, "2026-10", dt.date(2026, 10, 22), rules_with()) else set()
        assert taken == {5, 6, 7, 8}

    def test_a_draft_that_was_never_redrafted_keeps_its_first_records_fields_and_its_approval(self, tmp_path):
        ledger = Ledger(tmp_path / "l.jsonl")
        ledger.append("drafted", "2026-10-26-mushroom-monday", obs_id=1, obs_ids=[1, 2], month="2026-10", theme="mushroom-monday", for_date="2026-10-26")
        ledger.append("approved", "2026-10-26-mushroom-monday", obs_id=1, month="2026-10", hash="h1")
        v = ledger.drafts()["2026-10-26-mushroom-monday"]
        assert (v["state"], v["obs_ids"], v["approved_hash"], "approved_seq" in v) == ("approved", [1, 2], "h1", True)


class TestThemedDraftsTakeTheLockAndHonourTheWithdrawalCap:
    def test_the_themed_draft_append_gives_up_when_a_post_holds_the_lock_and_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(social, "LEDGER_LOCK_WAIT_S", 0.0)
        w = World(tmp_path)
        with social.ledger_lock(w.ledger.path):
            with pytest.raises(SocialError, match="another social-post run is posting"):
                w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        assert w.ledger.records() == []

    def test_positive_twin_without_a_running_post_the_group_is_recorded(self, tmp_path):
        w = World(tmp_path)
        assert w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")["drafted"] == ["2026-10-26-mushroom-monday"]
        assert [r["event"] for r in w.ledger.records()] == ["drafted"]

    def withdraw(self, w, obs, times, *, hook="live_recheck"):
        for k in range(times):
            did = f"2026-09-obs{obs}-try{k}"
            w.ledger.append("drafted", did, obs_id=obs, month="2026-09")
            w.ledger.append("withdrawn", did, obs_id=obs, month="2026-09", reason=f"gate said no {k}", **({"hook": hook} if hook else {}))

    def test_an_observation_withdrawn_three_times_is_left_out_of_the_group_and_audited(self, tmp_path):
        from bio_automation.social_rules import Audit

        w = World(tmp_path, n=5)
        self.withdraw(w, 1, 3)
        audit = Audit(tmp_path / "audit.jsonl")
        w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1", audit=audit)
        (rec,) = [r for r in w.ledger.records() if r["draft_id"] == "2026-10-26-mushroom-monday"]
        assert sorted(rec["obs_ids"]) == [2, 3, 4, 5]
        assert [(r["obs_id"], r["reason"]) for r in audit.records() if r["event"] == "draft_skipped"] == [(1, "withdrawn 3 times: gate said no 2")]

    def test_positive_twin_two_withdrawals_leave_the_observation_in_play(self, tmp_path):
        w = World(tmp_path, n=4)
        self.withdraw(w, 1, 2)
        w.draft([MONDAY], WED, 7, pick_model=lambda: "vis:1")
        (rec,) = [r for r in w.ledger.records() if r["draft_id"] == "2026-10-26-mushroom-monday"]
        assert sorted(rec["obs_ids"]) == [1, 2, 3, 4]
