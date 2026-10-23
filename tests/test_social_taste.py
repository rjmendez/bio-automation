"""social_taste: variety, seasonal boost and the photo cache cap, each through the hook framework with the real ledger and audit log."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest
from test_social import NOW, rules_with

from bio_automation import settings, social_hooks, social_taste
from bio_automation.social import Ledger
from bio_automation.social_rules import Audit

TODAY = dt.date(2026, 10, 23)


def entry(genus=None, observed_on=None, species=None):
    tx = {"name": species or genus, "rank": "species" if species else "genus", "lineage": []}
    if genus:
        tx["lineage"] = [{"rank": "genus", "name": genus}]
    return {"taxon": tx, "observed_on": observed_on}


def write_taxonomy(entries: dict):
    path = Path(settings.OBS_TAXONOMY_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in entries.items()}), encoding="utf-8")


def posted(ledger, draft_id, obs_ids):
    ledger.append("drafted", draft_id, obs_ids=obs_ids, obs_id=obs_ids[0], now=NOW)
    ledger.append("posted", draft_id, now=NOW)


def order(tmp_path, candidates, *, rules=None, ledger=None, kind="single", taxonomy=None, today=TODAY):
    social_taste.register_hooks()
    ctx = {"candidates": candidates, "kind": kind, "taxonomy": taxonomy, "ledger": ledger or Ledger(tmp_path / "l.jsonl"), "today": today,
           "rules": rules or rules_with()}
    social_hooks.run("order", ctx, rules=ctx["rules"], audit=Audit())
    return ctx["candidates"]


class TestVariety:
    def test_candidates_of_a_recent_genus_go_to_the_end_in_their_old_order(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 3: entry("Amanita"), 4: entry("Russula"), 9: entry("Amanita")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2, 3, 4], ledger=ledger) == [2, 4, 1, 3]

    def test_positive_twin_without_a_recent_post_of_the_genus_the_order_is_unchanged(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 3: entry("Amanita"), 4: entry("Russula"), 9: entry("Cortinarius")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2, 3, 4], ledger=ledger) == [1, 2, 3, 4]

    def test_only_the_last_n_posted_drafts_count(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 5: entry("Amanita"), 6: entry("Russula"), 7: entry("Cortinarius")})
        ledger = Ledger(tmp_path / "l.jsonl")
        for did, o in (("d5", 5), ("d6", 6), ("d7", 7)):
            posted(ledger, did, [o])
        assert order(tmp_path, [1, 2], ledger=ledger, rules=rules_with(variety=2)) == [1, 2]       # Amanita was 3 posts ago
        assert order(tmp_path, [1, 2], ledger=ledger, rules=rules_with(variety=3)) == [2, 1]       # within the last 3

    def test_every_observation_of_a_themed_post_counts_and_a_theme_list_is_reordered_too(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 3: entry("Russula"), 8: entry("Boletus"), 9: entry("Cortinarius")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "2026-10-19-mushroom-monday", [8, 9])
        assert order(tmp_path, [1, 2, 3], ledger=ledger, kind="theme:mushroom-monday") == [1, 3, 2]

    def test_unposted_drafts_do_not_count(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 9: entry("Amanita")})
        ledger = Ledger(tmp_path / "l.jsonl")
        ledger.append("drafted", "d9", obs_ids=[9], obs_id=9, now=NOW)
        assert order(tmp_path, [1, 2], ledger=ledger) == [1, 2]

    def test_a_missing_taxonomy_file_changes_nothing(self, tmp_path):
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert not Path(settings.OBS_TAXONOMY_FILE).exists()
        assert order(tmp_path, [3, 1, 2], ledger=ledger) == [3, 1, 2]
        assert Audit().records() == []                                                               # silent: nothing audited

    def test_an_empty_or_broken_taxonomy_changes_nothing(self, tmp_path):
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        write_taxonomy({})
        assert order(tmp_path, [3, 1, 2], ledger=ledger) == [3, 1, 2]
        Path(settings.OBS_TAXONOMY_FILE).write_text("{not json", encoding="utf-8")
        assert order(tmp_path, [3, 1, 2], ledger=ledger) == [3, 1, 2]

    def test_candidates_without_a_known_genus_keep_their_place_at_the_front(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 9: entry("Amanita")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2, 3], ledger=ledger) == [2, 3, 1]                                 # 2 and 3 are not in the taxonomy

    def test_a_species_without_lineage_falls_back_to_the_first_word_of_its_name(self, tmp_path):
        write_taxonomy({1: entry(species="Amanita muscaria"), 2: entry("Boletus"), 9: entry(species="Amanita phalloides")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2], ledger=ledger) == [2, 1]

    def test_the_context_taxonomy_is_used_when_the_hook_is_given_one(self, tmp_path):
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        tax = {1: entry("Amanita"), 2: entry("Boletus"), 9: entry("Amanita")}
        assert order(tmp_path, [1, 2], ledger=ledger, taxonomy=tax) == [2, 1]

    def test_switched_off_the_rule_leaves_the_order_alone(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Boletus"), 9: entry("Amanita")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2], ledger=ledger, rules=rules_with(variety=False)) == [1, 2]

    def test_nothing_is_removed(self, tmp_path):
        write_taxonomy({1: entry("Amanita"), 2: entry("Amanita"), 9: entry("Amanita")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [1, 2], ledger=ledger) == [1, 2]


class TestSeasonalBoost:
    ON = {"seasonal_boost": True, "variety": False}

    def boosted(self, tmp_path, dates, today=TODAY):
        write_taxonomy({o: entry("G", d) for o, d in dates.items()})
        return order(tmp_path, list(dates), rules=rules_with(**self.ON), today=today)

    def test_the_rule_is_off_by_default(self, tmp_path):
        write_taxonomy({1: entry("G", "2020-04-01"), 2: entry("G", "2020-10-23")})
        assert order(tmp_path, [1, 2], rules=rules_with(variety=False)) == [1, 2]

    def test_in_season_observations_of_any_year_go_to_the_front_in_their_old_order(self, tmp_path):
        assert self.boosted(tmp_path, {1: "2020-04-01", 2: "2019-10-20", 3: "2022-01-01", 4: "2023-10-30"}) == [2, 4, 1, 3]

    def test_fifteen_days_either_side_is_in_and_sixteen_is_out(self, tmp_path):
        got = self.boosted(tmp_path, {1: "2021-09-01", 2: "2021-10-08", 3: "2021-11-07", 4: "2021-11-08", 5: "2021-10-07"})
        assert got == [2, 3, 1, 4, 5]                      # 10-08 is 15 days before, 11-07 15 after; 10-07 and 11-08 are 16

    def test_the_season_wraps_around_the_year_end(self, tmp_path):
        today = dt.date(2027, 1, 5)
        got = self.boosted(tmp_path, {1: "2020-07-01", 2: "2020-12-22", 3: "2020-12-20", 4: "2019-01-19"}, today=today)
        assert got == [2, 4, 1, 3]                         # 12-22 is 14 days before 01-05; 12-20 is 16; 01-19 is 14 after

    def test_missing_or_unreadable_dates_stay_behind(self, tmp_path):
        write_taxonomy({1: entry("G", None), 2: entry("G", "garbage"), 3: entry("G", "2020-10-23")})
        assert order(tmp_path, [1, 2, 3, 4], rules=rules_with(**self.ON)) == [3, 1, 2, 4]

    def test_a_29_february_observation_is_read_without_error(self, tmp_path):
        assert self.boosted(tmp_path, {1: "2020-10-01", 2: "2020-02-29"}, today=dt.date(2027, 3, 5)) == [2, 1]

    def test_a_missing_or_empty_taxonomy_changes_nothing(self, tmp_path):
        assert order(tmp_path, [3, 1, 2], rules=rules_with(**self.ON)) == [3, 1, 2]
        write_taxonomy({})
        assert order(tmp_path, [3, 1, 2], rules=rules_with(**self.ON)) == [3, 1, 2]

    def test_variety_runs_last_so_a_recently_posted_genus_stays_demoted_even_in_season(self, tmp_path):
        write_taxonomy({1: entry("Amanita", "2020-10-23"), 2: entry("Boletus", "2020-04-01"), 9: entry("Amanita", "2020-04-01")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [2, 1], ledger=ledger, rules=rules_with(seasonal_boost=True)) == [2, 1]
        assert [h.name for h in social_hooks.hooks("order")] == ["seasonal_boost", "variety"]

    def test_positive_twin_seasonal_still_leads_among_genera_variety_does_not_demote(self, tmp_path):
        write_taxonomy({1: entry("Russula", "2020-10-23"), 2: entry("Boletus", "2020-04-01"), 9: entry("Amanita", "2020-04-01")})
        ledger = Ledger(tmp_path / "l.jsonl")
        posted(ledger, "d9", [9])
        assert order(tmp_path, [2, 1], ledger=ledger, rules=rules_with(seasonal_boost=True)) == [1, 2]


class Cache:
    """A photo cache directory with files of known size and age, and a ledger of pending drafts naming some of them."""

    def __init__(self, tmp_path, monkeypatch, cap_bytes):
        self.tmp, self.drafts = tmp_path, tmp_path / "drafts"
        self.dir = tmp_path / "inat_best"
        self.dir.mkdir()
        monkeypatch.setattr(settings, "INAT_BEST_PHOTO_DIR", self.dir)
        monkeypatch.setattr(social_taste, "BYTES_PER_GB", cap_bytes)           # the cap is 1 "GB" = cap_bytes
        self.ledger = Ledger(tmp_path / "l.jsonl")

    def photo(self, name, size, age_days):
        p = self.dir / name
        p.write_bytes(b"x" * size)
        t = NOW.timestamp() - age_days * 86400
        os.utime(p, (t, t))
        return p

    def pending(self, draft_id, sources, state="drafted"):
        d = self.drafts / draft_id
        d.mkdir(parents=True)
        (d / "draft.json").write_text(json.dumps({"images": [{"file": "1.jpg", "source": str(s)} for s in sources]}), encoding="utf-8")
        self.ledger.append("drafted", draft_id, obs_id=1, now=NOW)
        if state != "drafted":
            self.ledger.append(state, draft_id, now=NOW)

    def run(self, gb=1):
        social_taste.register_hooks()
        rules = rules_with(photo_cache_cap=gb)
        social_hooks.run("maintenance", {"drafts_dir": self.drafts, "ledger": self.ledger, "rules": rules, "today": TODAY}, rules=rules, audit=Audit())

    def left(self):
        return sorted(p.name for p in self.dir.iterdir())


@pytest.fixture
def cache(tmp_path, monkeypatch):
    return Cache(tmp_path, monkeypatch, 100)


class TestPhotoCacheCap:
    def test_the_least_recently_used_files_go_first_until_the_cache_is_under_the_cap(self, cache):
        cache.photo("new.jpg", 40, 1)
        cache.photo("old.jpg", 40, 30)
        cache.photo("mid.jpg", 40, 10)
        cache.run()                                              # 120 bytes, cap 100: the oldest alone brings it to 80
        assert cache.left() == ["mid.jpg", "new.jpg"]
        rec = [r for r in Audit().records() if r["event"] == "cache_evicted"]
        assert [(r["files"], r["bytes"], r["bytes_before"], r["bytes_after"], r["cap_gb"]) for r in rec] == [(1, 40, 120, 80, 1)]

    def test_deletion_keeps_going_until_under_the_cap(self, cache):
        for name, age in (("a.jpg", 40), ("b.jpg", 30), ("c.jpg", 20), ("d.jpg", 10)):
            cache.photo(name, 60, age)
        cache.run()                                              # 240 -> two oldest go (120 > 100 still: three go)
        assert cache.left() == ["d.jpg"]
        assert [(r["files"], r["bytes"]) for r in Audit().records() if r["event"] == "cache_evicted"] == [(3, 180)]

    def test_under_the_cap_nothing_is_deleted_or_audited(self, cache):
        cache.photo("a.jpg", 50, 5)
        cache.photo("b.jpg", 50, 50)                             # exactly 100 bytes = the cap
        cache.run()
        assert cache.left() == ["a.jpg", "b.jpg"]
        assert Audit().records() == []

    def test_one_byte_over_the_cap_deletes_the_oldest(self, cache):
        cache.photo("a.jpg", 50, 5)
        cache.photo("b.jpg", 51, 50)
        cache.run()
        assert cache.left() == ["a.jpg"]

    def test_a_file_a_pending_draft_names_is_never_deleted_even_when_it_is_the_oldest(self, cache):
        keep = cache.photo("old.jpg", 40, 90)
        cache.photo("mid.jpg", 40, 20)
        cache.photo("new.jpg", 40, 1)
        cache.pending("2026-10-obs1", [keep])
        cache.run()
        assert cache.left() == ["new.jpg", "old.jpg"]
        assert [r["kept_referenced"] for r in Audit().records() if r["event"] == "cache_evicted"] == [1]

    def test_approved_drafts_protect_their_files_and_posted_ones_do_not(self, cache):
        a = cache.photo("a.jpg", 40, 90)
        b = cache.photo("b.jpg", 40, 80)
        cache.photo("c.jpg", 40, 1)
        cache.pending("2026-10-obs1", [a], state="approved")
        cache.pending("2026-10-obs2", [b], state="posted")
        cache.run()
        assert cache.left() == ["a.jpg", "c.jpg"]

    def test_a_pending_draft_that_cannot_be_read_stops_all_deletion(self, cache):
        cache.photo("a.jpg", 80, 90)
        cache.photo("b.jpg", 80, 1)
        cache.ledger.append("drafted", "2026-10-obs3", obs_id=3, now=NOW)         # no draft.json on disk
        cache.run()
        assert cache.left() == ["a.jpg", "b.jpg"]

    def test_a_referencing_path_spelled_differently_still_protects_the_file(self, cache):
        cache.photo("old.jpg", 70, 90)
        cache.photo("mid.jpg", 30, 20)
        cache.photo("new.jpg", 20, 1)
        cache.pending("2026-10-obs1", [cache.dir / ".." / "inat_best" / "old.jpg"])
        cache.run()
        assert cache.left() == ["new.jpg", "old.jpg"]

    def test_a_missing_directory_is_a_no_op(self, cache, monkeypatch):
        monkeypatch.setattr(settings, "INAT_BEST_PHOTO_DIR", cache.tmp / "nope")
        cache.run()
        assert Audit().records() == []

    def test_the_rule_off_deletes_nothing(self, cache):
        cache.photo("a.jpg", 80, 90)
        cache.photo("b.jpg", 80, 1)
        social_taste.register_hooks()
        rules = rules_with(photo_cache_cap=False)
        social_hooks.run("maintenance", {"drafts_dir": cache.drafts, "ledger": cache.ledger, "rules": rules, "today": TODAY}, rules=rules, audit=Audit())
        assert cache.left() == ["a.jpg", "b.jpg"]

    def test_the_cap_is_in_gigabytes(self, tmp_path, monkeypatch):
        c = Cache(tmp_path, monkeypatch, 100)
        monkeypatch.setattr(social_taste, "BYTES_PER_GB", 100)
        c.photo("a.jpg", 150, 90)
        c.photo("b.jpg", 40, 1)
        c.run(gb=2)                                              # 190 bytes, cap 2 x 100 = 200
        assert c.left() == ["a.jpg", "b.jpg"]
        c.run(gb=1)                                              # cap 100
        assert c.left() == ["b.jpg"]


class TestPathSafety:
    def test_a_symlink_inside_the_cache_is_neither_followed_nor_deleted(self, cache):
        outside = cache.tmp / "outside"
        outside.mkdir()
        target = outside / "precious.jpg"
        target.write_bytes(b"y" * 500)
        link = cache.dir / "link.jpg"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks need privilege on this machine")
        cache.photo("a.jpg", 80, 90)
        cache.photo("b.jpg", 80, 1)
        cache.run()
        assert target.exists() and target.read_bytes() == b"y" * 500
        assert cache.left() == ["b.jpg", "link.jpg"]

    def test_a_linked_subdirectory_is_not_entered(self, cache):
        outside = cache.tmp / "outside"
        outside.mkdir()
        (outside / "precious.jpg").write_bytes(b"y" * 500)
        sub = cache.dir / "sub"
        try:
            sub.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks need privilege on this machine")
        cache.run()
        assert (outside / "precious.jpg").exists()

    def test_a_cache_directory_that_is_itself_a_link_is_left_alone(self, cache, monkeypatch):
        real = cache.tmp / "real"
        real.mkdir()
        (real / "a.jpg").write_bytes(b"y" * 500)
        link = cache.tmp / "linkdir"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks need privilege on this machine")
        monkeypatch.setattr(settings, "INAT_BEST_PHOTO_DIR", link)
        cache.run()
        assert (real / "a.jpg").read_bytes() == b"y" * 500
        assert Audit().records() == []

    def test_positive_twin_a_plain_file_in_a_subdirectory_is_evicted(self, cache):
        (cache.dir / "sub").mkdir()
        p = cache.dir / "sub" / "deep.jpg"
        p.write_bytes(b"z" * 200)
        os.utime(p, (1, 1))
        cache.photo("b.jpg", 10, 1)
        cache.run()
        assert not p.exists() and cache.left() == ["b.jpg", "sub"]

    def test_files_outside_the_cache_directory_are_never_touched(self, cache):
        sibling = cache.tmp / "inat_best_other"
        sibling.mkdir()
        (sibling / "x.jpg").write_bytes(b"y" * 500)
        cache.photo("a.jpg", 80, 90)
        cache.photo("b.jpg", 80, 1)
        cache.run()
        assert (sibling / "x.jpg").exists() and cache.left() == ["b.jpg"]
