"""themes: the taxonomy cache, the weekday calendar, the matching rules and the group picker. No network: iNat is a recording stub."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from bio_automation import themes
from bio_automation.themes import Theme, ThemeError


def entry(species="Amanita muscaria", *, kingdom="Fungi", order="Agaricales", klass=None, phylum=None, rank="species", grade="research",
          iconic="Fungi", common=None, genus=None):
    lineage = [{"rank": "kingdom", "name": kingdom, "common_name": None}]
    for r, n in (("phylum", phylum), ("class", klass), ("order", order)):
        if n:
            lineage.append({"rank": r, "name": n, "common_name": None})
    lineage.append({"rank": "genus", "name": genus or species.split()[0], "common_name": None})
    return {"taxon": {"id": 1, "name": species, "common_name": common, "rank": rank, "iconic": iconic, "lineage": lineage}, "grade": grade,
            "observed_on": "2024-10-01", "geoprivacy": None, "fetched": 1000.0}


class RecordingFetch:
    """Stands in for inat.fetch_inat_observations: records every call, answers from `known`."""

    def __init__(self, known, fail_after=None):
        self.known, self.calls, self.fail_after = known, [], fail_after

    def __call__(self, ids, *, with_lineage, lineage_cache):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("iNat is down")
        self.calls.append((list(ids), with_lineage))
        return {o: self.known[o] for o in ids if o in self.known}


def reshaped(name="Amanita muscaria", rank="species", grade="research"):
    return {"observation_id": 1, "quality_grade": grade, "observed_on": "2024-10-01", "geoprivacy": None,
            "taxon": {"id": 5, "name": name, "common_name": "Fly Agaric", "rank": rank, "iconic_taxon_name": "Fungi",
                      "lineage": [{"rank": "kingdom", "name": "Fungi", "common_name": None}]}}


class TestTaxonomy:
    def test_missing_observations_are_fetched_in_chunks_cached_and_not_fetched_twice(self, tmp_path):
        cache = tmp_path / "tax.json"
        fetch = RecordingFetch({1: reshaped(), 2: reshaped("Boletus edulis"), 3: reshaped("Lactarius")})
        tax, stats = themes.load_taxonomy([3, 1, 2, 1], cache, fetch=fetch, chunk=2, now=lambda: 1000.0)
        assert fetch.calls == [([1, 2], True), ([3], True)]
        assert stats == {"wanted": 3, "fetched": 3, "missing": 0, "cached": 0}
        assert sorted(tax) == [1, 2, 3] and tax[2]["taxon"]["name"] == "Boletus edulis" and tax[1]["grade"] == "research"
        again, stats2 = themes.load_taxonomy([1, 2, 3], cache, fetch=fetch, now=lambda: 1001.0)
        assert fetch.calls == [([1, 2], True), ([3], True)] and stats2 == {"wanted": 3, "fetched": 0, "missing": 0, "cached": 3}
        assert again == tax

    def test_an_observation_inat_no_longer_returns_is_recorded_missing_and_not_asked_for_again(self, tmp_path):
        cache = tmp_path / "tax.json"
        fetch = RecordingFetch({1: reshaped()})
        tax, stats = themes.load_taxonomy([1, 2], cache, fetch=fetch, now=lambda: 1000.0)
        assert tax[2] == {"missing": True, "fetched": 1000.0} and stats["missing"] == 1
        themes.load_taxonomy([1, 2], cache, fetch=fetch, now=lambda: 1001.0)
        assert len(fetch.calls) == 1

    def test_an_entry_older_than_max_age_is_fetched_again_and_a_young_one_is_not(self, tmp_path):
        cache = tmp_path / "tax.json"
        fetch = RecordingFetch({1: reshaped()})
        themes.load_taxonomy([1], cache, fetch=fetch, max_age_s=100, now=lambda: 1000.0)
        themes.load_taxonomy([1], cache, fetch=fetch, max_age_s=100, now=lambda: 1099.0)
        assert len(fetch.calls) == 1
        themes.load_taxonomy([1], cache, fetch=fetch, max_age_s=100, now=lambda: 1101.0)
        assert len(fetch.calls) == 2

    def test_what_was_fetched_before_a_failure_is_kept(self, tmp_path):
        cache = tmp_path / "tax.json"
        fetch = RecordingFetch({1: reshaped(), 2: reshaped(), 3: reshaped()}, fail_after=1)
        with pytest.raises(RuntimeError, match="iNat is down"):
            themes.load_taxonomy([1, 2, 3], cache, fetch=fetch, chunk=2, now=lambda: 1000.0)
        assert sorted(json.loads(cache.read_text(encoding="utf-8"))) == ["1", "2"]
        fetch2 = RecordingFetch({3: reshaped()})
        themes.load_taxonomy([1, 2, 3], cache, fetch=fetch2, chunk=2, now=lambda: 1001.0)
        assert fetch2.calls == [([3], True)]

    def test_ranks_come_from_the_lineage_and_the_taxon_itself(self):
        e = entry("Amanita muscaria", order="Agaricales", klass="Agaricomycetes", phylum="Basidiomycota")
        assert themes.ranks_of(e) == {"kingdom": "Fungi", "phylum": "Basidiomycota", "class": "Agaricomycetes", "order": "Agaricales",
                                      "genus": "Amanita", "species": "Amanita muscaria"}
        sub = entry("Amanita muscaria guessowii", rank="subspecies")
        assert themes.ranks_of(sub)["species"] == "Amanita muscaria" and themes.ranks_of(sub)["subspecies"] == "Amanita muscaria guessowii"


class TestMatching:
    def theme(self, include, exclude=None):
        return Theme("t", "T", 0, ("T",), "", include, exclude or {})

    def test_any_include_rule_matches_ignoring_case_and_an_exclude_rule_wins(self):
        e = entry(order="Agaricales", klass="Agaricomycetes")
        assert themes.matches(self.theme({"order": ["agaricales"]}), e)
        assert themes.matches(self.theme({"order": ["Boletales"], "class": ["AGARICOMYCETES"]}), e)
        assert not themes.matches(self.theme({"order": ["Boletales"]}), e)
        assert not themes.matches(self.theme({"kingdom": ["Fungi"]}, {"order": ["Agaricales"]}), e)
        assert themes.matches(self.theme({"kingdom": ["Fungi"]}, {"order": ["Boletales"]}), e)

    def test_iconic_taxon_is_a_rule_and_a_missing_or_nameless_entry_never_matches(self):
        assert themes.matches(self.theme({"iconic": ["fungi"]}), entry())
        assert not themes.matches(self.theme({"iconic": ["Plantae"]}), entry())
        assert not themes.matches(self.theme({"kingdom": ["Fungi"]}), {"missing": True, "fetched": 1.0})
        assert not themes.matches(self.theme({"kingdom": ["Fungi"]}), {})

    def test_the_default_calendar_has_one_enabled_theme_a_day_with_fungi_friday_on_friday(self):
        days = [t.weekday for t in themes.DEFAULT_THEMES if t.enabled]
        assert sorted(days) == [0, 1, 2, 3, 4, 5, 6]
        friday = themes.theme_for(dt.date(2026, 10, 23), themes.DEFAULT_THEMES)
        assert (friday.name, friday.tags[0], friday.weekday) == ("Fungi Friday", "FungiFriday", 4)

    def test_each_kind_of_organism_belongs_to_exactly_the_theme_of_its_day(self):
        cases = {
            "mushroom-monday": entry("Amanita muscaria", order="Agaricales", klass="Agaricomycetes"),
            "fungi-friday": entry("Trametes versicolor", order="Polyporales", klass="Agaricomycetes"),
            "texture-tuesday": entry("Cladonia rangiferina", order="Lecanorales", klass="Lecanoromycetes"),
            "wildlife-wednesday": entry("Anaxyrus americanus", kingdom="Animalia", order="Anura", klass="Amphibia", iconic="Amphibia"),
            "insect-thursday": entry("Lycorma delicatula", kingdom="Animalia", order="Hemiptera", klass="Insecta", iconic="Insecta"),
            "wildflower-saturday": entry("Solidago canadensis", kingdom="Plantae", order="Asterales", klass="Magnoliopsida", iconic="Plantae"),
            "slime-mold-sunday": entry("Lycogala epidendrum", kingdom="Protozoa", order="Liceida", klass="Myxomycetes", phylum="Mycetozoa", iconic="Protozoa"),
        }
        for key, e in cases.items():
            assert [t.key for t in themes.DEFAULT_THEMES if themes.matches(t, e)] == [key], key

    def test_mosses_are_texture_tuesday_and_not_wildflower_saturday_and_a_disabled_theme_has_no_day(self):
        moss = entry("Polytrichum commune", kingdom="Plantae", order="Polytrichales", klass="Polytrichopsida", phylum="Bryophyta", iconic="Plantae")
        assert [t.key for t in themes.DEFAULT_THEMES if themes.matches(t, moss)] == ["texture-tuesday"]
        off = [t._replace(enabled=False) if t.key == "fungi-friday" else t for t in themes.DEFAULT_THEMES]
        assert themes.theme_for(dt.date(2026, 10, 23), off) is None
        assert themes.theme_for(dt.date(2026, 10, 24), off).key == "wildflower-saturday"


class TestThemeFile:
    def write(self, tmp_path, body):
        p = tmp_path / "themes.toml"
        p.write_text(body, encoding="utf-8")
        return p

    def test_no_file_gives_the_defaults_and_the_written_defaults_read_back_equal(self, tmp_path):
        assert themes.load_themes(tmp_path / "none.toml") == themes.DEFAULT_THEMES
        assert themes.load_themes(self.write(tmp_path, themes.themes_toml())) == themes.DEFAULT_THEMES

    def test_a_file_replaces_the_calendar_with_its_own_themes(self, tmp_path):
        p = self.write(tmp_path, '[[theme]]\nkey = "ticks"\nname = "Tick Tuesday"\nweekday = "tuesday"\ntags = ["TickTuesday"]\n'
                                 "[theme.include]\nfamily = [\"Ixodidae\"]\n")
        assert themes.load_themes(p) == [Theme("ticks", "Tick Tuesday", 1, ("TickTuesday",), "", {"family": ["Ixodidae"]}, {}, 2, 4, True, "")]

    @pytest.mark.parametrize("body,match", [
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = "funday"\n[theme.include]\nclass = ["X"]\n', r"theme 'a': weekday must be one of monday"),
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = 9\n[theme.include]\nclass = ["X"]\n', r"theme 'a': weekday must be a day name or 0-6"),
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = "monday"\n[theme.include]\nbogus = ["X"]\n', r"unknown rule 'bogus'"),
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = "monday"\n[theme.include]\nclass = []\n', r"class must be a non-empty list of names"),
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = "monday"\nmin_photos = 5\n[theme.include]\nclass = ["X"]\n', r"1 <= min_photos <= max_photos <= 4"),
        ('[[theme]]\nkey = "a"\nweekday = "monday"\n[theme.include]\nclass = ["X"]\n', r"theme 'a': missing name"),
        ('[[theme]]\nkey = "a"\nname = "A"\nweekday = "monday"\n[theme.include]\nclass = ["X"]\n'
         '[[theme]]\nkey = "a"\nname = "B"\nweekday = "tuesday"\n[theme.include]\nclass = ["X"]\n', r"key used twice"),
        ("[[theme]\n", r"themes\.toml: "),
    ])
    def test_a_bad_file_is_refused_naming_the_theme_and_the_field(self, tmp_path, body, match):
        with pytest.raises(ThemeError, match=match):
            themes.load_themes(self.write(tmp_path, body))


def cands(spec):
    """{obs id: candidate} from [(obs id, species, print url or None, kwargs of entry)]."""
    return {o: {"entry": entry(sp, **kw), "print": pr} for o, sp, pr, kw in spec}


MONDAY = Theme("m", "Mushroom Monday", 0, ("MushroomMonday",), "", {"order": ["Agaricales"]}, {})
DAY = dt.date(2026, 10, 26)


class TestPickGroup:
    def test_different_species_come_first_so_four_species_beat_a_dozen_of_one(self):
        spec = [(i, "Amanita muscaria", None, {}) for i in range(1, 13)] + [(100 + i, f"Boletus sp{i}", None, {}) for i in range(3)]
        got = themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set(), reserve=0)
        assert len(got) == 4
        names = [spec_name for _, spec_name in sorted((o, next(s for oo, s, _, _ in spec if oo == o)) for o in got)]
        assert len(set(names)) == 4 and names.count("Amanita muscaria") == 1

    def test_the_reserve_follows_the_four_and_a_small_pool_is_returned_whole(self):
        spec = [(i, f"Amanita sp{i}", None, {}) for i in range(1, 11)]
        assert len(themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set())) == 7
        assert len(themes.pick_group(MONDAY, cands(spec[:3]), day=DAY, taken=set())) == 3

    def test_observations_with_a_print_and_species_level_ids_go_first(self):
        spec = [(1, "Amanita aa", None, {}), (2, "Amanita bb", PRINT_URL, {}), (3, "Amanita cc", None, {"rank": "genus"}),
                (4, "Amanita dd", None, {}), (5, "Amanita ee", None, {}), (6, "Amanita ff", None, {})]
        got = themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set(), reserve=2)
        assert got[0] == 2 and got[-1] == 3 and sorted(got) == [1, 2, 3, 4, 5, 6]

    def test_taken_casual_and_unrelated_observations_are_never_picked_and_a_pool_under_the_minimum_gives_nothing(self):
        spec = [(1, "Amanita aa", None, {}), (2, "Amanita bb", None, {"grade": "casual"}), (3, "Boletus cc", None, {"order": "Boletales"}),
                (4, "Amanita dd", None, {}), (5, "Amanita ee", None, {})]
        assert sorted(themes.pick_group(MONDAY, cands(spec), day=DAY, taken={4}, reserve=0)) == [1, 5]
        assert themes.pick_group(MONDAY, cands(spec), day=DAY, taken={4, 5}, reserve=0) == []

    def test_with_species_first_off_a_species_level_id_has_no_precedence(self):
        spec = [(i, f"Amanita sp{i}", None, {"rank": "genus"}) for i in range(1, 5)] + [(5, "Amanita fine", None, {})]
        on = themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set(), reserve=0)
        off = themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set(), reserve=0, species_first=False)
        assert on[0] == 5 and len(on) == 4
        assert 5 not in off and sorted(off) == [1, 2, 3, 4]                      # the pool is shuffled as it is: this day the species-level one is left out

    def test_the_same_day_gives_the_same_group_and_another_day_gives_another_order(self):
        spec = [(i, f"Amanita sp{i}", None, {}) for i in range(1, 41)]
        a = themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set())
        assert a == themes.pick_group(MONDAY, cands(spec), day=DAY, taken=set())
        assert a != themes.pick_group(MONDAY, cands(spec), day=DAY + dt.timedelta(days=7), taken=set())


PRINT_URL = "https://rjmendez.darkroom.com/products/1"
