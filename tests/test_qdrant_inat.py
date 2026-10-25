"""qdrant_inat: all of the owner's iNat data (no photos) as Qdrant points.

iNat is a FakeInat (an in-memory API that records every request and honours id_above / updated_since / only_id / id=), Qdrant is a
FakeQdrant (collections, points, deletes, counts), the embedders map a text / file to fixed vectors; the gallery is tiny real files."""
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from bio_automation import qdrant_inat as qi
from bio_automation.qdrant_inat import InatCache, QdrantInatError
from bio_automation.qdrant_obs import Qdrant

URL = "http://qdrant.test:6333"
T0 = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.timezone.utc)


def raw(i, *, taxon=None, grade="research", geoprivacy=None, taxon_geoprivacy=None, loc="39.5,-77.5", updated="2026-10-01T00:00:00+00:00",
        **extra):
    t = {"id": 100 + i, "name": f"Amanita sp{i}", "rank": "species", "preferred_common_name": f"Fly Agaric {i}", "iconic_taxon_name": "Fungi",
         "ancestor_ids": [48460, 47170, 47169, 100 + i]} if taxon is None else taxon
    return {"id": i, "uuid": f"u{i}", "uri": f"https://www.inaturalist.org/observations/{i}", "observed_on": "2025-10-0%d" % (i % 9 + 1),
            "quality_grade": grade, "geoprivacy": geoprivacy, "taxon_geoprivacy": taxon_geoprivacy, "location": loc, "obscured": False,
            "place_guess": "Bucks County, PA, USA", "place_ids": [1, 42], "license_code": "cc-by-nc", "updated_at": updated,
            "created_at": "2025-10-01T00:00:00+00:00", "taxon": t, "tags": ["fungi", "forest"], "description": "  in the leaf litter  ",
            "annotations": [{"controlled_attribute": {"label": "Life Stage"}, "controlled_value": {"label": "Adult"}}],
            "ofvs": [{"name": "Substrate", "value": "wood"}, {"name": "Empty", "value": ""}],
            "project_observations": [{"project": {"id": 7}}], "project_ids": [7, 9], "photos": [{"id": 5000 + i, "license_code": "cc-by-nc",
            "original_dimensions": {"width": 4000, "height": 3000}}], "identifications": [{"current": True}, {"current": False}],
            "num_identification_agreements": 1, "num_identification_disagreements": 0, "comments": [], "faves": [{}], "sounds": [], **extra}


class FakeInat:
    """The iNat API: `data` {id: raw observation}; records every URL; `places` {id: place}."""

    def __init__(self, data, places=None):
        self.data, self.places, self.calls = dict(data), places or {}, []

    def __call__(self, url):
        self.calls.append(url)
        parsed = urllib.parse.urlparse(url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        if parsed.path.endswith("/places/" + parsed.path.rsplit("/", 1)[-1]) and "/places/" in parsed.path:
            ids = [int(x) for x in parsed.path.rsplit("/", 1)[-1].split(",")]
            return {"results": [self.places[i] for i in ids if i in self.places]}
        rows = sorted(self.data.values(), key=lambda r: r["id"])
        if "id" in q:
            want = {int(x) for x in q["id"].split(",")}
            return {"results": [r for r in rows if r["id"] in want]}
        total = len(rows)
        rows = [r for r in rows if r["id"] > int(q.get("id_above", 0))]
        if "updated_since" in q:
            since = dt.datetime.fromisoformat(q["updated_since"])
            rows = [r for r in rows if dt.datetime.fromisoformat(r["updated_at"]) >= since]
        rows = rows[:int(q["per_page"])]
        if q.get("only_id") == "true":
            rows = [{"id": r["id"]} for r in rows]
        return {"total_results": total, "results": rows}

    def paths(self):
        return [urllib.parse.urlparse(u).path.rsplit("/", 1)[-1] + "?" + urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("id_above", [""])[0]
                for u in self.calls]


class TestShape:
    def test_the_fields_worth_keeping_are_shaped_and_blank_ones_dropped(self):
        s = qi.shape_observation(raw(1))
        assert (s["id"], s["quality_grade"], s["tags"], s["description"], s["lat"], s["lng"]) == (1, "research", ["fungi", "forest"], "in the leaf litter", 39.5, -77.5)
        assert s["annotations"] == {"Life Stage": ["Adult"]} and s["ofvs"] == [{"name": "Substrate", "value": "wood"}]
        assert (s["project_ids"], s["identifications_count"], s["identifications_current"], s["faves_count"], s["comments_count"]) == ([7, 9], 2, 1, 1, 0)
        assert s["photos"] == [{"id": 5001, "license": "cc-by-nc", "w": 4000, "h": 3000}]
        assert s["taxon"] == {"id": 101, "name": "Amanita sp1", "rank": "species", "common": "Fly Agaric 1", "iconic": "Fungi",
                              "ancestor_ids": [48460, 47170, 47169, 101]}

    def test_coordinates_are_dropped_for_private_geoprivacy_or_private_taxon_and_kept_for_open_and_obscured(self):
        for kw in ({"geoprivacy": "private"}, {"taxon_geoprivacy": "private"}):
            s = qi.shape_observation(raw(1, **kw))
            assert (s["lat"], s["lng"]) == (None, None)
        for kw in ({}, {"geoprivacy": "obscured"}, {"taxon_geoprivacy": "obscured"}):
            assert (qi.shape_observation(raw(1, **kw))["lat"]) == 39.5

    def test_an_observation_without_a_taxon_or_a_location_shapes_without_them(self):
        s = qi.shape_observation({**raw(2, taxon={}), "location": None, "taxon": None})
        assert s["taxon"] is None and s["lat"] is None


class TestSyncRecords:
    def cache(self, tmp_path):
        return InatCache(tmp_path / "inat.json")

    def test_the_first_run_pulls_every_page_by_id_above_and_saves_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qi, "PER_PAGE", 2)
        api = FakeInat({i: raw(i) for i in range(1, 6)})
        c = self.cache(tmp_path)
        stats = qi.sync_records(c, "rj", get=api, now=lambda: T0)
        assert stats == {"live_total": 5, "new": 5, "changed": 0, "dropped": 0, "refetched": 0, "cached": 5}
        assert api.paths() == ["observations?", "observations?0", "observations?2", "observations?4", "observations?5"]
        reread = InatCache(tmp_path / "inat.json")
        assert (sorted(reread.obs), reread.login, reread.last_sync) == ([1, 2, 3, 4, 5], "rj", "2026-10-22T12:00:00+00:00")

    def test_a_later_run_fetches_only_new_and_changed_observations(self, tmp_path):
        api = FakeInat({i: raw(i) for i in range(1, 4)})
        c = self.cache(tmp_path)
        qi.sync_records(c, "rj", get=api, now=lambda: T0)
        api.data[4] = raw(4, updated="2026-10-22T13:00:00+00:00")
        api.data[2] = raw(2, grade="needs_id", updated="2026-10-22T12:30:00+00:00")
        api.calls.clear()
        later = dt.datetime(2026, 10, 23, tzinfo=dt.timezone.utc)
        stats = qi.sync_records(c, "rj", get=api, now=lambda: later)
        assert stats == {"live_total": 4, "new": 1, "changed": 1, "dropped": 0, "refetched": 0, "cached": 4}
        assert c.obs[2]["quality_grade"] == "needs_id" and 4 in c.obs
        ids_requested = [urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("id_above") for u in api.calls]
        assert ["3"] in ids_requested and ["0"] in ids_requested                       # id_above the highest cached id, and the updated_since pull

    def test_a_deleted_observation_is_dropped_and_a_missing_one_is_fetched(self, tmp_path):
        api = FakeInat({i: raw(i) for i in range(1, 5)})
        c = self.cache(tmp_path)
        qi.sync_records(c, "rj", get=api, now=lambda: T0)
        del api.data[2]                                                                  # deleted on iNat
        stats = qi.sync_records(c, "rj", get=api, now=lambda: T0 + dt.timedelta(days=1))
        assert (stats["dropped"], stats["live_total"], stats["cached"]) == (1, 3, 3) and 2 not in c.obs
        del c.obs[3]                                                                     # lost from the cache: counted as missing
        stats = qi.sync_records(c, "rj", get=api, now=lambda: T0 + dt.timedelta(days=2))
        assert stats["refetched"] == 1 and 3 in c.obs

    def test_a_cache_that_still_disagrees_with_the_live_total_is_an_error(self, tmp_path):
        class Lying(FakeInat):
            def __call__(self, url):
                out = super().__call__(url)
                if "total_results" in out:
                    out["total_results"] += 1
                return out
        with pytest.raises(QdrantInatError, match=r"^the cache holds 3 observations but iNat reports 4 for rj$"):
            qi.sync_records(self.cache(tmp_path), "rj", get=Lying({i: raw(i) for i in range(1, 4)}), now=lambda: T0)

    def test_a_cache_of_another_user_is_refused_and_a_limit_skips_the_count_check(self, tmp_path):
        api = FakeInat({i: raw(i) for i in range(1, 4)})
        c = self.cache(tmp_path)
        qi.sync_records(c, "rj", get=api, now=lambda: T0)
        with pytest.raises(QdrantInatError, match=r"belongs to iNat user 'rj', not 'other'"):
            qi.sync_records(c, "other", get=api, now=lambda: T0)

    def test_the_wait_function_runs_before_every_page(self, tmp_path):
        waits = []
        api = FakeInat({i: raw(i) for i in range(1, 3)})
        qi.sync_records(self.cache(tmp_path), "rj", get=api, wait=lambda: waits.append(len(api.calls)), now=lambda: T0)
        assert waits == [1, 2]                                                           # before each page request, not before the count


PLACES = {1: {"id": 1, "name": "United States", "display_name": "United States", "admin_level": 0},
          42: {"id": 42, "name": "Pennsylvania", "display_name": "Pennsylvania, US", "admin_level": 10},
          43: {"id": 43, "name": "Bucks", "display_name": "Bucks, PA, US", "admin_level": 20},
          44: {"id": 44, "name": "Some Park", "display_name": "Some Park", "admin_level": None}}


class TestPlaces:
    def test_places_are_fetched_once_for_public_observations_and_named_by_admin_level(self, tmp_path):
        api = FakeInat({1: raw(1, place_ids=[1, 42, 43, 44]), 2: raw(2, geoprivacy="private", place_ids=[99])}, places=PLACES)
        c = InatCache(tmp_path / "c.json")
        qi.sync_records(c, "rj", get=api, now=lambda: T0)
        api.calls.clear()
        assert qi.resolve_places(c, get=api) == 4
        assert [u.rsplit("/", 1)[-1] for u in api.calls] == ["1,42,43,44"]                # the private observation's place 99 is never asked for
        assert qi.resolve_places(c, get=api) == 0 and len(api.calls) == 1
        assert qi.place_fields(c.obs[1], c.places) == {"place_country": "United States", "place_state": "Pennsylvania", "place_county": "Bucks",
                                                       "place_names": ["United States", "Pennsylvania", "Bucks"]}
        assert qi.place_fields(c.obs[2], c.places) == {}


def lineage(*names):
    return [{"rank": r, "name": n, "common_name": None} for r, n in names]


class TestTaxa:
    def cache(self, tmp_path, records):
        c = InatCache(tmp_path / "c.json")
        c.obs = {r["id"]: qi.shape_observation(r) for r in records}
        return c

    def test_lineages_take_their_ids_from_ancestor_ids_and_fall_back_to_names_when_the_lists_do_not_line_up(self, tmp_path):
        c = self.cache(tmp_path, [raw(1), raw(2, taxon={"id": 102, "name": "Boletus edulis", "rank": "species", "iconic_taxon_name": "Fungi",
                                                          "ancestor_ids": [1, 2, 3, 4, 102]})])
        fetched = {101: lineage(("stateofmatter", "Life"), ("kingdom", "Fungi"), ("phylum", "Basidiomycota")),
                   102: lineage(("kingdom", "Fungi"), ("phylum", "Basidiomycota"))}
        got = qi.taxon_lineages(c, {}, lambda ids: {t: fetched[t] for t in ids})
        assert [(a["id"], a["rank"]) for a in got[101]] == [(48460, "stateofmatter"), (47170, "kingdom"), (47169, "phylum")]
        assert [a["id"] for a in got[102]] == [None, None]

    def test_the_root_life_in_the_ancestor_ids_is_dropped_when_the_named_lineage_starts_at_the_kingdom(self, tmp_path):
        c = self.cache(tmp_path, [raw(1)])                                          # ancestor_ids [48460 Life, 47170, 47169, 101]
        got = qi.taxon_lineages(c, {}, lambda ids: {101: lineage(("kingdom", "Fungi"), ("phylum", "Basidiomycota"))})
        assert [(a["id"], a["name"]) for a in got[101]] == [(47170, "Fungi"), (47169, "Basidiomycota")]

    def test_lineage_check_counts_unaligned_lineages_and_ids_with_two_different_names(self):
        ok = {1: [{"id": 5, "rank": "kingdom", "name": "Fungi", "common": None}], 2: [{"id": 5, "rank": "kingdom", "name": "Fungi", "common": None}]}
        assert qi.lineage_check(ok) == {"without_ids": 0, "conflicts": 0}
        bad = {**ok, 3: [{"id": None, "rank": "kingdom", "name": "Fungi", "common": None}], 4: [{"id": 5, "rank": "kingdom", "name": "Plantae", "common": None}]}
        assert qi.lineage_check(bad) == {"without_ids": 1, "conflicts": 1}

    def test_a_cached_lineage_is_not_fetched_again(self, tmp_path):
        c = self.cache(tmp_path, [raw(1)])
        cache, asked = {101: lineage(("kingdom", "Fungi"), ("phylum", "X"), ("class", "Y"))}, []
        qi.taxon_lineages(c, cache, lambda ids: asked.append(ids) or {})
        assert asked == [[]]

    def test_counts_dates_and_species_roll_up_the_tree(self, tmp_path):
        recs = [raw(1, grade="research"), raw(2, grade="needs_id"), raw(3, grade="research",
                taxon={"id": 103, "name": "Amanita", "rank": "genus", "iconic_taxon_name": "Fungi", "ancestor_ids": [48460, 47170, 47169, 103]})]
        recs[0]["observed_on"], recs[1]["observed_on"], recs[2]["observed_on"] = "2025-03-01", "2024-05-09", "2026-01-02"
        c = self.cache(tmp_path, recs)
        lin = {101: [{"id": 47170, "rank": "kingdom", "name": "Fungi", "common": None}, {"id": 47169, "rank": "phylum", "name": "Basidiomycota", "common": None}],
               102: [{"id": 47170, "rank": "kingdom", "name": "Fungi", "common": None}, {"id": 47169, "rank": "phylum", "name": "Basidiomycota", "common": None}],
               103: [{"id": 47170, "rank": "kingdom", "name": "Fungi", "common": None}, {"id": 47169, "rank": "phylum", "name": "Basidiomycota", "common": None}]}
        taxa = qi.build_taxa(c, lin, local_obs={1, 3})
        phylum = taxa[47169]
        assert (phylum["obs_count"], phylum["direct_count"], phylum["research_count"], phylum["species_count"], phylum["local_obs_count"]) == (3, 0, 2, 2, 2)
        assert (phylum["first_observed"], phylum["last_observed"], phylum["parent_id"], phylum["kingdom"], phylum["rank"]) == (
            "2024-05-09", "2026-01-02", 47170, "Fungi", "phylum")
        assert "parent_id" not in taxa[47170] and taxa[47170]["species_count"] == 2
        genus = taxa[103]
        assert (genus["obs_count"], genus["direct_count"], genus["species_count"], genus["parent_id"]) == (1, 1, 0, 47169)
        sp = taxa[101]
        assert (sp["species_count"], sp["direct_count"], sp["iconic"], sp["species"]) == (1, 1, "Fungi", "Amanita sp1")
        assert "iconic" not in taxa[47170]                                               # only a taxon's own observations name its iconic group
        assert (sp["title"], sp["source"]) == ("Fly Agaric 1 / Amanita sp1", "https://www.inaturalist.org/taxa/101")
        assert sp["text"] == ("Fly Agaric 1 / Amanita sp1\nRank: species\nLineage: Fungi > Basidiomycota > Amanita sp1\n"
                              "1 observations (1 research grade), 1 species\nObserved 2025-03-01 to 2025-03-01")

    def test_the_root_of_life_is_not_a_taxon_point(self, tmp_path):
        c = self.cache(tmp_path, [raw(1)])
        lin = {101: [{"id": 48460, "rank": "stateofmatter", "name": "Life", "common": None}, {"id": 47170, "rank": "kingdom", "name": "Fungi", "common": None}]}
        assert sorted(qi.build_taxa(c, lin, set())) == [101, 47170]


class TestPayload:
    def build(self, r, **kw):
        c = InatCache(Path("unused.json"))
        s = qi.shape_observation(r)
        anc = [{"id": 47170, "rank": "kingdom", "name": "Fungi", "common": None}]
        places = {1: PLACES[1], 42: PLACES[42]}
        return qi.build_obs_payload(s, anc, places, **{"local_files": 0, "local_photos": 0, "prints": [], "status": None, **kw})

    def test_a_public_observation_carries_everything_with_the_fields_loci_reads(self):
        p = self.build(raw(1), local_files=2, local_photos=1, prints=["https://p/1"], status={"state": "posted", "draft_id": "d"})
        assert (p["obs_id"], p["taxon_id"], p["kingdom"], p["species"], p["year"], p["month"], p["quality_grade"]) == (1, 101, "Fungi", "Amanita sp1", 2025, 10, "research")
        assert (p["lat"], p["lng"], p["place"], p["place_country"], p["place_state"]) == (39.5, -77.5, "Bucks County, PA, USA", "United States", "Pennsylvania")
        assert (p["ann_life_stage"], p["annotations"], p["tags"], p["project_ids"]) == ("Adult", {"Life Stage": ["Adult"]}, ["fungi", "forest"], [7, 9])
        assert (p["local_files"], p["in_gallery"], p["inat_photos_with_local_source"], p["has_local_photo"], p["prints"], p["social_state"]) == (
            2, True, 1, True, ["https://p/1"], "posted")
        assert p["text"].startswith("Fly Agaric 1 / Amanita sp1\nLineage: Fungi > Amanita sp1\nGrade: research\nDate: 2025-10-02\n"
                                    "Place: United States, Pennsylvania\nAnnotations: Life Stage: Adult\nTags: fungi, forest\n"
                                    "Observation fields: Substrate: wood\nin the leaf litter")
        assert (p["title"], p["source"]) == ("Fly Agaric 1 / Amanita sp1", "https://www.inaturalist.org/observations/1")

    def test_a_private_observation_has_no_coordinates_no_place_and_no_accuracy(self):
        p = self.build(raw(1, geoprivacy="private", positional_accuracy=5))
        for k in ("lat", "lng", "place", "place_country", "place_state", "place_names", "positional_accuracy"):
            assert k not in p, k
        assert "Place:" not in p["text"] and p["geoprivacy"] == "private"

    def test_a_cached_record_that_still_holds_coordinates_is_stripped_when_its_geoprivacy_is_private(self):
        shaped = qi.shape_observation(raw(1))                                           # open when shaped: lat and lng are in the record
        for k, v in (("geoprivacy", "private"), ("taxon_geoprivacy", "private")):
            p = qi.build_obs_payload({**shaped, k: v}, [], PLACES, local_files=0, local_photos=0, prints=[], status=None)
            assert "lat" not in p and "lng" not in p and "place" not in p, k
        assert qi.build_obs_payload(shaped, [], PLACES, local_files=0, local_photos=0, prints=[], status=None)["lat"] == 39.5

    def test_local_photos_are_a_local_photo_even_without_a_gallery_file_and_none_means_none(self):
        assert self.build(raw(1), local_photos=2)["has_local_photo"] is True
        p = self.build(raw(1))
        assert (p["has_local_photo"], p["in_gallery"], p["local_files"]) == (False, False, 0)

    def test_an_unidentified_observation_still_has_a_title_and_text(self):
        p = self.build({**raw(7), "taxon": None})
        assert p["title"] == "Observation 7" and p["text"].startswith("Unidentified observation")

    def test_point_ids_are_stable_and_differ_by_kind_and_key(self):
        assert qi.point_id("obs", 1) == qi.point_id("obs", 1)
        assert len({qi.point_id("obs", 1), qi.point_id("obs", 2), qi.point_id("taxon", 1)}) == 3
        assert (qi.point_id("obs", 1), qi.point_id("taxon", 1)) == ("c7a0d2e4-eea7-5669-af29-2f364b4bc11e", "d348755f-6489-5b81-942c-d4664d5b32bf")


class FakeQdrant:
    def __init__(self, collections=None, aliases=(), wrong_count=False):
        self.collections, self.aliases, self.log, self.wrong_count = dict(collections or {}), list(aliases), [], wrong_count

    def __call__(self, method, url, headers, body):
        path, data = url[len(URL):], (json.loads(body) if body else None)
        self.log.append((method, path))
        ok = lambda r=None: (200, json.dumps({"result": r, "status": "ok"}).encode())
        if (method, path) == ("GET", "/aliases"):
            return ok({"aliases": [{"alias_name": a, "collection_name": c} for a, c in self.aliases]})
        if (method, path) == ("GET", "/collections"):
            return ok({"collections": [{"name": n} for n in self.collections]})
        m = re.fullmatch(r"/collections/([^/?]+)", path)
        if m and method == "GET":
            return ok({"config": self.collections[m.group(1)]["config"]})
        if m and method == "PUT":
            self.collections[m.group(1)] = {"config": {"params": {"vectors": data["vectors"]}, "metadata": data.get("metadata")}, "points": {}, "indexes": []}
            return ok(True)
        m = re.fullmatch(r"/collections/([^/?]+)/index\?wait=true", path)
        if m:
            self.collections[m.group(1)]["indexes"].append((data["field_name"], data["field_schema"]))
            return ok({})
        m = re.fullmatch(r"/collections/([^/?]+)/points\?wait=true", path)
        if m:
            for p in data["points"]:
                self.collections[m.group(1)]["points"][p["id"]] = p
            return ok({"status": "completed"})
        m = re.fullmatch(r"/collections/([^/?]+)/points/delete\?wait=true", path)
        if m:
            for i in data["points"]:
                self.collections[m.group(1)]["points"].pop(i, None)
            return ok({"status": "completed"})
        m = re.fullmatch(r"/collections/([^/?]+)/points/count", path)
        if m:
            return ok({"count": len(self.collections[m.group(1)]["points"]) + (1 if self.wrong_count else 0)})
        return 404, b"unknown"

    def client(self):
        return Qdrant(URL, "k", http=lambda m, u, h, b: self(m, u, h, b))

    def writes(self):
        return [(m, p) for m, p in self.log if m != "GET" and not p.endswith("/count")]


class TestCollections:
    def test_an_alias_and_a_foreign_collection_are_refused_before_any_write(self):
        fake = FakeQdrant({"theirs": {"config": {"params": {"vectors": {"size": 4}}, "metadata": None}, "points": {}, "indexes": []}},
                          aliases=[("loci_memory", "hermes_memory")])
        with pytest.raises(QdrantInatError, match=r"^'loci_memory' is an alias in Qdrant: refusing to write through it$"):
            qi.ensure(fake.client(), "loci_memory", qi.KIND_OBS, {}, (), ())
        with pytest.raises(QdrantInatError, match=r"^collection 'theirs' exists but was not created by bio-automation \(metadata \{\}\)"):
            qi.ensure(fake.client(), "theirs", qi.KIND_OBS, {}, (), ())
        assert fake.writes() == []

    def test_a_new_collection_gets_its_vectors_owner_kind_and_indexes_and_an_existing_one_is_reused_or_refused_on_a_size_change(self):
        fake = FakeQdrant()
        vectors = {"text": {"size": 768, "distance": "Cosine"}, "image": {"size": 512, "distance": "Cosine"}}
        assert qi.ensure(fake.client(), "bio_inat_observations", qi.KIND_OBS, vectors, ("kingdom",), ("year",), ("in_gallery",)) == "created"
        made = fake.collections["bio_inat_observations"]
        assert made["config"]["metadata"] == {"owner": "bio-automation", "kind": "inat-observations"} and made["config"]["params"]["vectors"] == vectors
        assert made["indexes"] == [("kingdom", "keyword"), ("year", "integer"), ("in_gallery", "bool")]
        assert qi.ensure(fake.client(), "bio_inat_observations", qi.KIND_OBS, vectors, ("kingdom",), ("year",)) == "existing"
        with pytest.raises(QdrantInatError, match=r"has vector sizes \{'text': 768, 'image': 512\}, this run needs \{'text': 1024, 'image': 512\}"):
            qi.ensure(fake.client(), "bio_inat_observations", qi.KIND_OBS, {**vectors, "text": {"size": 1024, "distance": "Cosine"}}, (), ())
        with pytest.raises(QdrantInatError, match=r"metadata \{'owner': 'bio-automation', 'kind': 'inat-observations'\}"):
            qi.ensure(fake.client(), "bio_inat_observations", qi.KIND_TAXA, vectors, (), ())     # ours, but the wrong kind: refused too


def embed_text(texts):
    embed_text.calls.append(list(texts))
    return [[float(len(t) % 7), 1.0, 0.0] for t in texts]


embed_text.calls = []


class World:
    """3 observations (two with local gallery photos), a gallery of tiny jpgs, a lineage for their taxa, and every dependency stubbed."""

    def __init__(self, tmp_path):
        from PIL import Image

        self.tmp = tmp_path
        embed_text.calls.clear()
        self.api = FakeInat({1: raw(1), 2: raw(2, geoprivacy="private", place_ids=[99]), 3: raw(3)}, places=PLACES)
        self.gallery = tmp_path / "gallery"
        self.gallery.mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            Image.new("RGB", (8, 8), (10, 20, 30)).save(self.gallery / name, "JPEG")
        self.index = {"files": {"a.jpg": 1, "b.jpg": 1, "c.jpg": 3}}
        self.vecs = {"a.jpg": np.array([1.0] + [0.0] * 511), "b.jpg": np.array([0.0, 1.0] + [0.0] * 510), "c.jpg": np.array([0.0, 0.0, 1.0] + [0.0] * 509)}
        self.embedded_images = []
        self.fake = FakeQdrant()
        self.lineages = {t: lineage(("kingdom", "Fungi"), ("phylum", "Basidiomycota"), ("class", "Agaricomycetes")) for t in (101, 102, 103)}
        (tmp_path / "coverage.csv").write_text("obs_id,year,photo_id,status,ready_file,ready_sim,library_id,library_sim\n"
                                               "2,2025,1,both,x,1,2,1\n2,2025,2,no_source_nearby,,,,\n", encoding="utf-8")

    def embed_image(self, path):
        self.embedded_images.append(Path(path).name)
        return self.vecs[Path(path).name]

    def run(self, **kw):
        args = dict(cache_file=self.tmp / "full.json", state_file=self.tmp / "state.json", client=self.fake.client(), login="rj", get=self.api,
                    wait=lambda: None, lineage_cache={}, fetch_lineages=lambda ids: {t: self.lineages[t] for t in ids}, embed_text=embed_text,
                    index=self.index, gallery_dir=self.gallery, work_dir=self.tmp / "work", embed_image=self.embed_image, prints={3: "https://p/3"},
                    ledger_drafts={"2026-10-obs3": {"obs_id": 3, "state": "posted"}}, coverage_csv=self.tmp / "coverage.csv", now=lambda: T0)
        args.update(kw)
        return qi.sync(**args)

    def points(self, name):
        return self.fake.collections[name]["points"]


class TestSync:
    def test_the_first_run_writes_both_collections_with_the_right_vectors_and_payloads(self, tmp_path):
        w = World(tmp_path)
        s = w.run()
        assert (s["observations"], s["taxa"], s["observations_count"], s["taxa_count"], s["written"]) == (3, 6, 3, 6, True)
        assert (s["observations_collection"], s["taxa_collection"]) == ("created", "created")
        assert (s["with_image_vector"], s["with_local_photo"], s["private"], s["gallery_embedded"]) == (2, 3, 1, 3)
        assert s["lineages"] == {"without_ids": 0, "conflicts": 0}
        obs = w.points("bio_inat_observations")
        p1, p2, p3 = (obs[qi.point_id("obs", i)] for i in (1, 2, 3))
        assert sorted(p1["vector"]) == ["image", "text"] and len(p1["vector"]["text"]) == 3
        assert np.allclose(p1["vector"]["image"], np.array([1, 1] + [0] * 510) / np.sqrt(2))               # the mean of a.jpg and b.jpg, normalised
        assert np.allclose(p3["vector"]["image"], [0, 0, 1] + [0] * 509)
        assert "image" not in p2["vector"]                                                                  # observation 2 has no local gallery photo
        assert (p2["payload"]["has_local_photo"], p2["payload"]["inat_photos_with_local_source"], p2["payload"]["in_gallery"]) == (True, 1, False)
        assert "lat" not in p2["payload"] and p1["payload"]["lat"] == 39.5
        assert (p3["payload"]["prints"], p3["payload"]["social_state"], p1["payload"]["local_files"]) == (["https://p/3"], "posted", 2)
        taxa = w.points("bio_inat_taxa")
        assert sorted(t["payload"]["name"] for t in taxa.values()) == ["Agaricomycetes", "Amanita sp1", "Amanita sp2", "Amanita sp3", "Basidiomycota", "Fungi"]
        kingdom = [t["payload"] for t in taxa.values() if t["payload"].get("rank") == "kingdom"][0]
        assert (kingdom["name"], kingdom["obs_count"], kingdom["species_count"]) == ("Fungi", 3, 3)

    def test_a_second_run_embeds_and_writes_nothing(self, tmp_path):
        w = World(tmp_path)
        w.run()
        writes_before, texts_before = len(w.fake.writes()), len(embed_text.calls)
        w.embedded_images.clear()
        s = w.run(now=lambda: T0 + dt.timedelta(days=1))
        assert (s["observations_write"], s["taxa_write"]) == ({"written": 0, "unchanged": 3}, {"written": 0, "unchanged": 6})
        assert len(w.fake.writes()) == writes_before and len(embed_text.calls) == texts_before + 1          # one call: the dimension probe
        assert w.embedded_images == []

    def test_only_the_observation_that_changed_is_embedded_and_rewritten(self, tmp_path):
        w = World(tmp_path)
        w.run()
        w.api.data[3] = raw(3, grade="needs_id", updated="2026-10-23T00:00:00+00:00")
        embed_text.calls.clear()
        s = w.run(now=lambda: T0 + dt.timedelta(days=2))
        assert s["observations_write"] == {"written": 1, "unchanged": 2}
        assert [len(c) for c in embed_text.calls] == [1, 1, 4]            # the probe, observation 3, and the 4 taxa whose research count it changed
        assert w.points("bio_inat_observations")[qi.point_id("obs", 3)]["payload"]["quality_grade"] == "needs_id"

    def test_an_observation_deleted_on_inat_loses_its_point_and_the_counts_are_verified(self, tmp_path):
        w = World(tmp_path)
        w.run()
        del w.api.data[2]
        s = w.run(now=lambda: T0 + dt.timedelta(days=2))
        assert (s["observations_deleted"], s["observations_count"]) == (1, 2)
        assert qi.point_id("obs", 2) not in w.points("bio_inat_observations")
        assert ("POST", "/collections/bio_inat_observations/points/delete?wait=true") in w.fake.log

    def test_a_count_that_does_not_match_after_the_write_is_an_error(self, tmp_path):
        w = World(tmp_path)
        w.fake = FakeQdrant(wrong_count=True)
        with pytest.raises(QdrantInatError, match=r"^bio_inat_observations holds 4 points after the sync, expected 3$"):
            w.run()

    def test_lineages_whose_ids_disagree_with_their_names_stop_the_write_before_anything_is_sent(self, tmp_path):
        w = World(tmp_path)
        w.lineages[102] = lineage(("kingdom", "Plantae"), ("phylum", "Basidiomycota"), ("class", "Agaricomycetes"))      # id 48460 named Fungi elsewhere
        with pytest.raises(QdrantInatError, match=r"^1 taxon id\(s\) carry two different names or ranks in the lineages: the ids and the names do not line up"):
            w.run()
        assert w.fake.log == []
        assert w.run(client=None, write=False)["lineages"] == {"without_ids": 0, "conflicts": 1}

    def test_a_dry_run_builds_everything_and_sends_nothing(self, tmp_path):
        w = World(tmp_path)
        s = w.run(client=None, write=False)
        assert (s["observations"], s["taxa"], s["written"]) == (3, 6, False) and "observations_count" not in s
        assert w.fake.log == [] and not (tmp_path / "state.json").exists()

    def test_a_trial_prefix_and_limit_write_other_collections_and_keep_the_full_cache_whole(self, tmp_path):
        w = World(tmp_path)
        s = w.run(prefix="zz_trial_", limit=2)
        assert (s["observations"], s["observations_count"]) == (2, 2) and sorted(w.fake.collections) == ["zz_trial_observations", "zz_trial_taxa"]
        assert sorted(InatCache(tmp_path / "full.json").obs) == [1, 2, 3]                                   # the cache is not truncated by a trial

    def test_the_text_embedder_must_return_one_vector_per_text(self, tmp_path):
        w = World(tmp_path)
        with pytest.raises(QdrantInatError, match=r"^the embedder returned 0 vectors for 3 texts$"):
            w.run(embed_text=lambda texts: [[0.0, 0.0, 0.0]] if texts == ["dimension probe"] else [])
