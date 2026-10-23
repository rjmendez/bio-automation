"""qdrant_obs: gallery images + iNat + Stock data as Qdrant points.

Qdrant is a FakeQdrant (an in-memory server that records every request), iNat is a fake batch fetch, exiftool is a fake runner, the
embedders map a file / text to fixed vectors; the gallery is three small real JPEGs under tmp_path."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation import qdrant_obs
from bio_automation.qdrant_obs import Qdrant, QdrantError

URL = "http://qdrant.test:6333"
OURS = {"owner": "bio-automation", "kind": "observations"}


class FakeQdrant:
    """An in-memory Qdrant: collections (config + points), aliases, and a log of (method, path) for every request."""

    def __init__(self, collections=None, aliases=()):
        self.collections = dict(collections or {})
        self.aliases = list(aliases)
        self.log = []
        self.bodies = []

    def __call__(self, method, url, headers, body):
        assert headers.get("api-key") == "k", "the api key is sent as a header"
        path = url[len(URL):]
        data = json.loads(body) if body else None
        self.log.append((method, path))
        self.bodies.append(data)
        ok = lambda result=None: (200, json.dumps({"result": result, "status": "ok"}).encode())
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
        m = re.fullmatch(r"/collections/([^/?]+)/points/count", path)
        if m:
            return ok({"count": len(self.collections[m.group(1)]["points"])})
        return 404, b"unknown"

    def client(self):
        return Qdrant(URL, "k", http=self)

    def writes(self):
        return [(m, p) for m, p in self.log if m != "GET" and not p.endswith("/count")]


def ours(text_dim=3):
    return {"config": {"params": {"vectors": {"image": {"size": 512}, "text": {"size": text_dim}}}, "metadata": OURS}, "points": {}, "indexes": []}


class TestCollectionSafety:
    def test_an_alias_is_refused_before_anything_is_written(self):
        fake = FakeQdrant({"hermes_memory": ours()}, aliases=[("loci_memory", "hermes_memory")])
        with pytest.raises(QdrantError, match=r"^'loci_memory' is an alias in Qdrant: refusing to write through it$"):
            qdrant_obs.ensure_collection(fake.client(), "loci_memory", 3)
        assert fake.writes() == []

    def test_a_collection_that_is_not_ours_is_refused_and_ours_is_accepted(self):
        theirs = {"config": {"params": {"vectors": {"size": 4}}, "metadata": None}, "points": {}, "indexes": []}
        fake = FakeQdrant({"loci_findings": theirs, "bio_observations": ours()})
        with pytest.raises(QdrantError, match=r"^collection 'loci_findings' exists but was not created by bio-automation \(metadata \{\}\)"):
            qdrant_obs.collection_state(fake.client(), "loci_findings")
        assert qdrant_obs.collection_state(fake.client(), "bio_observations") == "ours"
        assert qdrant_obs.collection_state(fake.client(), "something_new") == "absent"
        assert fake.writes() == []

    def test_a_new_collection_gets_both_named_vectors_the_owner_metadata_and_the_payload_indexes(self):
        fake = FakeQdrant({"other": ours()})
        assert qdrant_obs.ensure_collection(fake.client(), "bio_observations", 768) == "created"
        created = fake.collections["bio_observations"]
        assert created["config"]["params"]["vectors"] == {"image": {"size": 512, "distance": "Cosine"}, "text": {"size": 768, "distance": "Cosine"}}
        assert created["config"]["metadata"] == OURS
        assert created["indexes"] == [(f, "keyword") for f in ("kingdom", "phylum", "class", "order", "family", "genus", "species", "quality_grade")] \
            + [(f, "integer") for f in ("year", "month", "obs_id")]
        assert [(m, p) for m, p in fake.writes() if "/index" not in p] == [("PUT", "/collections/bio_observations")]

    def test_an_existing_collection_of_ours_is_reused_unless_the_text_size_changed(self):
        fake = FakeQdrant({"bio_observations": ours(768)})
        assert qdrant_obs.ensure_collection(fake.client(), "bio_observations", 768) == "existing"
        assert fake.writes() == []
        with pytest.raises(QdrantError, match=r"has vector sizes \{'image': 512, 'text': 768\}, this run needs image 512 and text 1024"):
            qdrant_obs.ensure_collection(fake.client(), "bio_observations", 1024)

    def test_an_http_error_is_a_qdrant_error_naming_the_request(self):
        client = Qdrant(URL, "k", http=lambda *a: (403, b'{"status":{"error":"forbidden"}}'))
        with pytest.raises(QdrantError, match=r"^GET /aliases: HTTP 403: .*forbidden"):
            client.call("GET", "/aliases")

    def test_the_config_comes_from_the_environment_and_a_missing_url_is_an_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIO_QDRANT_URL", "http://h:1/")
        monkeypatch.setenv("BIO_QDRANT_API_KEY", "secret")
        assert qdrant_obs.qdrant_config() == ("http://h:1", "secret")
        monkeypatch.delenv("BIO_QDRANT_URL")
        monkeypatch.delenv("BIO_QDRANT_API_KEY")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)                      # no ~/.loci/backends.toml here
        with pytest.raises(QdrantError, match=r"^no Qdrant url: set BIO_QDRANT_URL"):
            qdrant_obs.qdrant_config()


class TestIds:
    def test_a_point_id_is_a_fixed_function_of_the_gallery_path(self):
        assert qdrant_obs.point_id("2024/2024-10/a.jpg") == qdrant_obs.point_id("2024/2024-10/a.jpg")
        assert qdrant_obs.point_id("2024/2024-10/a.jpg") != qdrant_obs.point_id("2024/2024-10/b.jpg")
        assert qdrant_obs.point_id("2024/2024-10/a.jpg") == "766a0ca3-3fb1-58cc-9187-905c623c84ea"          # pinned: ids must be the same on every machine and run, or a re-run would duplicate


LINEAGE = [{"rank": "kingdom", "name": "Fungi"}, {"rank": "phylum", "name": "Basidiomycota"}, {"rank": "class", "name": "Agaricomycetes"},
           {"rank": "order", "name": "Cantharellales"}, {"rank": "family", "name": "Clavulinaceae"}, {"rank": "genus", "name": "Ramaria"}]
INAT = {"taxon": {"id": 7, "name": "Ramaria", "common_name": "coral fungi", "rank": "genus", "iconic": "Fungi", "lineage": LINEAGE},
        "grade": "needs_id", "observed_on": "2026-07-30", "geoprivacy": None, "place_guess": "Franklin County, PA, USA",
        "license": "cc-by-nc", "tags": ["fungi", "macro"]}
LABEL = {"title": "Ramaria - Pennsylvania", "caption": "Ramaria photographed in Pennsylvania.", "keywords": ["macro", "coral", "ramaria"]}


class TestPayload:
    def payload(self, **over):
        args = dict(rel="2026/2026-07/2026-07-30_DSC_1.jpg", obs_id=409468119, inat=INAT, label=LABEL, category=("Plants and Flowers", "13"),
                    prints=["https://shop/p/1"], status={"state": "posted", "draft_id": "2026-10-obs409468119"},
                    size={"width": 2667, "height": 4000, "bytes": 1700000})
        args.update(over)
        return qdrant_obs.build_payload(args["rel"], args["obs_id"], args["inat"], args["label"], args["category"], args["prints"], args["status"],
                                        args["size"])

    def test_everything_known_goes_in_with_flattened_ranks_year_month_and_merged_tags(self):
        assert self.payload() == {
            "obs_id": 409468119, "inat_url": "https://www.inaturalist.org/observations/409468119", "file": "2026/2026-07/2026-07-30_DSC_1.jpg",
            "year": 2026, "month": 7, "observed_on": "2026-07-30", "taxon_id": 7, "taxon_name": "Ramaria", "common_name": "coral fungi",
            "rank": "genus", "iconic": "Fungi", "kingdom": "Fungi", "phylum": "Basidiomycota", "class": "Agaricomycetes",
            "order": "Cantharellales", "family": "Clavulinaceae", "genus": "Ramaria", "quality_grade": "needs_id", "license": "cc-by-nc",
            "place": "Franklin County, PA, USA", "tags": ["fungi", "macro", "coral", "ramaria"], "stock_title": "Ramaria - Pennsylvania",
            "stock_caption": "Ramaria photographed in Pennsylvania.", "stock_keywords": ["macro", "coral", "ramaria"],
            "stock_category": "Plants and Flowers", "stock_category_id": "13", "prints": ["https://shop/p/1"], "social_state": "posted",
            "social_draft": "2026-10-obs409468119", "width": 2667, "height": 4000, "bytes": 1700000}

    def test_a_private_observation_has_no_place_and_an_obscured_one_keeps_it(self):
        assert "place" not in self.payload(inat={**INAT, "geoprivacy": "private"})
        keep = self.payload(inat={**INAT, "geoprivacy": "obscured"})
        assert (keep["place"], keep["geoprivacy"]) == ("Franklin County, PA, USA", "obscured")

    def test_missing_pieces_are_left_out_not_written_as_nulls(self):
        p = self.payload(label={}, category=None, prints=[], status=None, inat={"taxon": {"name": "Ramaria", "rank": "genus"}})
        assert set(p) == {"obs_id", "inat_url", "file", "year", "month", "taxon_name", "rank", "genus", "width", "height", "bytes"}

    def test_a_subspecies_also_fills_the_species_rank(self):
        taxon = {"name": "Homo sapiens neanderthalensis", "rank": "subspecies", "lineage": [{"rank": "genus", "name": "Homo"}]}
        assert qdrant_obs.lineage_ranks(taxon) == {"genus": "Homo", "species": "Homo sapiens"}

    def test_the_text_to_embed_lists_what_a_person_would_search_by(self):
        assert qdrant_obs.record_text(self.payload()) == (
            "Ramaria - Pennsylvania\ncoral fungi / Ramaria\nLineage: Fungi > Basidiomycota > Agaricomycetes > Cantharellales > Clavulinaceae > "
            "Ramaria\nGrade: needs_id\nPlace: Franklin County, PA, USA\nDate: 2026-07-30\nTags: fungi, macro, coral, ramaria\n"
            "Ramaria photographed in Pennsylvania.")


class TestSources:
    def test_exiftool_is_called_once_per_chunk_and_every_value_is_read_back_as_text(self):
        calls = []

        def run(args):
            calls.append(args)
            files = [a for a in args if a.endswith(".jpg")]
            return json.dumps([{"SourceFile": f.replace("\\", "/"), "Title": 12345, "Caption-Abstract": "c " + Path(f).name,
                                "Keywords": ["a", True] if f.endswith("a.jpg") else "single"} for f in files]).encode()

        paths = [Path(f"D:/g/{n}.jpg") for n in "abcde"]
        out = qdrant_obs.read_labels(paths, run=run, chunk=2)
        assert [len([a for a in c if a.endswith(".jpg")]) for c in calls] == [2, 2, 1]
        assert out[paths[0]] == {"title": "12345", "caption": "c a.jpg", "keywords": ["a", "true"]}
        assert out[paths[1]]["keywords"] == ["single"]

    def test_inat_is_fetched_in_chunks_cached_and_a_missing_observation_is_marked_not_refetched(self, tmp_path):
        calls = []

        def fetch(ids, with_lineage, lineage_cache):
            calls.append(list(ids))
            return {1: {"taxon": {"id": 9, "name": "A b", "common_name": "ab", "rank": "species", "iconic_taxon_name": "Plantae", "lineage": LINEAGE},
                        "quality_grade": "research", "observed_on": "2024-01-02", "geoprivacy": None, "place_guess": "X", "license": None,
                        "keywords_seed": ["t"]}, 2: {"error": "gone"}, 3: {"taxon": {}}}

        cache = tmp_path / "inat.json"
        got, st = qdrant_obs.load_inat([3, 1, 2], cache, fetch=fetch, chunk=2, now=lambda: 1000.0)
        assert calls == [[1, 2], [3]] and st == {"wanted": 3, "fetched": 3, "missing": 0, "cached": 0}
        assert got[1]["taxon"]["name"] == "A b" and got[1]["tags"] == ["t"]
        got2, st2 = qdrant_obs.load_inat([1, 2, 3], cache, fetch=fetch, chunk=2, now=lambda: 1001.0)
        assert calls == [[1, 2], [3]] and st2 == {"wanted": 3, "fetched": 0, "missing": 0, "cached": 3} and got2 == got

    def test_an_observation_inat_does_not_return_is_missing_and_a_stale_entry_is_fetched_again(self, tmp_path):
        calls = []
        responses = [{}, {1: {"taxon": {"name": "A b"}}}]

        def fetch(ids, with_lineage, lineage_cache):
            calls.append(list(ids))
            return responses[len(calls) - 1]

        cache = tmp_path / "inat.json"
        got, st = qdrant_obs.load_inat([1], cache, fetch=fetch, now=lambda: 0.0, max_age_s=100)
        assert got == {1: {"missing": True, "fetched": 0.0}} and st["missing"] == 1
        got, st = qdrant_obs.load_inat([1], cache, fetch=fetch, now=lambda: 500.0, max_age_s=100)
        assert calls == [[1], [1]] and got[1]["taxon"]["name"] == "A b"

    def test_categories_come_from_rows_that_carry_an_observation_and_a_category(self, tmp_path):
        (tmp_path / "c1.csv").write_text("obs_id,category,category_id,title\n101,Animals,1,t\n,Plants,2,x\n102,,3,y\n", encoding="utf-8")
        (tmp_path / "c2.csv").write_text("obs_id,category,category_id\n103,Nature,9\n", encoding="utf-8")
        assert qdrant_obs.categories_from_candidates([str(tmp_path / "c*.csv")]) == {101: ("Animals", "1"), 103: ("Nature", "9")}

    def test_the_most_advanced_social_state_of_an_observation_wins(self):
        drafts = {"2026-09-obs5": {"obs_id": 5, "state": "drafted"}, "2026-10-obs5": {"obs_id": 5, "state": "posted"},
                  "2026-10-obs6": {"obs_id": 6, "state": "rejected"}, "x": {"obs_id": None, "state": "drafted"}}
        assert qdrant_obs.social_states(drafts) == {5: {"state": "posted", "draft_id": "2026-10-obs5"},
                                                    6: {"state": "rejected", "draft_id": "2026-10-obs6"}}

    def test_every_darkroom_product_that_matches_an_observation_is_linked(self, tmp_path):
        files = {"a.jpg": 1, "b.jpg": 2}
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        (tmp_path / "products.json").write_text(json.dumps({"10": {"url": "https://shop/p/10"}, "11": {"url": "https://shop/p/11"},
                                                            "12": {"url": "https://shop/p/12"}}), encoding="utf-8")
        np.savez(tmp_path / "product_vecs.npz", ids=np.array(["10", "11", "12"]), vecs=np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8]]))
        assert qdrant_obs.darkroom_links(tmp_path, files, ["a.jpg", "b.jpg"], matrix, threshold=0.97, margin=0.005) == {
            1: ["https://shop/p/10", "https://shop/p/11"]}                  # product 12 looks like neither: below the threshold
        assert qdrant_obs.darkroom_links(tmp_path / "none", files, ["a.jpg", "b.jpg"], matrix, threshold=0.97, margin=0.005) == {}


GALLERY = {"2024/2024-10/a.jpg": 101, "2024/2024-10/b.jpg": 101, "2025/2025-07/c.jpg": 102}
IMAGE_VECS = {"a.jpg": [1.0, 0.0], "b.jpg": [0.0, 1.0], "c.jpg": [1.0, 1.0]}


class World:
    def __init__(self, tmp_path, fake=None, **over):
        self.tmp = tmp_path
        self.gallery = tmp_path / "gallery"
        for i, rel in enumerate(GALLERY):
            p = self.gallery / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (30 + i, 20), (i * 40, 0, 0)).save(p)
        (tmp_path / "cand.csv").write_text("obs_id,category,category_id\n101,Animals,1\n", encoding="utf-8")
        self.fake = fake or FakeQdrant()
        self.image_calls, self.text_calls = [], []
        self.over = over

    def embed_image(self, path):
        self.image_calls.append(path.name)
        return np.array(IMAGE_VECS[path.name])

    def embed_text(self, texts):
        self.text_calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.5] for t in texts]

    def fetch(self, ids, with_lineage, lineage_cache):
        return {i: {"taxon": {"id": i, "name": f"Taxon {i}", "common_name": f"common {i}", "rank": "species", "iconic_taxon_name": "Plantae",
                              "lineage": [{"rank": "kingdom", "name": "Plantae"}]}, "quality_grade": "research", "observed_on": "2024-10-0%d" % (i % 9),
                        "geoprivacy": "private" if i == 102 else None, "place_guess": f"place {i}", "license": "cc0",
                        "keywords_seed": [f"tag{i}"]} for i in ids}

    def label_run(self, args):
        files = [a for a in args if a.endswith(".jpg")]
        return json.dumps([{"SourceFile": f.replace("\\", "/"), "Title": f"title {Path(f).name}", "Caption-Abstract": "cap",
                            "Keywords": ["kw", Path(f).stem]} for f in files]).encode()

    def sync(self, **kw):
        args = dict(index={"files": GALLERY}, gallery_dir=self.gallery, client=self.fake.client(), collection="bio_observations",
                    work_dir=self.tmp / "work", embed_image=self.embed_image, embed_text=self.embed_text, fetch=self.fetch,
                    candidates=[str(self.tmp / "cand.csv")], prints={101: "https://shop/p/1"},
                    ledger_drafts={"2026-10-obs101": {"obs_id": 101, "state": "posted"}}, inat_cache=self.tmp / "inat.json",
                    label_run=self.label_run)
        args.update(self.over)
        args.update(kw)
        return qdrant_obs.sync(**args)


class TestSync:
    def test_one_point_per_image_with_both_vectors_the_payload_and_the_count_checked(self, tmp_path):
        w = World(tmp_path)
        summary = w.sync()
        points = w.fake.collections["bio_observations"]["points"]
        assert len(points) == 3 and summary["count"] == 3 and summary["written"] is True and summary["collection_action"] == "created"
        a = points[qdrant_obs.point_id("2024/2024-10/a.jpg")]
        assert a["vector"]["image"] == [1.0, 0.0] and len(a["vector"]["text"]) == 3
        assert (a["payload"]["obs_id"], a["payload"]["taxon_name"], a["payload"]["kingdom"], a["payload"]["stock_title"],
                a["payload"]["stock_category"], a["payload"]["prints"], a["payload"]["social_state"], a["payload"]["place"],
                a["payload"]["width"], a["payload"]["tags"]) == (101, "Taxon 101", "Plantae", "title a.jpg", "Animals", ["https://shop/p/1"], "posted",
                                                                 "place 101", 30, ["tag101", "kw", "a"])
        c = points[qdrant_obs.point_id("2025/2025-07/c.jpg")]["payload"]
        assert ("place" not in c, c["geoprivacy"] if "geoprivacy" in c else None, c["year"], c["month"]) == (True, "private", 2025, 7)
        assert summary["with"]["place"] == 2 and summary["with"]["prints"] == 2 and summary["with"]["stock_category"] == 2
        assert summary["text_dim"] == 3 and summary["gallery_embedded"] == 3

    def test_points_are_sent_in_batches(self, tmp_path):
        w = World(tmp_path)
        w.sync(batch=2)
        sizes = [len(b["points"]) for (m, p), b in zip(w.fake.log, w.fake.bodies) if m == "PUT" and p.endswith("/points?wait=true")]
        assert sizes == [2, 1]

    def test_a_second_run_embeds_no_image_and_updates_the_same_points(self, tmp_path):
        w = World(tmp_path)
        w.sync()
        ids = set(w.fake.collections["bio_observations"]["points"])
        w.image_calls.clear()
        summary = w.sync()
        assert (w.image_calls, summary["gallery_embedded"], summary["collection_action"], summary["count"]) == ([], 0, "existing", 3)
        assert set(w.fake.collections["bio_observations"]["points"]) == ids

    def test_a_trial_run_takes_only_the_first_files(self, tmp_path):
        w = World(tmp_path)
        summary = w.sync(limit=2)
        assert summary["images"] == 2 and summary["count"] == 2

    def test_a_dry_run_builds_everything_and_sends_nothing(self, tmp_path):
        w = World(tmp_path)
        summary = w.sync(write=False, client=None)
        assert summary["written"] is False and summary["images"] == 3 and "count" not in summary
        assert w.fake.log == []
        w.sync()
        assert len(w.fake.collections["bio_observations"]["points"]) == 3       # and the same arguments do write when asked

    def test_a_collection_that_is_not_ours_stops_the_run_before_any_write(self, tmp_path):
        foreign = {"config": {"params": {"vectors": {"size": 4}}, "metadata": None}, "points": {"x": {}}, "indexes": []}
        w = World(tmp_path, fake=FakeQdrant({"bio_observations": foreign}))
        with pytest.raises(QdrantError, match="exists but was not created by bio-automation"):
            w.sync()
        assert w.fake.writes() == [] and w.fake.collections["bio_observations"]["points"] == {"x": {}}

    def test_an_embedder_that_returns_the_wrong_number_of_vectors_is_an_error_and_nothing_is_written(self, tmp_path):
        w = World(tmp_path)
        w.embed_text = lambda texts: [[1.0, 2.0]]
        with pytest.raises(QdrantError, match=r"^the embedder returned 1 vectors for 3 texts$"):
            w.sync(embed_text=w.embed_text)
        assert w.fake.log == []
