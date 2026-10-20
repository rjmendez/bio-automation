"""inat_stock: matching frames to the owner's live iNat observations, building and embedding the Stock label. Real exiftool + Pillow JPEGs;
the iNat API is the stubbed dependency (a recording fake for fetch)."""
from __future__ import annotations

import csv
import datetime as dt
import os

import numpy as np
import pytest
from PIL import Image

from bio_automation import inat_stock
from bio_automation.inat_stock import LabelError, SnapshotStale
from bio_automation.models import Observation


def fetched(name="Leiobunum verrucosum", common="Warty Harvestman", iconic="Arachnida", rank="species", grade="research", place="Pennsylvania, US"):
    return {"observation_id": 1, "source": "inaturalist",
            "taxon": {"id": 9, "name": name, "preferred_common_name": common, "common_name": common, "iconic_taxon_name": iconic, "rank": rank},
            "place_guess": place, "license": "cc-by", "quality_grade": grade, "keywords_seed": []}


def epoch(iso):
    return dt.datetime.fromisoformat(iso).timestamp()


class TestMatchFrames:
    OBS = [{"id": 100, "time": "2026-06-13T09:34:00-04:00"}, {"id": 200, "time": "2026-06-13T09:40:00-04:00"}, {"id": 300}]

    def test_nearest_observation_within_tolerance_wins(self):
        frames = [{"id": "a", "t": epoch("2026-06-13T09:34:30-04:00")}, {"id": "b", "t": epoch("2026-06-13T09:38:50-04:00")}]
        assert inat_stock.match_frames(frames, self.OBS) == {"a": 100, "b": 200}                # b is 70 s from 200 and 290 s from 100

    def test_tolerance_boundary_is_90_seconds_inclusive(self):
        assert inat_stock.match_frames([{"id": "x", "t": epoch("2026-06-13T09:35:30-04:00")}], self.OBS) == {"x": 100}
        assert inat_stock.match_frames([{"id": "x", "t": epoch("2026-06-13T09:35:31-04:00")}], self.OBS) == {}

    def test_observation_without_a_time_never_matches(self):
        assert inat_stock.match_frames([{"id": "x", "t": 0.0}], [{"id": 300}]) == {}


class TestSnapshot:
    def test_fresh_snapshot_loads_and_stale_one_is_refused(self, tmp_path):
        p = tmp_path / "o.json"
        p.write_text('{"observations": [{"id": 1}]}', encoding="utf-8")
        now = p.stat().st_mtime
        assert inat_stock.load_snapshot(p, now=now + 2.9 * 86400) == [{"id": 1}]
        with pytest.raises(SnapshotStale, match=r"o\.json is 3\.1 days old \(limit 3\); refresh it from the live iNat account first"):
            inat_stock.load_snapshot(p, now=now + 3.1 * 86400)


class TestLabel:
    def test_species_label_is_built_from_the_fetched_observation(self):
        calls = []
        label = inat_stock.label_observation(408509974, "2026-06-13", fetch=lambda i: calls.append(i) or fetched())
        assert calls == [408509974]
        assert label == {
            "title": "Warty Harvestman Leiobunum verrucosum - Pennsylvania",
            "keywords": ["warty", "harvestman", "leiobunum", "verrucosum", "summer", "pennsylvania", "wildlife", "nature", "outdoor",
                         "photography", "macro", "close-up", "wild", "habitat", "natural", "species"],
            "caption": "Warty Harvestman (Leiobunum verrucosum) photographed at Pennsylvania, US on 2026-06-13.",
            "category": "Animals", "category_id": 1, "taxon": "Leiobunum verrucosum", "grade": "research", "rank": "species"}

    def test_fungus_without_a_common_name_has_too_few_keywords_to_ship(self):
        label = inat_stock.label_observation(1, "2026-07-30", fetch=lambda i: fetched("Amanita muscaria", None, "Fungi"))
        assert (label["title"], label["category"], label["category_id"], len(label["keywords"])) == ("Amanita muscaria - Pennsylvania", "Science", 16, 13)
        assert inat_stock.label_problems("a.jpg", label) == ["13 keywords (need at least 15)"]

    @pytest.mark.parametrize("grade,rank,ok", [("research", "species", True), ("research", "genus", True), ("needs_id", "species", True),
                                                ("needs_id", "genus", True), ("needs_id", "family", True), ("casual", "species", False),
                                                ("casual", None, False)])
    def test_label_policy_refuses_only_casual(self, grade, rank, ok):
        assert (inat_stock.label_policy(grade, rank) is None) is ok

    def test_policy_reason_is_named(self):
        assert inat_stock.label_policy("casual", "species") == "casual observation"

    def test_genus_level_observation_is_labelled_at_genus(self):
        label = inat_stock.label_observation(5, "2026-06-13", fetch=lambda i: fetched("Aphis", "Aphids", "Insecta", "genus", "needs_id"))
        assert (label["title"], label["rank"], label["grade"]) == ("Aphids Aphis - Pennsylvania", "genus", "needs_id")

    def test_refused_observation_raises_with_its_id_and_reason(self):
        with pytest.raises(LabelError, match="observation 77: casual observation"):
            inat_stock.label_observation(77, "2026-06-13", fetch=lambda i: fetched(grade="casual"))

    def test_no_taxon_raises(self):
        with pytest.raises(LabelError, match="observation 5: no taxon on iNat"):
            inat_stock.label_observation(5, None, fetch=lambda i: {**fetched(), "taxon": {}})

    def test_fetch_failure_is_reported_not_swallowed(self):
        def boom(i):
            raise ConnectionError("no route")

        with pytest.raises(LabelError, match="observation 8: fetch failed: ConnectionError: no route"):
            inat_stock.label_observation(8, None, fetch=boom)

    def test_label_from_an_already_merged_observation(self):
        obs = Observation(observation_id=4, obsdate="2026-07-02", taxon=fetched()["taxon"], place_guess="Pennsylvania, US", quality_grade="research")
        assert inat_stock.label_from_observation(obs)["title"] == "Warty Harvestman Leiobunum verrucosum - Pennsylvania"


class TestTitleAndProblems:
    def test_title_cleaning(self):
        assert inat_stock.clean_title("Common (Sci name) — Place, State") == "Common Sci name - Place State"
        long = inat_stock.clean_title("x" * 80)
        assert (len(long), long.endswith("...")) == (70, True)

    def good(self):
        return {"title": "A B", "keywords": [f"k{i}" for i in range(15)], "caption": "c", "category": "Animals", "category_id": 1}

    def test_clean_label_has_no_problems(self):
        assert inat_stock.label_problems("a.jpg", self.good()) == []

    def test_each_limit(self):
        g = self.good()
        assert inat_stock.label_problems("x" * 31 + ".jpg", g) == ["filename is 35 characters (limit 30)"]
        assert inat_stock.label_problems("x" * 26 + ".jpg", g) == []                                   # exactly 30
        assert inat_stock.label_problems("a.jpg", {**g, "title": "a, b"}) == ["title over 70 characters or contains a comma"]
        assert inat_stock.label_problems("a.jpg", {**g, "keywords": g["keywords"][:14]}) == ["14 keywords (need at least 15)"]
        assert inat_stock.label_problems("a.jpg", {**g, "keywords": [f"k{i}" for i in range(51)]}) == ["51 keywords (limit 50)"]
        assert inat_stock.label_problems("a.jpg", {**g, "caption": ""}) == ["no caption"]
        assert inat_stock.label_problems("a.jpg", {**g, "category": "Plumbing", "category_id": None}) == ["no Adobe category number for 'Plumbing'"]
        assert inat_stock.label_problems("a.jpg", {}) == ["no title", "0 keywords (need at least 15)", "no caption", "no Adobe category number for None"]


class TestEmbed:
    LABEL = {"title": "Warty Harvestman Leiobunum verrucosum - Pennsylvania", "keywords": ["warty", "harvestman", "close-up", "café"],
             "caption": "Warty Harvestman photographed at Pennsylvania, US on 2026-06-13.", "category": "Animals", "category_id": 1}

    def test_label_reads_back_exactly_and_replaces_an_earlier_one(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (32, 32), (10, 90, 10)).save(p, "JPEG")
        inat_stock.embed_label(p, {**self.LABEL, "keywords": ["old1", "old2", "old3"]})
        inat_stock.embed_label(p, self.LABEL)
        assert inat_stock.read_label(p) == {"title": self.LABEL["title"], "caption": self.LABEL["caption"], "keywords": self.LABEL["keywords"]}

    def test_unlabelled_file_reads_back_empty(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (32, 32)).save(p, "JPEG")
        assert inat_stock.read_label(p) == {"title": "", "caption": "", "keywords": []}

    def test_embedding_never_adds_gps_and_keeps_the_pixels(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (32, 32), (10, 90, 10)).save(p, "JPEG", quality=100)
        before = np.asarray(Image.open(p).convert("RGB")).tobytes()
        inat_stock.embed_label(p, self.LABEL)
        after = Image.open(p)
        assert np.asarray(after.convert("RGB")).tobytes() == before and 0x8825 not in after.getexif()

    def test_missing_file_raises_with_the_path(self, tmp_path):
        with pytest.raises(LabelError, match=r"exiftool failed on .*nothere\.jpg"):
            inat_stock.embed_label(tmp_path / "nothere.jpg", self.LABEL)


class TestAdobeCsv:
    def test_rows_use_adobes_columns_and_category_numbers(self, tmp_path):
        label = {"title": "T", "keywords": ["a", "b", "c"], "category_id": 14}
        assert inat_stock.write_adobe_csv([("one.jpg", label)], tmp_path / "a.csv") == 1
        raw = (tmp_path / "a.csv").read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")                                                  # UTF-8 with BOM, what Excel/Adobe expect
        rows = list(csv.DictReader(open(tmp_path / "a.csv", newline="", encoding="utf-8-sig")))
        assert rows == [{"Filename": "one.jpg", "Title": "T", "Keywords": "a,b,c", "Category": "14", "Releases": ""}]
        assert os.path.getsize(tmp_path / "a.csv") < 1_000_000
