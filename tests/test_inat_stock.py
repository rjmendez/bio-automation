"""inat_stock: matching frames to the owner's live iNat observations, building and embedding the Stock label. Real exiftool + Pillow JPEGs;
the iNat API is the stubbed dependency (a recording fake for fetch)."""
from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

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

    def test_a_missing_observation_is_a_refusal_with_its_id(self):
        from bio_automation.inat import InatNotFoundError

        def gone(i):
            raise InatNotFoundError("observation not found (404): https://api.inaturalist.org/v1/observations/8")

        with pytest.raises(LabelError, match=r"observation 8: fetch failed: InatNotFoundError: observation not found \(404\)"):
            inat_stock.label_observation(8, None, fetch=gone)

    def test_a_rate_limit_is_not_a_refusal_it_propagates(self):
        from bio_automation.inat import InatTransientError

        def throttled(i):
            raise InatTransientError("iNaturalist API error 429 for x, still failing after 6 attempts")

        with pytest.raises(InatTransientError, match=r"iNaturalist API error 429 for x, still failing after 6 attempts"):
            inat_stock.label_observation(8, None, fetch=throttled)

    def test_a_network_error_propagates_unchanged(self):
        def boom(i):
            raise ConnectionError("no route")

        with pytest.raises(ConnectionError, match=r"^no route$"):
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

    def test_keywords_that_look_like_json_literals_or_numbers_read_back_as_the_same_text(self, tmp_path):
        p = tmp_path / "a.jpg"
        Image.new("RGB", (32, 32)).save(p, "JPEG")
        label = {**self.LABEL, "keywords": ["true", "false", "2026", "warty"]}
        inat_stock.embed_label(p, label)
        assert inat_stock.read_label(p)["keywords"] == ["true", "false", "2026", "warty"]

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


class TestMatchPhotosToFrames:
    T0 = epoch("2026-06-13T10:51:00-04:00")
    VEC = {"p1": (1.0, 0.0), "p2": (0.0, 1.0), "f1": (1.0, 0.0), "f2": (0.0, 1.0), "sib": (0.96, 0.28), "far": (1.0, 0.0), "noimg": (1.0, 0.0)}

    def obs(self, oid, offset=0.0, photos=("p1", "p2")):
        iso = dt.datetime.fromtimestamp(self.T0 + offset).astimezone().isoformat()
        return {"id": oid, "time": iso, "photos": [{"id": p, "url": f"https://x/photos/{p}/square.jpg"} for p in photos]}

    def frames(self, tmp_path, spec):
        out = []
        for fid, offset in spec.items():
            (tmp_path / f"{fid}.jpg").write_bytes(b"x")
            out.append({"id": fid, "t": self.T0 + offset, "image": tmp_path / f"{fid}.jpg"})
        return out

    def embed(self, path):
        return np.array(self.VEC[Path(path).stem], dtype=np.float32)

    def run(self, tmp_path, spec, observations, **kw):
        return inat_stock.match_photos_to_frames(self.frames(tmp_path, spec), observations, self.embed, lambda pid, url: Path(f"{pid}.jpg"), **kw)

    def test_each_photo_takes_the_frame_that_looks_like_it_and_siblings_are_left_out(self, tmp_path):
        out = self.run(tmp_path, {"f1": -50, "f2": 10, "sib": 20}, [self.obs(100)])
        assert out == {"f1": (100, pytest.approx(1.0)), "f2": (100, pytest.approx(1.0))}                  # 'sib' (0.96 to p1) was never uploaded

    def test_match_is_inclusive_at_the_threshold_and_exclusive_just_below(self, tmp_path):
        self.VEC["sib"] = (0.97, (1 - 0.97 ** 2) ** 0.5)
        try:
            obs = [self.obs(100, photos=("p1",))]
            sim = self.run(tmp_path, {"sib": 0}, obs, min_similarity=0.0)["sib"][1]
            assert self.run(tmp_path, {"sib": 0}, obs, min_similarity=sim) == {"sib": (100, sim)}                  # exactly at the threshold: kept
            assert self.run(tmp_path, {"sib": 0}, obs, min_similarity=float(np.nextafter(sim, 1.0))) == {}          # a hair above: dropped
        finally:
            self.VEC["sib"] = (0.96, 0.28)

    def test_window_boundary_is_15_minutes(self, tmp_path):
        assert self.run(tmp_path, {"f1": 900}, [self.obs(100, photos=("p1",))]) != {}
        assert self.run(tmp_path, {"f1": 901}, [self.obs(100, photos=("p1",))]) == {}

    def test_a_frame_claimed_by_two_observations_keeps_the_closer_look(self, tmp_path):
        self.VEC.update({"g": (1.0, 0.0), "pa": (0.98, (1 - 0.98 ** 2) ** 0.5), "pb": (0.99, (1 - 0.99 ** 2) ** 0.5)})
        a, b = self.obs(100, 0, photos=("pa",)), self.obs(200, 30, photos=("pb",))
        out = self.run(tmp_path, {"g": 10}, [a, b])
        assert out["g"][0] == 200 and out["g"][1] == pytest.approx(0.99, abs=1e-4)
        assert self.run(tmp_path, {"g": 10}, [b, a])["g"][0] == 200                                           # the order of the observations does not matter

    def test_frames_without_an_image_and_observations_without_a_time_are_ignored(self, tmp_path):
        frames = self.frames(tmp_path, {"f1": 0})
        frames.append({"id": "noimg", "t": self.T0, "image": tmp_path / "missing.jpg"})
        no_time = {"id": 5, "photos": [{"id": "p1", "url": "https://x/photos/p1/square.jpg"}]}
        out = inat_stock.match_photos_to_frames(frames, [no_time, self.obs(100, photos=("p1",))], self.embed, lambda pid, url: Path(f"{pid}.jpg"))
        assert list(out) == ["f1"]

    def test_photos_are_only_fetched_for_observations_with_frames_nearby(self, tmp_path):
        asked = []
        far = self.obs(300, 86400)
        inat_stock.match_photos_to_frames(self.frames(tmp_path, {"f1": 0}), [far, self.obs(100, photos=("p1",))], self.embed,
                                          lambda pid, url: asked.append(pid) or Path(f"{pid}.jpg"))
        assert asked == ["p1"]


class TestPhotoMatches:
    VEC = {"p1": (1.0, 0.0), "p2": (0.0, 1.0), "f1": (1.0, 0.0), "f2": (0.6, 0.8)}

    def test_every_photo_gets_its_best_frame_and_similarity_even_below_the_threshold(self, tmp_path):
        T0 = epoch("2026-06-13T10:51:00-04:00")
        for n in ("f1", "f2"):
            (tmp_path / f"{n}.jpg").write_bytes(b"x")
        frames = [{"id": "f1", "t": T0, "image": tmp_path / "f1.jpg"}, {"id": "f2", "t": T0 + 5, "image": tmp_path / "f2.jpg"}]
        obs = [{"id": 7, "time": dt.datetime.fromtimestamp(T0).astimezone().isoformat(),
                "photos": [{"id": "p1", "url": "https://x/photos/p1/square.jpg"}, {"id": "p2", "url": "https://x/photos/p2/square.jpg"}]},
               {"id": 8, "time": dt.datetime.fromtimestamp(T0 + 86400).astimezone().isoformat(), "photos": [{"id": "p1", "url": "https://x/photos/p1/square.jpg"}]}]
        asked = []
        out = inat_stock.photo_matches(frames, obs, lambda p: np.array(self.VEC[Path(p).stem], dtype=np.float32),
                                       lambda pid, url: asked.append(pid) or Path(f"{pid}.jpg"))
        assert out[(7, "p1")] == ("f1", pytest.approx(1.0))
        assert out[(7, "p2")] == ("f2", pytest.approx(0.8))                      # best available, well under 0.97: reported, not dropped
        assert (8, "p1") not in out and asked == ["p1", "p2"]                    # an observation with no frame nearby is not even fetched


class TestVisualCheck:
    def test_photo_urls_use_the_requested_size(self):
        rec = {"photos": [{"id": 5, "url": "https://x/photos/5/square.jpg"}, {"id": None, "url": "https://x/photos/6/square.jpg"}, {"id": 7, "url": ""}]}
        assert inat_stock.observation_photo_urls(rec) == [("5", "https://x/photos/5/medium.jpg")]
        assert inat_stock.observation_photo_urls(rec, "large") == [("5", "https://x/photos/5/large.jpg")]
        assert inat_stock.observation_photo_urls({}) == []

    def test_photo_is_downloaded_once_politely_and_then_served_from_the_cache(self, tmp_path):
        slept, got = [], []
        kw = {"get_bytes": lambda url: got.append(url) or b"JPEGDATA", "sleep": slept.append, "delay": 0.4}
        p1 = inat_stock.cached_photo("5", "https://x/photos/5/medium.jpg", tmp_path / "c", **kw)
        p2 = inat_stock.cached_photo("5", "https://x/photos/5/medium.jpg", tmp_path / "c", **kw)
        assert p1 == p2 == tmp_path / "c" / "5.jpg" and p1.read_bytes() == b"JPEGDATA"
        assert got == ["https://x/photos/5/medium.jpg"] and slept == [0.4]

    def test_empty_cached_file_is_downloaded_again(self, tmp_path):
        (tmp_path / "5.jpg").write_bytes(b"")
        inat_stock.cached_photo("5", "u", tmp_path, get_bytes=lambda url: b"x", sleep=lambda d: None)
        assert (tmp_path / "5.jpg").read_bytes() == b"x"

    def test_thresholds_are_the_measured_ones(self):
        assert (inat_stock.PHOTO_MATCH_MIN_SIMILARITY, inat_stock.PHOTO_MATCH_WINDOW_S) == (0.97, 900.0)
