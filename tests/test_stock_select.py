"""stock_select: Stock candidate measurement, tagging and publishing.

Image maths is tested on synthetic arrays whose answers are computable by hand (a 0/255 checkerboard has a Laplacian of +-1020 everywhere, so a
tile of it has variance 1020**2 = 1040400). Raw decode / exiftool / the Stock renderer are the dependencies and are stubbed where a test is about
the selection logic; one test per raw helper runs against the committed NEF fixtures."""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from bio_automation import cli, pipeline, stock_select
from bio_automation.io import read_json, write_json
from bio_automation.models import Asset
from bio_automation.paths import metadata_dir, source_raw_dir
from bio_automation.render import ExportBoundsError
from bio_automation.stock_select import FrameScore, StockFloorMissing

FIXTURES = Path(__file__).parent / "fixtures"
FLOOR = {"n": 100, "p10": 18.4, "p25": 49.7, "median": 114.0}
CHECKER_VAR = 1020.0 ** 2


def checker(h, w, cols=None):
    """0/255 checkerboard RGB image; cols limits the checkerboard to image columns [0, cols), the rest is flat mid-grey."""
    yy, xx = np.indices((h, w))
    v = np.where((yy + xx) % 2 == 0, 255, 0).astype(np.uint8)
    if cols is not None:
        v[:, cols:] = 128
    return np.stack([v, v, v], axis=2)


class TestTileSharpness:
    def test_checkerboard_tiles_have_the_hand_computed_variance(self):
        s = stock_select.tile_sharpness(checker(66, 66))
        assert s.sharp == pytest.approx(CHECKER_VAR, rel=1e-5)
        assert s.clip == pytest.approx(0.5, rel=1e-6)           # half the pixels are 255

    def test_flat_image_has_zero_sharpness(self):
        s = stock_select.tile_sharpness(np.full((66, 66, 3), 128, np.uint8))
        assert (s.sharp, s.clip) == (0.0, 0.0)

    def test_sharp_quarter_of_the_frame_counts_but_a_single_sharp_tile_does_not(self):
        quarter = stock_select.tile_sharpness(checker(66, 66, cols=41))           # 4 of 8 tile columns -> 32 of 64 tiles
        assert quarter.sharp == pytest.approx(CHECKER_VAR, rel=1e-5)
        one_tile = np.full((66, 66, 3), 128, np.uint8)
        one_tile[26:34, 26:34] = checker(8, 8)[:, :, :]                          # 1 of 64 tiles: below the 90th percentile
        assert stock_select.tile_sharpness(one_tile).sharp == 0.0

    def test_clip_boundary_is_250_inclusive(self):
        img = np.full((66, 66, 3), 249, np.uint8)
        assert stock_select.tile_sharpness(img).clip == 0.0
        img[:, :33] = 250
        assert stock_select.tile_sharpness(img).clip == 0.5

    def test_clip_uses_the_brightest_channel(self):
        img = np.zeros((66, 66, 3), np.uint8)
        img[..., 1] = 255
        assert stock_select.tile_sharpness(img).clip == 1.0

    def test_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"rgb must be HxWx3, got shape \(66, 66\)"):
            stock_select.tile_sharpness(np.zeros((66, 66), np.uint8))

    def test_too_small_image_is_refused_and_the_minimum_size_works(self):
        with pytest.raises(ValueError, match="image 23x23 is too small for an 8x8 tile grid"):
            stock_select.tile_sharpness(np.zeros((23, 23, 3), np.uint8))
        assert stock_select.tile_sharpness(np.zeros((24, 24, 3), np.uint8)).sharp == 0.0


class TestMedianLuma:
    def test_uniform_colour_uses_rec601_weights(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[...] = (100, 0, 0)
        assert stock_select.median_luma(img) == pytest.approx(29.9, rel=1e-5)
        img[...] = (0, 100, 0)
        assert stock_select.median_luma(img) == pytest.approx(58.7, rel=1e-5)

    def test_median_not_mean(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, :4] = 255                                                  # 40% white: mean 102, median 0
        assert stock_select.median_luma(img) == 0.0
        img[:, :6] = 255                                                  # 60% white: median flips to white
        assert stock_select.median_luma(img) == pytest.approx(255.0, rel=1e-5)

    def test_wrong_shape_refused(self):
        with pytest.raises(ValueError, match=r"rgb must be HxWx3, got shape \(4, 4\)"):
            stock_select.median_luma(np.zeros((4, 4), np.uint8))


class TestFloor:
    def test_percentiles_of_1_to_100(self):
        f = stock_select.calibrate_floor([float(i) for i in range(1, 101)])
        assert f["n"] == 100
        assert f["p10"] == pytest.approx(10.9)
        assert f["p25"] == pytest.approx(25.75)
        assert f["median"] == pytest.approx(50.5)

    def test_luma_floor_is_added_when_measured(self):
        f = stock_select.calibrate_floor([1.0, 2.0], [float(i) for i in range(1, 101)])
        assert f["luma_p10"] == pytest.approx(10.9)
        assert "luma_p10" not in stock_select.calibrate_floor([1.0, 2.0])

    def test_empty_calibration_is_refused(self):
        with pytest.raises(ValueError, match="cannot calibrate a floor from zero measurements"):
            stock_select.calibrate_floor([])

    def test_missing_floor_names_the_path_and_the_fix(self, tmp_path):
        with pytest.raises(StockFloorMissing, match=r"no sharpness floor at .*nofloor\.json; run `bio stock-calibrate"):
            stock_select.load_floor(tmp_path / "nofloor.json")

    def test_saved_floor_loads_back_identically(self, tmp_path):
        write_json(tmp_path / "f.json", FLOOR)
        assert stock_select.load_floor(tmp_path / "f.json") == FLOOR

    def test_calibrate_from_folder_measures_halved_images_and_saves(self, tmp_path):
        from PIL import Image

        folder = tmp_path / "approved" / "sub"
        folder.mkdir(parents=True)
        big = np.kron(checker(66, 66), np.ones((2, 2, 1), np.uint8))                                # 2x2-pixel blocks: halves to the 66x66 checker
        Image.fromarray(big).save(folder / "a.jpg", quality=100, subsampling=0)
        (tmp_path / "approved" / "notes.txt").write_text("not an image")
        floor = stock_select.calibrate_from_folder(tmp_path / "approved", tmp_path / "floor.json")
        assert floor["n"] == 1
        assert floor["p10"] == floor["median"] == floor["p25"]
        assert floor["p10"] == pytest.approx(CHECKER_VAR, rel=0.02)                                 # JPEG q100 is the only deviation from exact
        assert floor["luma_p10"] == pytest.approx(127.5, abs=1.0)                                   # half the pixels 255, half 0: the median sits at the 127/128 midpoint
        assert read_json(tmp_path / "floor.json") == floor

    def test_calibrate_from_folder_without_jpegs(self, tmp_path):
        with pytest.raises(ValueError, match="no JPEGs under"):
            stock_select.calibrate_from_folder(tmp_path, tmp_path / "f.json")


def passing(**kw):
    base = dict(mp=45.4, iso=800, score=FrameScore(50.0, 0.0), floor=FLOOR)
    base.update(kw)
    return stock_select.technical_reasons(**base)


class TestTechnicalReasons:
    def test_clean_frame_has_no_reasons(self):
        assert passing() == []

    @pytest.mark.parametrize("mp,ok", [(4.0, True), (3.99, False), (100.0, True), (100.01, False)])
    def test_megapixel_bounds(self, mp, ok):
        assert (passing(mp=mp) == []) is ok

    def test_iso_boundary_and_message(self):
        assert passing(iso=3200) == []
        assert passing(iso=3201) == ["ISO 3201 > 3200"]
        assert passing(iso=6400, ) == ["ISO 6400 > 3200"]

    def test_unknown_iso_is_a_reason_not_a_pass(self):
        assert passing(iso=None) == ["ISO unknown"]

    def test_sharpness_floor_boundary(self):
        assert passing(score=FrameScore(18.4, 0.0)) == []
        assert passing(score=FrameScore(18.3, 0.0)) == ["sharpness 18.3 < floor 18.4"]

    def test_clipping_boundary(self):
        assert passing(score=FrameScore(50.0, 0.02)) == []
        assert passing(score=FrameScore(50.0, 0.0201)) == ["2.0% clipped highlights"]

    def test_location_is_a_reason(self):
        assert passing(located=True) == ["carries a location"]

    def test_unscored_frame_is_only_that_reason(self):
        assert passing(score=None, iso=9999, mp=1.0) == ["unscored"]

    def test_reasons_accumulate_in_a_fixed_order(self):
        assert passing(mp=2.0, iso=6400, score=FrameScore(1.0, 0.5), located=True) == [
            "2.0 MP outside 4-100", "ISO 6400 > 3200", "sharpness 1.0 < floor 18.4", "50.0% clipped highlights", "carries a location"]

    def test_custom_max_iso(self):
        assert passing(iso=6400, max_iso=6400) == []


def frame(i, group, iso=800, mp=45.4, located=False):
    return {"id": i, "group": group, "mp": mp, "iso": iso, "located": located}


class TestDecide:
    def test_sharpest_two_per_group_win_even_when_listed_last(self):
        frames = [frame(1, "a"), frame(2, "a"), frame(3, "a"), frame(4, "b")]
        scores = {1: FrameScore(40, 0), 2: FrameScore(90, 0), 3: FrameScore(60, 0), 4: FrameScore(25, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR)}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [2, 3, 4]
        assert rows[1]["reasons"] == ["burst: not among the 2 sharpest of its group"]
        assert rows[1]["tech_pass"] is True and rows[1]["stock_candidate"] is False

    def test_sharpest_frame_failing_the_bar_does_not_take_a_slot(self):
        frames = [frame(1, "a", iso=6400), frame(2, "a"), frame(3, "a")]
        scores = {1: FrameScore(500, 0), 2: FrameScore(30, 0), 3: FrameScore(20, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR, top_per_group=1)}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [2]
        assert rows[1]["reasons"] == ["ISO 6400 > 3200"] and rows[1]["tech_pass"] is False

    def test_flagged_group_is_excluded_even_when_its_frames_are_the_sharpest(self):
        frames = [frame(1, "a"), frame(2, "a"), frame(3, "b"), frame(4, "b")]
        scores = {1: FrameScore(400, 0), 2: FrameScore(300, 0), 3: FrameScore(30, 0), 4: FrameScore(25, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR, excluded_groups={"a": "flagged people_visible, sign_or_text"})}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [3, 4]
        assert rows[1]["reasons"] == ["group excluded: flagged people_visible, sign_or_text"] and rows[1]["tech_pass"] is False
        unflagged = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR)}
        assert [i for i, r in unflagged.items() if r["stock_candidate"]] == [1, 2, 3, 4]       # positive twin: without the exclusion they win

    def test_every_row_is_marked_release_unscreened_and_scores_are_rounded(self):
        rows = stock_select.decide([frame(1, "a")], {1: FrameScore(50.126, 0.012345678)}, FLOOR)
        assert (rows[0]["release"], rows[0]["sharp"], rows[0]["clip"], rows[0]["mp"]) == ("unscreened", 50.13, 0.01235, 45.4)

    def test_unscored_frame_is_reported_not_picked(self):
        rows = stock_select.decide([frame(1, "a")], {}, FLOOR)
        assert rows[0]["stock_candidate"] is False and rows[0]["reasons"] == ["unscored"] and rows[0]["sharp"] is None


class TestRawHelpers:
    def test_real_nef_fixture_measures_to_pinned_values(self):
        raw = FIXTURES / "sample_raw.NEF"
        s = stock_select.score_raw(raw)
        assert s.sharp == pytest.approx(48.284, rel=1e-3)
        assert s.clip == pytest.approx(0.04104, rel=1e-3)
        assert stock_select.raw_megapixels(raw) == pytest.approx(45.74976, rel=1e-6)
        assert stock_select.raw_iso(raw) == 64

    def test_second_fixture_is_distinguishable(self):
        s = stock_select.score_raw(FIXTURES / "sample_raw_fungus.NEF")
        assert s.sharp == pytest.approx(70.648, rel=1e-3)
        assert s.clip == pytest.approx(0.013399, rel=1e-3)


def seed_folder(tmp_path):
    pipeline.ensure_layout(tmp_path)
    raw_dir = source_raw_dir(tmp_path)
    for name in ("DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"):
        (raw_dir / name).write_bytes(b"x")
    pipeline.save_assets(tmp_path, [
        Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0003.NEF", role="original", status="rejected", selected=False),
    ])
    write_json(tmp_path / "floor.json", FLOOR)


class TestTagObservation:
    def test_kept_raws_are_scored_tagged_and_written(self, tmp_path, monkeypatch):
        seed_folder(tmp_path)
        measured = {"DSC_0001.NEF": FrameScore(80.0, 0.0), "DSC_0002.NEF": FrameScore(5.0, 0.0)}
        scored = []
        monkeypatch.setattr(stock_select, "score_raw", lambda p: scored.append(p.name) or measured[p.name])
        monkeypatch.setattr(stock_select, "raw_megapixels", lambda p: 45.4)
        monkeypatch.setattr(stock_select, "raw_iso", lambda p: 400)
        result = stock_select.tag_observation(tmp_path, tmp_path / "floor.json")
        assert scored == ["DSC_0001.NEF", "DSC_0002.NEF"]                      # the rejected raw is never read
        by_id = {f["id"]: f for f in result["frames"]}
        assert by_id["source/raw/DSC_0001.NEF"]["stock_candidate"] is True
        assert by_id["source/raw/DSC_0002.NEF"]["reasons"] == ["sharpness 5.0 < floor 18.4"]
        assert read_json(metadata_dir(tmp_path) / "stock_tags.json") == result

    def test_a_raw_that_cannot_be_decoded_is_tagged_unscored(self, tmp_path, monkeypatch):
        seed_folder(tmp_path)

        def decode(path):
            if path.name == "DSC_0001.NEF":
                raise RuntimeError("bad raw")
            return FrameScore(80.0, 0.0)

        monkeypatch.setattr(stock_select, "score_raw", decode)
        monkeypatch.setattr(stock_select, "raw_megapixels", lambda p: 45.4)
        monkeypatch.setattr(stock_select, "raw_iso", lambda p: 400)
        by_id = {f["id"]: f for f in stock_select.tag_observation(tmp_path, tmp_path / "floor.json")["frames"]}
        assert by_id["source/raw/DSC_0001.NEF"]["reasons"] == ["unscored"]
        assert by_id["source/raw/DSC_0002.NEF"]["stock_candidate"] is True

    def test_missing_floor_raises(self, tmp_path):
        seed_folder(tmp_path)
        with pytest.raises(StockFloorMissing, match="no sharpness floor at"):
            stock_select.tag_observation(tmp_path, tmp_path / "absent.json")


class TestPipelineHook:
    def test_missing_floor_is_reported_and_logged(self, tmp_path, monkeypatch, caplog):
        def raiser(folder):
            raise StockFloorMissing("no sharpness floor at X; run `bio stock-calibrate`")

        monkeypatch.setattr(stock_select, "tag_observation", raiser)
        with caplog.at_level(logging.WARNING):
            out = pipeline._tag_stock(tmp_path)
        assert out == {"skipped": "no sharpness floor at X; run `bio stock-calibrate`"}
        assert "stock tagging skipped: no sharpness floor at X" in caplog.text

    def test_tags_are_returned_when_a_floor_exists(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(stock_select, "tag_observation", lambda folder: calls.append(folder) or {"frames": [1]})
        assert pipeline._tag_stock(tmp_path) == {"frames": [1]}
        assert calls == [tmp_path]

    def test_other_errors_propagate(self, tmp_path, monkeypatch):
        def boom(folder):
            raise OSError("disk gone")

        monkeypatch.setattr(stock_select, "tag_observation", boom)
        with pytest.raises(OSError, match="disk gone"):
            pipeline._tag_stock(tmp_path)


def queue_item(lc_id, group, raw, iso=800, loc="none", w=8256, h=5504, tmp=None):
    return {"lc_id": lc_id, "group": group, "raw": raw, "day": "2026-07-31", "w": w, "h": h, "loc_source": loc, "copy": str(tmp / f"{lc_id}.jpg")}


def make_copy(path, iso):
    from PIL import Image

    ex = Image.Exif()
    ex.get_ifd(0x8769)[0x8827] = iso
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=ex.tobytes())


class TestTagQueue:
    def seed(self, tmp_path):
        items = [queue_item(1, 1, "2026/d/DSC_1.NEF", tmp=tmp_path), queue_item(2, 1, "2026/d/DSC_2.NEF", tmp=tmp_path),
                 queue_item(3, 1, "2026/d/DSC_3.NEF", tmp=tmp_path), queue_item(4, 2, "2026/d/DSC_4.NEF", tmp=tmp_path, loc="phone")]
        for i, iso in ((1, 100), (2, 200), (3, 6400), (4, 100)):
            make_copy(tmp_path / f"{i}.jpg", iso)
        write_json(tmp_path / "q.json", items)
        write_json(tmp_path / "floor.json", FLOOR)
        return items

    def test_scores_picks_and_writes_the_csv(self, tmp_path, monkeypatch):
        self.seed(tmp_path)
        sharp = {"DSC_1.NEF": 30.0, "DSC_2.NEF": 90.0, "DSC_3.NEF": 500.0, "DSC_4.NEF": 90.0}
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(sharp[p.name], 0.0))
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path / "raws", tmp_path / "out.csv",
                                      floor_path=tmp_path / "floor.json", workers=1, top_per_group=1)
        by = {r["id"]: r for r in rows}
        assert [i for i, r in by.items() if r["stock_candidate"]] == [2]                 # 3 is sharper but ISO 6400; 4 carries a location
        assert by[3]["reasons"] == "ISO 6400 > 3200"
        assert by[4]["reasons"] == "carries a location"
        assert by[1]["reasons"] == "burst: not among the 1 sharpest of its group"
        with open(tmp_path / "out.csv", newline="", encoding="utf-8") as f:
            written = list(csv.DictReader(f))
        assert [(r["id"], r["stock_candidate"], r["day"], r["raw"]) for r in written][1] == ("2", "True", "2026-07-31", "2026/d/DSC_2.NEF")

    def test_excluded_group_never_becomes_a_candidate(self, tmp_path, monkeypatch):
        self.seed(tmp_path)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(90.0, 0.0))
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", floor_path=tmp_path / "floor.json", workers=1,
                                      excluded_groups={1: "flagged pet"})
        by = {r["id"]: r for r in rows}
        assert by[2]["stock_candidate"] is False and by[2]["reasons"] == "group excluded: flagged pet"

    def test_rerun_uses_the_cache_and_does_not_rescore(self, tmp_path, monkeypatch):
        self.seed(tmp_path)
        calls = []
        monkeypatch.setattr(stock_select, "score_raw", lambda p: calls.append(p.name) or FrameScore(90.0, 0.0))
        kw = dict(floor_path=tmp_path / "floor.json", workers=1)
        stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", **kw)
        assert len(calls) == 4
        stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", **kw)
        assert len(calls) == 4

    def test_unreadable_raw_is_an_unscored_row(self, tmp_path, monkeypatch):
        self.seed(tmp_path)

        def decode(p):
            if p.name == "DSC_2.NEF":
                raise RuntimeError("corrupt")
            return FrameScore(90.0, 0.0)

        monkeypatch.setattr(stock_select, "score_raw", decode)
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", floor_path=tmp_path / "floor.json", workers=1)
        assert {r["id"]: r["reasons"] for r in rows}[2] == "unscored"

    def test_limit_scores_only_that_many_and_marks_the_rest_unscored(self, tmp_path, monkeypatch):
        self.seed(tmp_path)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(90.0, 0.0))
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", floor_path=tmp_path / "floor.json", workers=1, limit=2)
        assert [r["reasons"] for r in rows if r["id"] in (3, 4)] == ["unscored", "unscored"]


class TestPublish:
    ROWS = [{"raw": "2026/d/DSC_1.NEF", "day": "2026-07-31", "stock_candidate": "True"},
            {"raw": "2026/d/DSC_2.NEF", "day": "2026-07-31", "stock_candidate": "False"},
            {"raw": "2026/d/DSC_3.NEF", "day": "2026-08-01", "stock_candidate": "True"}]

    @pytest.fixture(autouse=True)
    def floor_file(self, tmp_path, monkeypatch):
        write_json(tmp_path / "pubfloor.json", {**FLOOR, "luma_p10": 35.0})
        monkeypatch.setattr(stock_select, "DEFAULT_FLOOR_PATH", tmp_path / "pubfloor.json")

    def csv_file(self, tmp_path, rows=None):
        p = tmp_path / "c.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["raw", "day", "stock_candidate"])
            w.writeheader()
            w.writerows(rows or self.ROWS)
        return p

    def render(self, tmp_path, day, name, size=(2000, 2000), quality=95, gps=False, color=(90, 120, 60)):
        from PIL import Image, ImageCms

        p = tmp_path / "render" / day[:4] / day / name
        p.parent.mkdir(parents=True, exist_ok=True)
        kw = {"icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()}
        if gps:
            ex = Image.Exif()
            g = ex.get_ifd(0x8825)
            g[1], g[2], g[3], g[4] = "N", (40.0, 0.0, 0.0), "W", (75.0, 0.0, 0.0)
            kw["exif"] = ex.tobytes()
        Image.new("RGB", size, color).save(p, "JPEG", quality=quality, **kw)
        return p

    def test_render_name_carries_the_day(self):
        assert stock_select.render_name(self.ROWS[0]) == "2026-07-31_DSC_1.jpg"

    def test_retouched_renders_are_linked_into_year_month_folders_and_non_candidates_are_not(self, tmp_path):
        src1 = self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg")
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_2.jpg")                       # a render exists for the non-candidate: still not shipped
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg")
        out = stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path / "raws", tmp_path / "render", tmp_path / "gallery")
        assert out == {"candidates": 2, "plain_decoded": 0, "linked": 2, "missing": [], "failed": []}
        assert os.path.samefile(src1, tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_1.jpg")
        assert (tmp_path / "gallery/2026/2026-08/2026-08-01_DSC_3.jpg").is_file()
        assert not (tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_2.jpg").exists()
        again = stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path / "raws", tmp_path / "render", tmp_path / "gallery")
        assert again["linked"] == 0 and again["failed"] == []

    def test_a_candidate_without_a_retouched_render_is_missing_and_never_decoded_plain(self, tmp_path, monkeypatch):
        from bio_automation import render

        decoded = []
        monkeypatch.setattr(render, "export_stock_jpeg", lambda raw, dest: decoded.append(raw))
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg")
        out = stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path / "raws", tmp_path / "render", tmp_path / "gallery")
        assert out["missing"] == ["2026/d/DSC_1.NEF"] and out["linked"] == 1
        assert decoded == []
        assert not (tmp_path / "gallery/2026/2026-07").exists()

    def test_plain_decode_only_runs_when_asked_for(self, tmp_path, monkeypatch):
        from PIL import Image

        from bio_automation import render

        decoded = []

        def fake(raw, dest):
            decoded.append(raw.name)
            self.render(tmp_path, "2026-07-31", dest.name)

        monkeypatch.setattr(render, "export_stock_jpeg", fake)
        out = stock_select.publish_candidates(self.csv_file(tmp_path, [self.ROWS[0]]), tmp_path / "raws", tmp_path / "render",
                                              tmp_path / "gallery", plain_decode=True)
        assert decoded == ["DSC_1.NEF"]
        assert (out["plain_decoded"], out["linked"], out["missing"]) == (1, 1, [])
        assert Image.open(tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_1.jpg").size == (2000, 2000)

    def test_renders_that_break_the_stock_standard_are_failed_not_linked(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", size=(100, 100))                 # under 4 MP
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg", gps=True)                        # carries a GPS block
        out = stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path / "raws", tmp_path / "render", tmp_path / "gallery")
        assert out["linked"] == 0
        assert out["failed"] == [
            {"raw": "2026/d/DSC_1.NEF", "error": "100x100 = 0.01 MP is below the 4 MP minimum"},
            {"raw": "2026/d/DSC_3.NEF", "error": "contains a GPS block (Stock files must not carry coordinates)"}]
        assert not (tmp_path / "gallery").exists()

    def test_dark_render_is_failed_and_one_just_bright_enough_is_linked(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", color=(10, 10, 10))              # median luma 10 < 35
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg", color=(36, 36, 36))              # median luma 36 >= 35
        out = stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path, tmp_path / "render", tmp_path / "gallery")
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "under-exposed: median luma 10.0 < floor 35.0"}]
        assert out["linked"] == 1 and (tmp_path / "gallery/2026/2026-08/2026-08-01_DSC_3.jpg").is_file()

    def test_floor_without_luma_is_refused(self, tmp_path):
        write_json(tmp_path / "pubfloor.json", FLOOR)
        with pytest.raises(StockFloorMissing, match="the sharpness floor has no luma_p10; re-run `bio stock-calibrate"):
            stock_select.publish_candidates(self.csv_file(tmp_path), tmp_path, tmp_path / "render", tmp_path / "gallery")

    def test_low_quality_render_is_failed(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", quality=80)
        out = stock_select.publish_candidates(self.csv_file(tmp_path, [self.ROWS[0]]), tmp_path, tmp_path / "render", tmp_path / "gallery")
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "JPEG quality below 95 (quantisation table coarser than q95)"}]


class TestCli:
    def test_tag_queue_without_a_floor_exits_with_the_fix(self, tmp_path, monkeypatch):
        write_json(tmp_path / "q.json", [])
        monkeypatch.setenv("BIO_STOCK_FLOOR", str(tmp_path / "none.json"))
        monkeypatch.setattr(stock_select, "DEFAULT_FLOOR_PATH", tmp_path / "none.json")
        args = cli.build_parser().parse_args(["stock-tag-queue", str(tmp_path / "q.json"), "--raw-root", str(tmp_path), "--out", str(tmp_path / "o.csv")])
        with pytest.raises(SystemExit, match=r"no sharpness floor at .*none\.json; run `bio stock-calibrate"):
            cli.cmd_stock_tag_queue(args)

    def test_tag_queue_passes_the_notes_groups_as_exclusions(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: seen.update(k) or [{"stock_candidate": False, "group": 1}])
        notes = tmp_path / "n.json"
        notes.write_text('{"92": {"flags": ["not_wildlife", "electronics"], "note": "board"}}', encoding="utf-8")
        args = cli.build_parser().parse_args(["stock-tag-queue", "q.json", "--raw-root", "r", "--out", "o.csv", "--notes", str(notes)])
        cli.cmd_stock_tag_queue(args)
        assert seen["excluded_groups"] == {92: "flagged not_wildlife, electronics"}

    def test_tag_queue_prints_the_counts(self, tmp_path, monkeypatch, capsys):
        rows = [{"stock_candidate": True, "group": 1}, {"stock_candidate": False, "group": 1}, {"stock_candidate": True, "group": 2},
                {"stock_candidate": False, "group": 3}]
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: rows)
        args = cli.build_parser().parse_args(["stock-tag-queue", "q.json", "--raw-root", "r", "--out", "o.csv"])
        assert cli.cmd_stock_tag_queue(args) == 0
        assert capsys.readouterr().out == "2 Stock candidates of 4 frames in 2 of 3 groups -> o.csv\n"

    def test_publish_exit_code_reflects_failures_and_missing_renders(self, monkeypatch):
        seen = []
        result = {"candidates": 1, "plain_decoded": 0, "linked": 0, "missing": [], "failed": [{"raw": "x", "error": "e"}]}
        monkeypatch.setattr(stock_select, "publish_candidates", lambda *a, **k: seen.append(k) or result)
        argv = ["stock-publish", "c.csv", "--raw-root", "r", "--render-dir", "d", "--gallery-dir", "g"]
        args = cli.build_parser().parse_args(argv)
        assert cli.cmd_stock_publish(args) == 1
        result = {**result, "failed": [], "missing": ["m"]}
        assert cli.cmd_stock_publish(args) == 1                                           # a missing render is also not a clean run
        result = {**result, "missing": []}
        assert cli.cmd_stock_publish(args) == 0
        assert seen == [{"plain_decode": False, "floor_path": None}] * 3
        cli.cmd_stock_publish(cli.build_parser().parse_args(argv + ["--plain-decode"]))
        assert seen[-1] == {"plain_decode": True, "floor_path": None}
        cli.cmd_stock_publish(cli.build_parser().parse_args(argv + ["--floor", "f.json"]))
        assert seen[-1] == {"plain_decode": False, "floor_path": Path("f.json")}
