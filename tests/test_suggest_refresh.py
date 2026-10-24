"""inat_photos_fetch (the photo cache for newly labelled observations) and suggest_refresh (the one entry point that refreshes suggestions and gallery)."""
from __future__ import annotations

import json
import os
import time

import pytest
from test_bioclip_suggest import stage  # noqa: F401  (a fixture)


def obs(oid, taxon, photos):
    return {"id": oid, "taxon": {"name": taxon} if taxon else None,
            "photos": [{"id": pid, "url": f"https://static.inaturalist.org/photos/{pid}/square.{ext}"} for pid, ext in photos]}


class TestMediumUrl:
    def test_the_size_token_becomes_medium_and_the_original_extension_is_kept(self, stage):
        pf = stage("inat_photos_fetch")
        assert pf.medium_url("https://static.inaturalist.org/photos/7/square.jpeg") == "https://static.inaturalist.org/photos/7/medium.jpeg"
        assert pf.medium_url("https://static.inaturalist.org/photos/7/original.png") == "https://static.inaturalist.org/photos/7/medium.png"
        assert pf.medium_url("https://static.inaturalist.org/photos/7/medium.jpg") == "https://static.inaturalist.org/photos/7/medium.jpg"


class TestWanted:
    def test_only_photos_of_labelled_observations_that_are_missing_or_empty_are_wanted_in_export_order(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        (tmp_path / "10.jpg").write_bytes(b"cached")
        (tmp_path / "11.jpg").write_bytes(b"")                       # an empty file is not a cached photo
        observations = [obs(1, "Amanita muscaria", [(10, "jpeg"), (11, "jpg"), (12, "png")]),
                        obs(2, None, [(20, "jpeg")]),                # no taxon: not wanted
                        obs(3, "Apis mellifera", [(30, "jpeg")])]
        assert pf.wanted(observations, str(tmp_path)) == [
            ("11", "https://static.inaturalist.org/photos/11/medium.jpg"),
            ("12", "https://static.inaturalist.org/photos/12/medium.png"),
            ("30", "https://static.inaturalist.org/photos/30/medium.jpeg")]

    def test_when_every_photo_is_cached_nothing_is_wanted(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        (tmp_path / "10.jpg").write_bytes(b"x")
        assert pf.wanted([obs(1, "Amanita muscaria", [(10, "jpeg")])], str(tmp_path)) == []


class TestHighWaterMark:
    def test_only_observations_above_the_mark_are_wanted_and_the_mark_itself_is_excluded(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        observations = [obs(100, "A a", [(1, "jpeg")]), obs(101, "B b", [(2, "jpeg")]), obs(102, "C c", [(3, "jpeg")])]
        assert [pid for pid, _ in pf.wanted(observations, str(tmp_path), min_obs_id=101)] == ["3"]
        assert [pid for pid, _ in pf.wanted(observations, str(tmp_path), min_obs_id=0)] == ["1", "2", "3"]

    def test_the_mark_comes_from_the_calibration_file(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        p = tmp_path / "zeroshot_bins.json"
        p.write_text(json.dumps({"max_obs_id": 4567, "bins": []}), encoding="utf-8")
        assert pf.since_obs_id(str(p)) == 4567

    def test_no_calibration_file_or_no_mark_in_it_is_refused_with_the_way_out(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        with pytest.raises(SystemExit, match=r"nope\.json does not exist; run taxon_zeroshot\.py calibrate once to set the mark, or pass --all"):
            pf.since_obs_id(str(tmp_path / "nope.json"))
        p = tmp_path / "old.json"
        p.write_text(json.dumps({"bins": []}), encoding="utf-8")
        with pytest.raises(SystemExit, match=r"old\.json has no max_obs_id; run taxon_zeroshot\.py calibrate once to set it, or pass --all"):
            pf.since_obs_id(str(p))


class TestFetchMain:
    def _setup(self, tmp_path, stage, monkeypatch, bins):
        pf = stage("inat_photos_fetch")
        ei = stage("embed_index")
        photos = tmp_path / "photos"
        export = tmp_path / "inat_observations.json"
        export.write_text(json.dumps({"observations": [obs(100, "Old o", [(1, "jpeg")]), obs(200, "New n", [(2, "jpeg"), (3, "png")])]}), encoding="utf-8")
        monkeypatch.setitem(ei.CFG, "inat_export", str(export))
        monkeypatch.setitem(ei.CFG["bioclip"], "sources", {"inat_photo": str(photos)})
        if bins is not None:
            (tmp_path / "data" / "zeroshot_bins.json").write_text(json.dumps(bins), encoding="utf-8")
        got = []
        monkeypatch.setattr(pf, "http_get", lambda url: got.append(url) or b"jpegbytes")
        monkeypatch.setattr(pf.time, "sleep", lambda s: None)
        return pf, photos, got

    def test_by_default_only_the_observations_newer_than_the_last_calibration_are_downloaded(self, stage, tmp_path, monkeypatch):
        pf, photos, got = self._setup(tmp_path, stage, monkeypatch, {"max_obs_id": 150, "bins": []})
        monkeypatch.setattr("sys.argv", ["inat_photos_fetch.py"])
        assert pf.main() == 0
        assert got == ["https://static.inaturalist.org/photos/2/medium.jpeg", "https://static.inaturalist.org/photos/3/medium.png"]
        assert sorted(os.listdir(photos)) == ["2.jpg", "3.jpg"]

    def test_all_fetches_every_labelled_observations_photos(self, stage, tmp_path, monkeypatch):
        pf, photos, got = self._setup(tmp_path, stage, monkeypatch, {"max_obs_id": 150, "bins": []})
        monkeypatch.setattr("sys.argv", ["inat_photos_fetch.py", "--all"])
        assert pf.main() == 0 and len(got) == 3
        assert sorted(os.listdir(photos)) == ["1.jpg", "2.jpg", "3.jpg"]

    def test_without_a_mark_nothing_is_downloaded(self, stage, tmp_path, monkeypatch):
        pf, photos, got = self._setup(tmp_path, stage, monkeypatch, None)
        monkeypatch.setattr("sys.argv", ["inat_photos_fetch.py"])
        with pytest.raises(SystemExit, match=r"zeroshot_bins\.json does not exist"):
            pf.main()
        assert got == [] and not photos.exists()


class TestFetch:
    def test_a_download_is_written_atomically_with_no_part_file_left(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        d = tmp_path / "photos"
        d.mkdir()
        dst = str(d / "7.jpg")
        pf.fetch_one("https://x/7", dst, lambda url: b"JPEGDATA")
        assert open(dst, "rb").read() == b"JPEGDATA" and sorted(os.listdir(d)) == ["7.jpg"]

    def test_a_failure_between_writing_and_renaming_leaves_neither_the_photo_nor_a_part_file(self, stage, tmp_path, monkeypatch):
        pf = stage("inat_photos_fetch")
        d = tmp_path / "photos"
        d.mkdir()

        def no_space(src, dst):
            raise OSError("disk full")
        monkeypatch.setattr(pf.os, "replace", no_space)
        with pytest.raises(OSError, match="disk full"):
            pf.fetch_one("https://x/7", str(d / "7.jpg"), lambda url: b"JPEGDATA")
        assert os.listdir(d) == []

    def test_a_failed_or_empty_download_leaves_no_file_at_all(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        d = tmp_path / "photos"
        d.mkdir()

        def boom(url):
            raise OSError("connection reset")
        with pytest.raises(OSError, match="connection reset"):
            pf.fetch_one("https://x/7", str(d / "7.jpg"), boom)
        with pytest.raises(ValueError, match=r"https://x/8: empty response"):
            pf.fetch_one("https://x/8", str(d / "8.jpg"), lambda url: b"")
        assert os.listdir(d) == []                                   # no empty file a later run would take for a cached photo

    def test_run_keeps_going_after_a_failure_reports_it_and_paces_between_downloads(self, stage, tmp_path):
        pf = stage("inat_photos_fetch")
        calls, sleeps, lines = [], [], []

        def get(url):
            calls.append(url)
            if url.endswith("/2"):
                raise OSError("404")
            return b"ok:" + url.encode()
        todo = [("1", "https://x/1"), ("2", "https://x/2"), ("3", "https://x/3")]
        fetched, failed = pf.run(todo, str(tmp_path / "photos"), get, sleep=sleeps.append, pace=0.25, log=lines.append)
        assert (fetched, failed) == (2, ["2"])
        assert calls == ["https://x/1", "https://x/2", "https://x/3"] and sleeps == [0.25, 0.25]      # between downloads, not after the last
        assert sorted(os.listdir(tmp_path / "photos")) == ["1.jpg", "3.jpg"]
        assert lines == ["FAIL 2: OSError: 404"]


CFG = {"python_gpu": "py-gpu", "python_ops": "py-ops", "stages_dir": "stages", "package_src": "pkg-src", "hf_home": "hf",
       "embed_config": "embed.json", "queues": [{"tag": "_lc", "prefix": "g"}, {"tag": "_ex", "prefix": "e"}]}


class TestBuildSteps:
    def test_the_steps_run_in_order_with_the_right_interpreter_and_environment(self, stage):
        sr = stage("suggest_refresh")
        steps = sr.build_steps(CFG, data="data")
        j = os.path.join
        assert [s["name"] for s in steps] == ["fetch-live-observations", "fetch-photos", "embed-photos", "neighbour-bins", "zero-shot-calibrate",
                                              "suggest_lc", "gallery_lc", "suggest_ex", "gallery_ex"]
        gpu = {"HF_HOME": "hf", "HF_HUB_OFFLINE": "1", "EMBED_CONFIG": "embed.json", "PYTHONUNBUFFERED": "1"}
        ops = {"PYTHONPATH": "pkg-src", "PYTHONUNBUFFERED": "1"}
        by = {s["name"]: s for s in steps}
        assert by["fetch-live-observations"]["argv"] == ["py-ops", j("data", "fetch_inat_live.py")] and by["fetch-live-observations"]["env"] == ops
        assert by["embed-photos"]["argv"] == ["py-gpu", j("stages", "embed_index.py"), "--backend", "bioclip2"] and by["embed-photos"]["env"] == gpu
        assert by["zero-shot-calibrate"]["argv"] == ["py-gpu", j("stages", "taxon_zeroshot.py"), "calibrate"]
        assert by["suggest_ex"]["argv"] == ["py-gpu", j("stages", "inat_suggest.py")] and by["suggest_ex"]["env"] == {**gpu, "BIO_QUEUE_TAG": "_ex"}
        assert by["gallery_ex"]["argv"] == ["py-ops", "-m", "bio_automation.cli", "inat-gallery", "--queue", j("data", "inat_queue_ex.json"),
                                            "--suggestions", j("data", "inat_suggestions_ex.json"), "--prefix", "e"]
        assert by["gallery_lc"]["argv"][-2:] == ["--prefix", "g"] and by["gallery_lc"]["env"] == ops

    def test_a_config_without_queues_yields_only_the_five_shared_steps(self, stage):
        sr = stage("suggest_refresh")
        assert len(sr.build_steps({**CFG, "queues": []}, data="data")) == 5


class TestRunSteps:
    STEPS = [{"name": n, "argv": [n], "env": {}} for n in ("a", "b", "c")]

    def test_all_steps_run_in_order_and_the_result_records_each_exit_code(self, stage):
        sr = stage("suggest_refresh")
        seen, lines = [], []
        ticks = iter([0.0, 1.5, 2.0, 5.0, 5.0, 5.25])
        result = sr.run_steps(self.STEPS, lambda argv, env: seen.append(argv[0]) or 0, clock=lambda: next(ticks), log=lines.append)
        assert seen == ["a", "b", "c"]
        assert result == {"ok": True, "failed": None, "steps": [{"name": "a", "rc": 0, "seconds": 1.5}, {"name": "b", "rc": 0, "seconds": 3.0},
                                                                  {"name": "c", "rc": 0, "seconds": 0.2}]}

    def test_the_first_failure_stops_the_run_and_the_later_steps_never_start(self, stage):
        sr = stage("suggest_refresh")
        seen, lines = [], []
        result = sr.run_steps(self.STEPS, lambda argv, env: seen.append(argv[0]) or (3 if argv[0] == "b" else 0), log=lines.append)
        assert seen == ["a", "b"]                                              # c was never run
        assert (result["ok"], result["failed"], [(s["name"], s["rc"]) for s in result["steps"]]) == (False, "b", [("a", 0), ("b", 3)])
        assert lines[-1] == "== b FAILED with exit code 3; the remaining steps are NOT run"


class TestLock:
    def test_a_fresh_lock_blocks_a_second_run_and_a_stale_one_is_taken_over(self, stage, tmp_path):
        sr = stage("suggest_refresh")
        lock = str(tmp_path / "x.lock")
        sr.acquire_lock(lock, pid=lambda: 111)
        assert open(lock).read().split()[0] == "111"
        held_for = lambda seconds: (lambda: os.path.getmtime(lock) + seconds)   # noqa: E731  (a clock that reads `seconds` after the lock was made)
        with pytest.raises(sr.LockError, match=r"x\.lock is held \(5 min old\): another refresh is running"):
            sr.acquire_lock(lock, now=held_for(300), pid=lambda: 222)
        assert open(lock).read().split()[0] == "111"                          # the holder's lock was not disturbed
        sr.acquire_lock(lock, now=held_for(sr.LOCK_STALE_S + 60), pid=lambda: 333)    # twin: the same call succeeds once the lock is stale
        assert open(lock).read().split()[0] == "333"

    def test_a_lock_stamped_a_moment_in_the_future_is_zero_minutes_old_and_still_held(self, stage, tmp_path):
        sr = stage("suggest_refresh")
        lock = str(tmp_path / "x.lock")
        sr.acquire_lock(lock, pid=lambda: 111)
        behind = lambda: os.path.getmtime(lock) - 5                               # noqa: E731  (the runner's clock reads before the file's timestamp)
        with pytest.raises(sr.LockError, match=r"x\.lock is held \(0 min old\): another refresh is running"):
            sr.acquire_lock(lock, now=behind, pid=lambda: 222)
        assert open(lock).read().split()[0] == "111"


class TestMain:
    def _setup(self, tmp_path, stage, monkeypatch, runner):
        sr = stage("suggest_refresh")
        cfg_path = tmp_path / "data" / "suggest_refresh.json"
        cfg_path.write_text(json.dumps({**CFG, "queues": [{"tag": "_lc", "prefix": "g"}]}), encoding="utf-8")
        monkeypatch.setattr(sr, "subprocess_runner", runner)
        return sr

    def test_a_successful_run_writes_the_result_and_log_and_releases_the_lock(self, stage, tmp_path, monkeypatch):
        calls = []
        sr = self._setup(tmp_path, stage, monkeypatch, lambda argv, env: calls.append(argv) or 0)
        assert sr.main() == 0
        data = tmp_path / "data"
        last = json.loads((data / "suggest_refresh_last.json").read_text(encoding="utf-8"))
        assert (last["ok"], last["failed"], len(last["steps"])) == (True, None, 7) and len(calls) == 7
        assert not (data / "suggest_refresh.lock").exists()
        assert (data / "suggest_refresh.log").read_text(encoding="utf-8").strip().endswith("refresh OK")

    def test_a_failing_step_exits_1_names_it_in_the_result_and_still_releases_the_lock(self, stage, tmp_path, monkeypatch):
        calls = []
        sr = self._setup(tmp_path, stage, monkeypatch, lambda argv, env: calls.append(argv) or (2 if len(calls) == 3 else 0))
        assert sr.main() == 1
        last = json.loads((tmp_path / "data" / "suggest_refresh_last.json").read_text(encoding="utf-8"))
        assert (last["ok"], last["failed"], len(calls)) == (False, "embed-photos", 3)
        assert not (tmp_path / "data" / "suggest_refresh.lock").exists()

    def test_a_held_lock_exits_1_and_runs_nothing(self, stage, tmp_path, monkeypatch, capsys):
        calls = []
        sr = self._setup(tmp_path, stage, monkeypatch, lambda argv, env: calls.append(argv) or 0)
        (tmp_path / "data" / "suggest_refresh.lock").write_text("999 earlier\n")
        assert sr.main() == 1 and calls == []
        assert "another refresh is running" in capsys.readouterr().out
        assert (tmp_path / "data" / "suggest_refresh.lock").read_text() == "999 earlier\n"       # not removed by a run that did not take it

    def test_dry_run_prints_the_steps_and_touches_nothing(self, stage, tmp_path, monkeypatch, capsys):
        import sys
        calls = []
        sr = self._setup(tmp_path, stage, monkeypatch, lambda argv, env: calls.append(argv) or 0)
        monkeypatch.setattr(sys, "argv", ["suggest_refresh.py", "--dry-run"])
        assert sr.main() == 0 and calls == []
        assert len(capsys.readouterr().out.strip().splitlines()) == 7
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["suggest_refresh.json"]

    def test_a_config_missing_a_key_is_refused_with_the_names(self, stage, tmp_path, monkeypatch):
        sr = stage("suggest_refresh")
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"python_gpu": "x"}), encoding="utf-8")
        with pytest.raises(SystemExit, match=r"c\.json lacks python_ops, stages_dir, package_src, hf_home, embed_config, queues"):
            sr.load_config(str(p))
