"""inat-gallery: the proposed iNat upload copies, hardlinked into the gallery under inat-pending/<predicted group>/g<group>[_<Species>_p<share>]/."""
from __future__ import annotations

import json
import os
import sys

import pytest

from bio_automation import cli, inat_gallery
from bio_automation.inat_gallery import GalleryLinkError, apply, group_folder, plan
from bio_automation.queue import Queue


def sug(iconic, iconic_share, species=None, species_share=0.0):
    return {"photos": 3, "iconic": [{"iconic": iconic, "vote_share": iconic_share}],
            "species": [{"taxon": species, "vote_share": species_share}] if species else []}


class TestGroupFolder:
    def test_names_the_species_and_its_share_when_both_reach_the_threshold(self):
        assert group_folder(7, sug("Fungi", 0.9, "Amanita muscaria", 0.62)) == "inat-pending/Fungi/g0007_Amanita-muscaria_p62"

    def test_the_threshold_is_inclusive_and_just_under_it_names_no_species(self):
        assert group_folder(7, sug("Fungi", 0.9, "Amanita muscaria", 0.5)) == "inat-pending/Fungi/g0007_Amanita-muscaria_p50"
        assert group_folder(7, sug("Fungi", 0.9, "Amanita muscaria", 0.499)) == "inat-pending/Fungi/g0007"

    def test_an_iconic_group_under_the_threshold_goes_under_unsure_and_just_at_it_is_named(self):
        assert group_folder(12, sug("Insecta", 0.49)) == "inat-pending/Unsure/g0012"
        assert group_folder(12, sug("Insecta", 0.5)) == "inat-pending/Insecta/g0012"

    def test_no_suggestion_at_all_is_unsure_and_the_threshold_is_a_parameter(self):
        assert group_folder(3, None) == "inat-pending/Unsure/g0003"
        assert group_folder(3, sug("Aves", 0.7, "Corvus corax", 0.7), min_share=0.8) == "inat-pending/Unsure/g0003"
        assert group_folder(3, sug("Aves", 0.7, "Corvus corax", 0.7), min_share=0.7) == "inat-pending/Aves/g0003_Corvus-corax_p70"

    def test_the_folder_never_ends_in_a_dash_and_two_digits_so_it_is_never_a_year_month_folder(self):
        for s in (None, sug("Fungi", 0.9), sug("Fungi", 0.9, "Rubus sp. 10", 0.55)):
            assert not __import__("re").search(r"-\d\d$", group_folder(5, s))


def _copy(tmp_path, name, body=b"jpeg"):
    p = tmp_path / "copies" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(body)
    return str(p)


class TestPlan:
    def test_links_every_entry_with_a_copy_and_reports_the_ones_without(self, tmp_path):
        a, b = _copy(tmp_path, "a.jpg"), _copy(tmp_path, "b.jpg")
        gone = {"group": 9, "copy": str(tmp_path / "copies" / "gone.jpg")}
        queue = [{"group": 1, "copy": a}, {"group": 2, "copy": b}, gone, {"group": 4}]
        links, skipped = plan(queue, {"1": sug("Fungi", 0.9, "Amanita muscaria", 0.8)})
        assert links == {"inat-pending/Fungi/g0001_Amanita-muscaria_p80/a.jpg": a, "inat-pending/Unsure/g0002/b.jpg": b}
        assert skipped == [gone, {"group": 4}]

    def test_two_different_copies_with_one_name_in_one_group_are_refused_but_one_copy_listed_twice_is_fine(self, tmp_path):
        a1 = _copy(tmp_path, "a.jpg", b"one")
        other = tmp_path / "other"
        other.mkdir()
        a2 = str(other / "a.jpg")
        open(a2, "wb").write(b"two")
        with pytest.raises(GalleryLinkError, match=r"two different upload copies would both be inat-pending/Unsure/g0001/a\.jpg"):
            plan([{"group": 1, "copy": a1}, {"group": 1, "copy": a2}], {})
        links, _ = plan([{"group": 1, "copy": a1}, {"group": 1, "copy": a1}], {})
        assert links == {"inat-pending/Unsure/g0001/a.jpg": a1}


@pytest.fixture
def env(tmp_path):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    return gallery, tmp_path / "state" / "manifest.json", Queue(tmp_path / "q.sqlite3")


class TestApply:
    def test_hardlinks_the_files_records_them_and_publishes_one_reindex_message_for_the_folders(self, tmp_path, env):
        gallery, manifest, q = env
        a, b = _copy(tmp_path, "a.jpg", b"AAAA"), _copy(tmp_path, "b.jpg", b"BB")
        links = {"inat-pending/Fungi/g0001_Amanita-muscaria_p80/a.jpg": a, "inat-pending/Unsure/g0002/b.jpg": b}
        r = apply(links, gallery, manifest, queue=q)
        assert (r["linked"], r["already_there"], r["removed"], r["refused"]) == (2, 0, 0, {})
        dst = gallery / "inat-pending/Fungi/g0001_Amanita-muscaria_p80/a.jpg"
        assert os.path.samefile(dst, a) and dst.read_bytes() == b"AAAA" and os.stat(a).st_nlink == 2
        assert json.load(open(manifest, encoding="utf-8")) == links
        msg = q.claim(topics=["gallery.reindex"], worker_id="t")
        assert msg.id == r["reindex_message"]
        assert msg.payload == {"gallery_dir": str(gallery), "folders": [
            "inat-pending", "inat-pending/Fungi", "inat-pending/Fungi/g0001_Amanita-muscaria_p80",
            "inat-pending/Unsure", "inat-pending/Unsure/g0002"]}
        assert q.claim(topics=["gallery.reindex"], worker_id="t") is None

    def test_a_second_identical_run_changes_nothing_and_publishes_nothing(self, tmp_path, env):
        gallery, manifest, q = env
        links = {"inat-pending/Unsure/g0002/b.jpg": _copy(tmp_path, "b.jpg")}
        apply(links, gallery, manifest, queue=q)
        q.claim(topics=["gallery.reindex"], worker_id="t")
        r = apply(links, gallery, manifest, queue=q)
        assert (r["linked"], r["already_there"], r["removed"], r["reindex_message"]) == (0, 1, 0, None)
        assert q.claim(topics=["gallery.reindex"], worker_id="t") is None

    def test_a_different_file_already_at_the_destination_is_refused_and_left_intact(self, tmp_path, env):
        gallery, manifest, q = env
        src = _copy(tmp_path, "b.jpg", b"mine")
        rel = "inat-pending/Unsure/g0002/b.jpg"
        (gallery / rel).parent.mkdir(parents=True)
        (gallery / rel).write_bytes(b"someone else's")
        r = apply({rel: src}, gallery, manifest, queue=q)
        assert (gallery / rel).read_bytes() == b"someone else's"
        assert r["refused"] == {rel: "the gallery already holds a different file here (14 bytes); not replaced"}
        assert (r["linked"], r["already_there"], r["reindex_message"]) == (0, 0, None)
        assert json.load(open(manifest, encoding="utf-8")) == {}                    # a refused file is not ours, so it is not in the manifest

    def test_a_regrouped_observation_moves_its_link_and_the_emptied_folders_go(self, tmp_path, env):
        gallery, manifest, q = env
        src = _copy(tmp_path, "a.jpg", b"AA")
        old_rel, new_rel = "inat-pending/Unsure/g0001/a.jpg", "inat-pending/Fungi/g0001_Amanita-muscaria_p80/a.jpg"
        apply({old_rel: src}, gallery, manifest, queue=q)
        r = apply({new_rel: src}, gallery, manifest, queue=q)
        assert (r["linked"], r["removed"]) == (1, 1)
        assert not (gallery / old_rel).exists() and not (gallery / "inat-pending/Unsure").exists()
        assert os.path.samefile(gallery / new_rel, src) and open(src, "rb").read() == b"AA"      # the upload copy itself is untouched
        assert json.load(open(manifest, encoding="utf-8")) == {new_rel: src}
        first = q.claim(topics=["gallery.reindex"], worker_id="t")
        second = q.claim(topics=["gallery.reindex"], worker_id="t")
        assert "inat-pending/Unsure" in first.payload["folders"]
        assert second.payload["folders"] == ["inat-pending", "inat-pending/Fungi", "inat-pending/Fungi/g0001_Amanita-muscaria_p80"]
        assert [f for f in second.payload["folders"] if "Unsure" in f] == []         # a pruned-away folder cannot be indexed, so it is not named

    def test_prune_never_touches_a_file_that_is_not_in_the_manifest(self, tmp_path, env):
        gallery, manifest, q = env
        foreign = gallery / "inat-pending/Unsure/g0009/mine.jpg"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"x")
        apply({"inat-pending/Unsure/g0002/b.jpg": _copy(tmp_path, "b.jpg")}, gallery, manifest, queue=q)
        assert foreign.read_bytes() == b"x"

    def test_prune_leaves_a_manifest_path_alone_when_someone_replaced_the_file_with_another_one(self, tmp_path, env):
        gallery, manifest, q = env
        a, b = _copy(tmp_path, "a.jpg", b"AA"), _copy(tmp_path, "b.jpg", b"BB")
        rel = "inat-pending/Unsure/g0001/a.jpg"
        apply({rel: a}, gallery, manifest, queue=q)
        (gallery / rel).unlink()
        (gallery / rel).write_bytes(b"the owner put their own file here")      # same name, no longer the link of the upload copy
        r = apply({"inat-pending/Unsure/g0002/b.jpg": b}, gallery, manifest, queue=q)
        assert r["removed"] == 0 and (gallery / rel).read_bytes() == b"the owner put their own file here"

    def test_no_prune_keeps_the_old_link_and_its_manifest_entry(self, tmp_path, env):
        gallery, manifest, q = env
        a, b = _copy(tmp_path, "a.jpg"), _copy(tmp_path, "b.jpg")
        apply({"inat-pending/Unsure/g0001/a.jpg": a}, gallery, manifest, queue=q)
        r = apply({"inat-pending/Unsure/g0002/b.jpg": b}, gallery, manifest, queue=q, prune=False)
        assert r["removed"] == 0 and (gallery / "inat-pending/Unsure/g0001/a.jpg").exists()
        assert sorted(json.load(open(manifest, encoding="utf-8"))) == ["inat-pending/Unsure/g0001/a.jpg", "inat-pending/Unsure/g0002/b.jpg"]

    def test_a_failed_hardlink_names_the_same_volume_requirement(self, tmp_path, env, monkeypatch):
        gallery, manifest, q = env
        src = _copy(tmp_path, "a.jpg")

        def refuse(a, b):
            raise OSError(18, "Invalid cross-device link")
        monkeypatch.setattr(inat_gallery.os, "link", refuse)
        with pytest.raises(GalleryLinkError, match=r"cannot hardlink .*a\.jpg.*Invalid cross-device link.*same volume"):
            apply({"inat-pending/Unsure/g0001/a.jpg": src}, gallery, manifest, queue=q)
        assert q.claim(topics=["gallery.reindex"], worker_id="t") is None

    def test_without_a_queue_nothing_is_published_but_the_links_are_made(self, tmp_path, env):
        gallery, manifest, _ = env
        r = apply({"inat-pending/Unsure/g0001/a.jpg": _copy(tmp_path, "a.jpg")}, gallery, manifest)
        assert (r["linked"], r["reindex_message"]) == (1, None)


class TestCli:
    def _files(self, tmp_path):
        a = _copy(tmp_path, "a.jpg")
        q, s = tmp_path / "queue.json", tmp_path / "sug.json"
        q.write_text(json.dumps([{"group": 1, "copy": a}, {"group": 2, "copy": str(tmp_path / "gone.jpg")}]), encoding="utf-8")
        s.write_text(json.dumps({"1": sug("Fungi", 0.9, "Amanita muscaria", 0.8)}), encoding="utf-8")
        return q, s

    def test_dry_run_reports_the_plan_and_creates_no_link_manifest_or_queue_message(self, tmp_path, monkeypatch, capsys):
        q, s = self._files(tmp_path)
        gallery, manifest = tmp_path / "gal", tmp_path / "m.json"
        monkeypatch.setattr(sys, "argv", ["bio", "inat-gallery", "--queue", str(q), "--suggestions", str(s), "--gallery-dir", str(gallery),
                                          "--manifest", str(manifest), "--queue-db", str(tmp_path / "q.sqlite3"), "--dry-run"])
        assert cli.main() == 0
        assert json.loads(capsys.readouterr().out) == {"would_link": 1, "folders": 1, "queue_entries_without_a_copy": 1}
        assert not gallery.exists() and not manifest.exists() and not (tmp_path / "q.sqlite3").exists()

    def test_the_same_command_without_dry_run_links_and_enqueues(self, tmp_path, monkeypatch, capsys):
        q, s = self._files(tmp_path)
        gallery, manifest = tmp_path / "gal", tmp_path / "m.json"
        gallery.mkdir()
        monkeypatch.setattr(sys, "argv", ["bio", "inat-gallery", "--queue", str(q), "--suggestions", str(s), "--gallery-dir", str(gallery),
                                          "--manifest", str(manifest), "--queue-db", str(tmp_path / "q.sqlite3")])
        assert cli.main() == 0
        out = json.loads(capsys.readouterr().out)
        assert (out["linked"], out["queue_entries_without_a_copy"], out["refused"]) == (1, 1, {})
        assert (gallery / "inat-pending/Fungi/g0001_Amanita-muscaria_p80/a.jpg").exists()
        assert out["reindex_message"] == 1
