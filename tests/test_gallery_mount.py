"""gallery_mount.ensure_results_mount on real folders and real junctions under tmp_path."""
import os

import pytest

from bio_automation import gallery_mount as gm


def test_creates_the_results_folder_and_a_junction_to_it(tmp_path):
    gallery, results = tmp_path / "F" / "gallery", tmp_path / "D" / "gallery"
    gallery.mkdir(parents=True)
    mount = gm.ensure_results_mount(gallery, results, "results")
    assert mount == gallery / "results"
    assert os.path.isjunction(mount)
    (results / "2025").mkdir()
    assert (mount / "2025").is_dir()                      # the gallery sees what the results folder holds


def test_an_existing_correct_junction_is_reused(tmp_path):
    gallery, results = tmp_path / "gallery", tmp_path / "results-gallery"
    gallery.mkdir()
    gm.ensure_results_mount(gallery, results, "results")
    calls = []
    assert gm.ensure_results_mount(gallery, results, "results", create=lambda *a: calls.append(a)) == gallery / "results"
    assert calls == []


def test_a_real_folder_at_the_mount_path_is_refused(tmp_path):
    gallery = tmp_path / "gallery"
    (gallery / "results").mkdir(parents=True)
    with pytest.raises(gm.GalleryMountError, match=r"results exists and is not a junction: refusing to publish into it"):
        gm.ensure_results_mount(gallery, tmp_path / "results-gallery", "results")


def test_a_junction_to_another_folder_is_refused(tmp_path):
    gallery, other = tmp_path / "gallery", tmp_path / "other"
    gallery.mkdir()
    gm.ensure_results_mount(gallery, other, "results")
    with pytest.raises(gm.GalleryMountError, match=r"is a junction to .*other, not to .*results-gallery"):
        gm.ensure_results_mount(gallery, tmp_path / "results-gallery", "results")
