"""BioCLIP 2 taxon suggestions: GPU-only embedder, --backend selection, leave-one-out bins, inat_suggest. No GPU, no network, no real model.

Qdrant is real (embedded local mode in tmp_path); only the CUDA/torch/open_clip surface and the image decoder are faked, and every fake records its calls.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

STAGES = Path(__file__).resolve().parents[1] / "pipeline" / "stages"
_FLAT = ("embed_index", "ort_providers", "bioclip_embed", "embed_loo", "inat_suggest", "taxon_zeroshot")


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Flat stage modules imported the way the scripts do, against a private embed_config.json and BIO_DATA, never leaking into other tests."""
    monkeypatch.syspath_prepend(str(STAGES))
    for name in _FLAT:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for var in ("BIO_BIOCLIP_CUDA_DEVICE", "BIO_ORT_CUDA_DEVICE", "BIO_SUGGEST_BACKEND", "BIO_QUEUE_TAG"):
        monkeypatch.delenv(var, raising=False)
    cfg = {"qdrant_path": str(tmp_path / "qdrant"), "collection": "bio_photos", "batch_size": 4,
           "sources": {"inat_photo": str(tmp_path / "siglip_photos")},
           "bioclip": {"model": "hf-hub:fake/bioclip", "collection": "bio_photos_bioclip2", "cuda_device": "auto"}}
    path = tmp_path / "embed_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("EMBED_CONFIG", str(path))
    monkeypatch.setenv("BIO_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    yield importlib.import_module
    for name in _FLAT:
        sys.modules.pop(name, None)


class FakeCuda:
    def __init__(self, free_bytes=(), available=True):
        self.free_bytes = list(free_bytes)
        self.available = available
        self.mem_calls = []

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.free_bytes)

    def mem_get_info(self, i):
        self.mem_calls.append(i)
        return (self.free_bytes[i], 12 * 2**30)


class FakeBatch:
    def __init__(self, items):
        self.items, self.moved_to, self.half_called = items, None, False

    def to(self, device):
        self.moved_to = device
        return self

    def half(self):
        self.half_called = True
        return self


class FakeTorch:
    """The torch surface bioclip_embed uses; stack/to/half are recorded, the 'tensor' is the list of per-image vectors."""

    def __init__(self, free_bytes=(2 * 2**30, 9 * 2**30), available=True):
        self.cuda = FakeCuda(free_bytes, available)
        self.batches = []

    def stack(self, items):
        b = FakeBatch(list(items))
        self.batches.append(b)
        return b

    def no_grad(self):
        return contextlib.nullcontext()


class FakeOut:
    def __init__(self, arr):
        self.arr = arr

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeModel:
    def __init__(self):
        self.events = []

    def to(self, device):
        self.events.append(("to", device))
        return self

    def half(self):
        self.events.append(("half",))
        return self

    def eval(self):
        self.events.append(("eval",))
        return self

    def encode_image(self, batch):
        self.events.append(("encode", len(batch.items)))
        return FakeOut(np.array(batch.items, dtype=np.float32))


class FakeOpenClip:
    def __init__(self):
        self.model = FakeModel()
        self.created = []

    def create_model_and_transforms(self, model_id):
        self.created.append(model_id)
        return self.model, None, (lambda image: image)


CFG = {"model": "hf-hub:fake/bioclip", "collection": "c", "cuda_device": "auto"}


# ---------------------------------------------------------------- device choice

class TestPickDevice:
    def test_auto_takes_the_card_with_the_most_free_vram_not_the_first(self, stage):
        be = stage("bioclip_embed")
        t = FakeTorch(free_bytes=(2 * 2**30, 9 * 2**30, 5 * 2**30))
        assert be.pick_device(t, "auto", environ={}) == "cuda:1"
        assert t.cuda.mem_calls == [0, 1, 2]

    def test_auto_is_the_default_and_picks_the_first_when_it_has_the_most_free(self, stage):
        be = stage("bioclip_embed")
        assert be.pick_device(FakeTorch(free_bytes=(9 * 2**30, 2 * 2**30)), None, environ={}) == "cuda:0"

    def test_configured_index_is_used_without_probing_memory(self, stage):
        be = stage("bioclip_embed")
        t = FakeTorch(free_bytes=(9 * 2**30, 2 * 2**30))
        assert be.pick_device(t, 1, environ={}) == "cuda:1"
        assert t.cuda.mem_calls == []

    def test_env_overrides_the_configured_index(self, stage):
        be = stage("bioclip_embed")
        assert be.pick_device(FakeTorch(), 0, environ={"BIO_BIOCLIP_CUDA_DEVICE": "1"}) == "cuda:1"

    def test_no_cuda_is_an_error_not_a_cpu_fallback(self, stage):
        be = stage("bioclip_embed")
        with pytest.raises(RuntimeError, match="CUDA is not available to torch; this stage runs on the GPU only"):
            be.pick_device(FakeTorch(available=False), "auto", environ={})

    def test_index_beyond_the_cards_torch_sees_is_an_error(self, stage):
        be = stage("bioclip_embed")
        with pytest.raises(RuntimeError, match=r"cuda device 2 requested but torch sees 2 CUDA device\(s\)"):
            be.pick_device(FakeTorch(), 2, environ={})
        assert be.pick_device(FakeTorch(), 1, environ={}) == "cuda:1"

    def test_non_integer_device_is_an_error(self, stage):
        be = stage("bioclip_embed")
        with pytest.raises(RuntimeError, match=r"cuda device 'gpu0' is not an integer or \"auto\""):
            be.pick_device(FakeTorch(), "gpu0", environ={})


# ---------------------------------------------------------------- embedder

class TestBioClipEmbedder:
    def test_model_is_loaded_by_the_configured_id_and_moved_to_the_chosen_card_in_half_precision(self, stage):
        be = stage("bioclip_embed")
        oc, t = FakeOpenClip(), FakeTorch(free_bytes=(1 * 2**30, 8 * 2**30))
        emb = be.BioClipEmbedder(CFG, torch_mod=t, open_clip_mod=oc, load_image=lambda p: p, environ={})
        assert oc.created == ["hf-hub:fake/bioclip"]
        assert oc.model.events == [("to", "cuda:1"), ("half",), ("eval",)]
        assert emb.device == "cuda:1"

    def test_no_cuda_stops_before_any_model_is_created(self, stage):
        be = stage("bioclip_embed")
        oc = FakeOpenClip()
        with pytest.raises(RuntimeError, match="CUDA is not available to torch"):
            be.BioClipEmbedder(CFG, torch_mod=FakeTorch(available=False), open_clip_mod=oc, load_image=lambda p: p, environ={})
        assert oc.created == []

    def test_embed_returns_unit_vectors_in_input_order_and_batches_on_the_device(self, stage):
        be = stage("bioclip_embed")
        images = {"a.jpg": [3.0, 4.0, 0.0], "b.jpg": [0.0, 0.0, 2.0], "c.jpg": [1.0, 0.0, 0.0]}
        oc, t = FakeOpenClip(), FakeTorch(free_bytes=(8 * 2**30,))
        emb = be.BioClipEmbedder(CFG, torch_mod=t, open_clip_mod=oc, load_image=images.__getitem__, environ={})
        out = emb.embed(["a.jpg", "b.jpg", "c.jpg"])
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], atol=1e-6)
        (batch,) = t.batches
        assert (batch.moved_to, batch.half_called, len(batch.items)) == ("cuda:0", True, 3)

    def test_an_all_zero_vector_stays_zero_instead_of_becoming_nan(self, stage):
        be = stage("bioclip_embed")
        emb = be.BioClipEmbedder(CFG, torch_mod=FakeTorch(free_bytes=(8 * 2**30,)), open_clip_mod=FakeOpenClip(),
                                 load_image=lambda p: [0.0, 0.0], environ={})
        np.testing.assert_array_equal(emb.embed(["z.jpg"]), [[0.0, 0.0]])


# ---------------------------------------------------------------- embed_index --backend

class TestBackendSetup:
    def test_siglip_is_the_default_index_and_uses_the_top_level_config(self, stage):
        ei = stage("embed_index")
        made = []
        emb, collection, sources, batch = ei.backend_setup("siglip", embedder_cls=lambda: made.append("siglip") or "SIGLIP")
        assert (emb, collection, batch, made) == ("SIGLIP", "bio_photos", 4, ["siglip"])
        assert list(sources) == ["inat_photo"] and sources["inat_photo"].endswith("siglip_photos")

    def test_bioclip2_uses_its_own_collection_and_model_config(self, stage):
        ei = stage("embed_index")
        seen = []
        emb, collection, sources, batch = ei.backend_setup("bioclip2", bioclip_cls=lambda bc: seen.append(bc) or "BIOCLIP")
        assert (emb, collection, batch) == ("BIOCLIP", "bio_photos_bioclip2", 4)
        assert seen == [{"model": "hf-hub:fake/bioclip", "collection": "bio_photos_bioclip2", "cuda_device": "auto"}]
        assert sources["inat_photo"].endswith("siglip_photos")

    def test_bioclip2_sources_and_batch_size_override_the_top_level_ones(self, stage):
        ei = stage("embed_index")
        cfg = dict(ei.CFG, bioclip={"model": "m", "collection": "col", "sources": {"inat_photo": "D:/photos"}, "batch_size": 32})
        _, collection, sources, batch = ei.backend_setup("bioclip2", cfg=cfg, bioclip_cls=lambda bc: None)
        assert (collection, sources, batch) == ("col", {"inat_photo": "D:/photos"}, 32)

    def test_unknown_backend_is_refused_and_builds_no_embedder(self, stage):
        ei = stage("embed_index")
        built = []
        with pytest.raises(SystemExit, match=r"unknown --backend 'clip' \(siglip or bioclip2\)"):
            ei.backend_setup("clip", embedder_cls=lambda: built.append(1), bioclip_cls=lambda bc: built.append(2))
        assert built == []


# ---------------------------------------------------------------- leave-one-out + bins

def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestLeaveOneOut:
    # A1 and A2 are burst siblings (nearly identical vectors); C (species Y) is the nearest OTHER photo to A1; B (species X) is farther.
    VECS = [_unit(1, 0.00), _unit(1, 0.01), _unit(1, 0.30), _unit(1, 1.50)]   # A1, A2, C, B
    LABELS = np.array(["X", "X", "Y", "X"])

    def test_a_sibling_of_the_same_observation_is_never_a_neighbour(self, stage):
        lo = stage("embed_loo")
        obs, day = np.array([1, 1, 3, 2]), np.array(["d1", "d1", "d3", "d2"])
        conf, hit = lo.leave_one_out(self.VECS, obs, day, self.LABELS, k=1)
        assert list(hit) == [False, False, False, False] and list(conf) == [1.0, 1.0, 1.0, 1.0]

    def test_same_fixture_with_the_sibling_in_another_observation_and_day_it_is_the_neighbour(self, stage):
        lo = stage("embed_loo")
        obs, day = np.array([1, 5, 3, 2]), np.array(["d1", "d5", "d3", "d2"])
        _, hit = lo.leave_one_out(self.VECS, obs, day, self.LABELS, k=1)
        assert list(hit) == [True, True, False, False]   # A1<->A2 now vote for each other; C and B still lose

    def test_a_sibling_of_the_same_observation_is_excluded_even_when_taken_on_another_day(self, stage):
        lo = stage("embed_loo")
        obs, day = np.array([1, 1, 3, 2]), np.array(["d1", "d9", "d3", "d2"])   # A1 and A2: one observation, two days
        _, hit = lo.leave_one_out(self.VECS, obs, day, self.LABELS, k=1)
        assert list(hit)[:2] == [False, False]
        obs2 = np.array([1, 5, 3, 2])                                            # twin: same days, siblings now in different observations
        _, hit2 = lo.leave_one_out(self.VECS, obs2, day, self.LABELS, k=1)
        assert list(hit2)[:2] == [True, True]

    def test_a_neighbour_from_the_same_day_is_excluded_even_in_another_observation(self, stage):
        lo = stage("embed_loo")
        obs, day = np.array([1, 5, 3, 2]), np.array(["d1", "d1", "d3", "d2"])
        _, hit = lo.leave_one_out(self.VECS, obs, day, self.LABELS, k=1)
        assert list(hit)[:2] == [False, False]

    def test_unlabelled_queries_are_skipped_and_unlabelled_neighbours_do_not_vote(self, stage):
        lo = stage("embed_loo")
        obs, day = np.array([1, 5, 3, 2]), np.array(["d1", "d5", "d3", "d2"])
        labels = np.array(["X", "", "Y", "X"])
        conf, hit = lo.leave_one_out(self.VECS, obs, day, labels, k=2)
        # queries A1, C, B (3 labelled). A1's 2 nearest are A2 (unlabelled, ignored) and C -> only C votes: Y != X
        assert (len(conf), list(hit)) == (3, [False, False, False])


class TestBins:
    def test_bins_hold_exact_counts_and_accuracies_with_half_open_edges(self, stage):
        lo = stage("embed_loo")
        conf = [0.1, 0.2, 0.3, 0.5, 0.5, 0.7, 0.9, 1.0]
        hit = [True, False, True, True, False, False, True, True]
        bins = lo.bins_from_votes(conf, hit)
        assert bins == [
            {"lo": 0.0, "hi": 0.3, "n": 2, "accuracy": 0.5},
            {"lo": 0.3, "hi": 0.5, "n": 1, "accuracy": 1.0},
            {"lo": 0.5, "hi": 0.7, "n": 2, "accuracy": 0.5},
            {"lo": 0.7, "hi": 0.9, "n": 1, "accuracy": 0.0},
            {"lo": 0.9, "hi": 1.01, "n": 2, "accuracy": 1.0},
        ]

    def test_an_empty_bin_has_no_accuracy_rather_than_zero(self, stage):
        lo = stage("embed_loo")
        assert lo.bins_from_votes([0.1], [True])[3] == {"lo": 0.7, "hi": 0.9, "n": 0, "accuracy": None}

    def test_write_bins_keeps_the_other_backend_and_replaces_its_own(self, stage, tmp_path):
        lo = stage("embed_loo")
        path = str(tmp_path / "suggest_bins.json")
        lo.write_bins(path, "siglip", "bio_photos", [{"lo": 0.0, "hi": 1.01, "n": 4, "accuracy": 0.25}], 4)
        lo.write_bins(path, "bioclip2", "bio_photos_bioclip2", [{"lo": 0.0, "hi": 1.01, "n": 9, "accuracy": 0.5}], 9)
        lo.write_bins(path, "bioclip2", "bio_photos_bioclip2", [{"lo": 0.0, "hi": 1.01, "n": 8, "accuracy": 0.75}], 8)
        data = json.load(open(path, encoding="utf-8"))
        assert data["siglip"] == {"collection": "bio_photos", "k": 15, "queries": 4,
                                  "species_bins": [{"lo": 0.0, "hi": 1.01, "n": 4, "accuracy": 0.25}]}
        assert data["bioclip2"]["queries"] == 8 and data["bioclip2"]["species_bins"][0]["accuracy"] == 0.75
        assert not (tmp_path / "suggest_bins.json.tmp").exists()


# ---------------------------------------------------------------- inat_suggest

BINS = [(0.0, 0.3, 0.15), (0.3, 0.7, 0.4), (0.7, 1.01, 0.85)]


class TestMeasuredAndBinsFile:
    def test_measured_picks_the_bin_with_the_boundary_in_the_upper_one(self, stage):
        su = stage("inat_suggest")
        assert [su.measured(s, BINS) for s in (0.0, 0.299, 0.3, 0.69, 0.7, 1.0)] == [0.15, 0.15, 0.4, 0.4, 0.85, 0.85]
        assert su.measured(1.5, BINS) is None

    def test_load_bins_reads_the_named_backend(self, stage, tmp_path):
        su = stage("inat_suggest")
        p = tmp_path / "suggest_bins.json"
        p.write_text(json.dumps({"bioclip2": {"species_bins": [{"lo": 0.0, "hi": 1.01, "n": 3, "accuracy": 0.5}]},
                                 "siglip": {"species_bins": [{"lo": 0.0, "hi": 1.01, "n": 3, "accuracy": 0.1}]}}), encoding="utf-8")
        assert su.load_bins(str(p), "bioclip2") == [(0.0, 1.01, 0.5)]
        assert su.load_bins(str(p), "siglip") == [(0.0, 1.01, 0.1)]

    def test_missing_file_and_missing_backend_both_refuse_with_the_command_to_run(self, stage, tmp_path):
        su = stage("inat_suggest")
        with pytest.raises(SystemExit, match=r"does not exist; run embed_loo.py --backend bioclip2 first"):
            su.load_bins(str(tmp_path / "nope.json"), "bioclip2")
        p = tmp_path / "suggest_bins.json"
        p.write_text(json.dumps({"siglip": {"species_bins": []}}), encoding="utf-8")
        with pytest.raises(SystemExit, match=r"has no bins for backend 'bioclip2'; run embed_loo.py --backend bioclip2 first"):
            su.load_bins(str(p), "bioclip2")


def _neigh(taxon, iconic, rank="species", common=None):
    return {"taxon": taxon, "iconic": iconic, "rank": rank, "common": common}


class TestSuggestGroups:
    def test_votes_pool_per_group_with_exact_shares_ranking_and_measured_accuracy(self, stage):
        su = stage("inat_suggest")
        photo1 = [_neigh("Amanita muscaria", "Fungi", common="fly agaric")] * 3 + [_neigh("Boletus edulis", "Fungi")]
        photo2 = [_neigh("Amanita muscaria", "Fungi", common="fly agaric")] * 2 + [_neigh("Boletus edulis", "Fungi")] * 2
        out = su.suggest_groups({7: [photo1, photo2]}, {"Amanita muscaria": 47169}, BINS)
        assert out == {"7": {
            "photos": 2,
            "species": [
                {"taxon": "Amanita muscaria", "taxon_id": 47169, "common": "fly agaric", "iconic": "Fungi", "vote_share": 0.625, "measured_accuracy": 0.4},
                {"taxon": "Boletus edulis", "taxon_id": None, "common": None, "iconic": "Fungi", "vote_share": 0.375, "measured_accuracy": 0.4},
            ],
            "iconic": [{"iconic": "Fungi", "vote_share": 1.0}],
        }}

    def test_genus_rank_neighbours_vote_for_iconic_only_never_as_a_species(self, stage):
        su = stage("inat_suggest")
        out = su.suggest_groups({1: [[_neigh("Amanita", "Fungi", rank="genus")] * 2 + [_neigh("Apis mellifera", "Insecta")] * 2]}, {}, BINS)
        assert [s["taxon"] for s in out["1"]["species"]] == ["Apis mellifera"]
        assert out["1"]["iconic"] == [{"iconic": "Fungi", "vote_share": 0.5}, {"iconic": "Insecta", "vote_share": 0.5}]


class FakeEmbedder:
    """Maps each copy path to a fixed vector and records every batch it is asked to embed."""

    def __init__(self, vectors):
        self.vectors, self.batches = vectors, []

    def embed(self, paths):
        self.batches.append(list(paths))
        return np.array([self.vectors[Path(p).name] for p in paths], dtype=np.float32)


def install_fake_qdrant(monkeypatch, collections):
    """A recording stand-in for the qdrant_client module (CI has no qdrant-client): {collection: [(vector, payload), ...]} served through scroll()
    with the real call shape (collection, limit, with_payload, with_vectors, offset) -> (points, next_offset). Returns the ordered call list."""
    calls = []

    class FakeClient:
        def __init__(self, path=None, **kw):
            calls.append(("open", path))

        def scroll(self, collection, limit, with_payload, with_vectors, offset=None):
            calls.append(("scroll", collection, offset))
            pts = collections[collection]
            start = offset or 0
            end = min(start + limit, len(pts))
            return [SimpleNamespace(vector=v, payload=p) for v, p in pts[start:end]], (end if end < len(pts) else None)

    mod = SimpleNamespace(QdrantClient=FakeClient)
    monkeypatch.setitem(sys.modules, "qdrant_client", mod)
    return calls


class TestInatSuggestMain:
    def _setup(self, tmp_path, stage, monkeypatch, with_copy, reference=()):
        su = stage("inat_suggest")
        ei = stage("embed_index")
        data = tmp_path / "data"
        copy = tmp_path / "g1.jpg"
        if with_copy:
            copy.write_bytes(b"jpeg")
        (data / "inat_queue.json").write_text(json.dumps([{"group": 7, "copy": str(copy)}, {"group": 8, "copy": str(tmp_path / "gone.jpg")}]),
                                              encoding="utf-8")
        (data / "inat_observations.json").write_text(json.dumps({"observations": [{"taxon": {"name": "Amanita muscaria", "id": 47169}}]}), encoding="utf-8")
        (data / "suggest_bins.json").write_text(json.dumps({"bioclip2": {"species_bins": [{"lo": 0.0, "hi": 1.01, "n": 5, "accuracy": 0.6}]}}),
                                                encoding="utf-8")
        old = data / "inat_suggestions.json"
        old.write_text('{"7": {"photos": 1, "species": [], "iconic": []}}', encoding="utf-8")
        emb = FakeEmbedder({"g1.jpg": [1.0, 0.0]})
        monkeypatch.setattr(ei, "backend_setup", lambda name: (emb, "bio_photos_bioclip2", {}, 4))
        calls = install_fake_qdrant(monkeypatch, {"bio_photos_bioclip2": list(reference)})
        return su, emb, old, calls

    def test_no_upload_copy_on_disk_writes_nothing_and_keeps_the_previous_suggestions(self, stage, tmp_path, monkeypatch):
        su, emb, old, calls = self._setup(tmp_path, stage, monkeypatch, with_copy=False)
        before = old.read_text(encoding="utf-8")
        with pytest.raises(SystemExit, match=r"none of the 2 queue entries has an upload copy on disk; nothing written"):
            su.main()
        assert old.read_text(encoding="utf-8") == before
        assert emb.batches == [] and calls == []                       # no photo embedded and the reference store never opened
        assert not (tmp_path / "data" / "inat_suggestions.meta.json").exists()

    def test_same_queue_with_a_copy_present_votes_over_the_reference_and_writes_suggestions_and_meta(self, stage, tmp_path, monkeypatch):
        reference = [
            ((1.0, 0.0), {"source": "inat_photo", "key": "p1", "obs_id": 1, "taxon": "Amanita muscaria", "rank": "species", "iconic": "Fungi", "common": "fly agaric"}),
            ((0.9, 0.1), {"source": "inat_photo", "key": "p2", "obs_id": 2, "taxon": "Amanita muscaria", "rank": "species", "iconic": "Fungi", "common": "fly agaric"}),
            ((0.0, 1.0), {"source": "inat_photo", "key": "p3", "obs_id": 3, "taxon": "Apis mellifera", "rank": "species", "iconic": "Insecta"}),
            ((0.5, 0.5), {"source": "library_render", "key": "17"}),                     # not an iNat photo: never a voter
        ]
        su, emb, old, calls = self._setup(tmp_path, stage, monkeypatch, with_copy=True, reference=reference)
        su.main()
        assert calls == [("open", str(tmp_path / "qdrant")), ("scroll", "bio_photos_bioclip2", None)]
        assert emb.batches == [[str(tmp_path / "g1.jpg")]]
        out = json.loads(old.read_text(encoding="utf-8"))
        # K=15 > 3 reference photos, so every reference photo is a neighbour: 2 of 3 vote Amanita, 1 of 3 Apis
        assert out == {"7": {"photos": 1, "species": [
            {"taxon": "Amanita muscaria", "taxon_id": 47169, "common": "fly agaric", "iconic": "Fungi", "vote_share": 0.667, "measured_accuracy": 0.6},
            {"taxon": "Apis mellifera", "taxon_id": None, "common": None, "iconic": "Insecta", "vote_share": 0.333, "measured_accuracy": 0.6}],
            "iconic": [{"iconic": "Fungi", "vote_share": 0.667}, {"iconic": "Insecta", "vote_share": 0.333}]}}
        meta = json.loads((tmp_path / "data" / "inat_suggestions.meta.json").read_text(encoding="utf-8"))
        assert meta == {"backend": "bioclip2", "collection": "bio_photos_bioclip2", "reference_photos": 3, "queue_entries": 2,
                        "entries_with_copy": 1, "groups": 1, "zero_shot_index": None}
