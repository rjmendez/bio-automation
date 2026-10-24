"""taxon_zeroshot: the species text index built from iNaturalist's taxonomy DwC-A, and open-set zero-shot scoring against it. No GPU, no model, no network."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import zipfile

import numpy as np
import pytest
from test_bioclip_suggest import CFG, FakeCuda, FakeModel, FakeOpenClip, FakeTorch, stage  # noqa: F401  (stage is a fixture)

HEADER = ["id", "taxonID", "identifier", "parentNameUsageID", "kingdom", "phylum", "class", "order", "family", "genus",
          "specificEpithet", "infraspecificEpithet", "modified", "scientificName", "taxonRank", "references"]


def row(id_, name, rank="species", kingdom="Animalia", phylum="Chordata", klass="Aves", order="Gruiformes", family="Aramidae", genus="Aramus"):
    return [id_, "", "", "", kingdom, phylum, klass, order, family, genus, "", "", "", name, rank, ""]


def make_dwca(tmp_path, rows):
    path = tmp_path / "dwca.zip"
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows(rows)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("taxa.csv", buf.getvalue())
        z.writestr("VernacularNames-english.csv", "id,vernacularName\n7,limpkin\n")
    return str(path)


ROWS = [
    row("7", "Aramus guarauna"),
    row("8", "Aramus", rank="genus"),                                           # not a species
    row("9", "Aramus guarauna dolosus", rank="subspecies"),                     # not a species
    row("10", "Bubo virginianus", family="Strigidae", order="Strigiformes", genus="Bubo"),
    row("11", "Mystery species", kingdom=""),                                   # no kingdom
    row("12", "Nobinomial"),                                                    # no epithet in the name
    row("13", "Amanita muscaria", kingdom="Fungi", phylum="Basidiomycota", klass="Agaricomycetes", order="", family="Amanitaceae", genus="Amanita"),
]


class TestSpeciesRows:
    def test_only_species_with_a_kingdom_and_a_binomial_name_are_yielded(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        rows = list(tz.species_rows(make_dwca(tmp_path, ROWS)))
        assert [(r["id"], r["name"]) for r in rows] == [(7, "Aramus guarauna"), (10, "Bubo virginianus"), (13, "Amanita muscaria")]
        assert rows[1] == {"id": 10, "name": "Bubo virginianus", "genus": "Bubo", "kingdom": "Animalia", "phylum": "Chordata",
                           "class": "Aves", "order": "Strigiformes", "family": "Strigidae"}

    def test_limit_stops_after_that_many_species_and_zero_means_all(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        z = make_dwca(tmp_path, ROWS)
        assert [r["id"] for r in tz.species_rows(z, limit=2)] == [7, 10]
        assert [r["id"] for r in tz.species_rows(z, limit=0)] == [7, 10, 13]


class TestPrompt:
    def test_prompt_is_the_taxonomic_string_ending_in_the_binomial(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        first = next(tz.species_rows(make_dwca(tmp_path, ROWS)))
        assert tz.prompt(first) == "a photo of Animalia Chordata Aves Gruiformes Aramidae Aramus guarauna."

    def test_an_empty_rank_is_left_out_instead_of_leaving_a_double_space(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        fungus = list(tz.species_rows(make_dwca(tmp_path, ROWS)))[2]
        assert tz.prompt(fungus) == "a photo of Fungi Basidiomycota Agaricomycetes Amanitaceae Amanita muscaria."


class RecordingEncoder:
    """encode_text stand-in: one fixed 2-d vector per prompt (by its position in the call order) and a record of every batch it was given."""

    def __init__(self, vectors):
        self.vectors, self.batches, self.n = vectors, [], 0

    def __call__(self, texts):
        self.batches.append(list(texts))
        out = np.array(self.vectors[self.n:self.n + len(texts)], dtype=np.float16)
        self.n += len(texts)
        return out


class TestBuildIndex:
    def test_writes_vectors_species_and_meta_with_exact_contents_in_chunks(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        zip_path, out = make_dwca(tmp_path, ROWS), str(tmp_path / "index")
        enc, progress = RecordingEncoder([[1, 0], [0, 1], [0.5, 0.5]]), []
        n = tz.build_index(zip_path, out, enc, chunk=2, progress=lambda d, t: progress.append((d, t)))
        assert n == 3 and progress == [(2, 3), (3, 3)]
        assert [len(b) for b in enc.batches] == [2, 1]
        assert enc.batches[0][0] == "a photo of Animalia Chordata Aves Gruiformes Aramidae Aramus guarauna."
        vecs = np.load(f"{out}/species_text.npy")
        assert vecs.dtype == np.float16 and vecs.tolist() == [[1, 0], [0, 1], [0.5, 0.5]]
        assert json.load(open(f"{out}/species.json", encoding="utf-8")) == [
            [7, "Aramus guarauna", "Animalia", "Chordata", "Aves", "Gruiformes", "Aramidae", "Aramus"],
            [10, "Bubo virginianus", "Animalia", "Chordata", "Aves", "Strigiformes", "Strigidae", "Bubo"],
            [13, "Amanita muscaria", "Fungi", "Basidiomycota", "Agaricomycetes", "", "Amanitaceae", "Amanita"]]
        assert json.load(open(f"{out}/meta.json", encoding="utf-8")) == {
            "source": "dwca.zip", "source_bytes": (tmp_path / "dwca.zip").stat().st_size, "species": 3, "dim": 2}
        assert sorted(p.name for p in (tmp_path / "index").iterdir()) == ["meta.json", "species.json", "species_text.npy"]   # no .tmp left behind

    def test_a_taxonomy_with_no_species_is_an_error_and_nothing_is_encoded(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        enc = RecordingEncoder([])
        with pytest.raises(ValueError, match=r"dwca\.zip: no species rows found in taxa\.csv"):
            tz.build_index(make_dwca(tmp_path, [row("8", "Aramus", rank="genus")]), str(tmp_path / "index"), enc)
        assert enc.batches == [] and not (tmp_path / "index").exists()

    def test_an_encoder_that_returns_the_wrong_number_of_vectors_is_an_error_and_writes_nothing(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        with pytest.raises(ValueError, match=r"encoder returned 2 vectors for 3 species"):
            tz.build_index(make_dwca(tmp_path, ROWS), str(tmp_path / "index"), lambda texts: np.zeros((2, 2), dtype=np.float16))
        assert not (tmp_path / "index").exists()


class NpTensor:
    """The slice of a torch tensor ZeroShot uses, on numpy. `log` records every device move."""

    def __init__(self, a, log):
        self.a, self.log = a, log
        self.dtype = a.dtype

    def to(self, target):
        if isinstance(target, str):
            self.log.append(target)
            return self
        return NpTensor(self.a.astype(target), self.log)

    @property
    def T(self):
        return NpTensor(self.a.T, self.log)

    def __matmul__(self, o):
        return NpTensor(self.a.astype(np.float32) @ o.a.astype(np.float32), self.log)

    def float(self):
        return NpTensor(self.a.astype(np.float32), self.log)

    def __mul__(self, s):
        return NpTensor(self.a * s, self.log)

    def __sub__(self, o):
        return NpTensor(self.a - o.a, self.log)

    def exp(self):
        return NpTensor(np.exp(self.a), self.log)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class NpTorch:
    def __init__(self):
        self.log = []

    def from_numpy(self, a):
        return NpTensor(np.asarray(a), self.log)

    def no_grad(self):
        return contextlib.nullcontext()

    def logsumexp(self, x, dim, keepdim):
        m = x.a.max(axis=dim, keepdims=True)
        return NpTensor(m + np.log(np.exp(x.a - m).sum(axis=dim, keepdims=True)), self.log)

    def topk(self, x, k, dim):
        idx = np.argsort(-x.a, axis=dim)[:, :k]
        return NpTensor(np.take_along_axis(x.a, idx, axis=dim), self.log), NpTensor(idx, self.log)


def softmax_p(logits, i):
    z = [math.exp(v) for v in logits]
    return z[i] / sum(z)


class TestZeroShotPredict:
    SPECIES = [[7, "Aramus guarauna", "Animalia", "Chordata", "Aves", "Gruiformes", "Aramidae", "Aramus"],
               [10, "Bubo virginianus", "Animalia", "Chordata", "Aves", "Strigiformes", "Strigidae", "Bubo"],
               [13, "Amanita muscaria", "Fungi", "Basidiomycota", "Agaricomycetes", "", "Amanitaceae", "Amanita"]]

    def _index(self, tmp_path, vecs=None, species=None):
        d = tmp_path / "index"
        d.mkdir(exist_ok=True)
        np.save(d / "species_text.npy", np.eye(3, dtype=np.float16) if vecs is None else vecs)
        (d / "species.json").write_text(json.dumps(self.SPECIES if species is None else species), encoding="utf-8")
        return str(d)

    def test_probability_is_the_softmax_of_scaled_cosines_and_the_winner_is_not_first_by_accident(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        t = NpTorch()
        zs = tz.ZeroShot(self._index(tmp_path), t, "cuda:1")
        (preds,) = zs.predict(np.array([[0.6, 0.0, 0.8]]), scale=10.0, top_k=2)
        logits = [6.0, 0.0, 8.0]
        assert [p[0][1] for p in preds] == ["Amanita muscaria", "Aramus guarauna"]
        assert preds[0][1] == pytest.approx(softmax_p(logits, 2), abs=2e-3)
        assert preds[1][1] == pytest.approx(softmax_p(logits, 0), abs=2e-3)
        assert preds[0][0] == self.SPECIES[2]                                            # the full species row comes back, not just the name
        assert t.log == ["cuda:1", "cuda:1"]                                             # the index and the image batch were both moved to the device

    def test_more_images_than_one_batch_keep_their_order(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        zs = tz.ZeroShot(self._index(tmp_path), NpTorch(), "cuda:0")
        images = np.array([[0.1, 0.0, 0.99], [0.0, 1.0, 0.0], [0.99, 0.1, 0.0]])
        out = zs.predict(images, scale=20.0, top_k=1, batch=2)
        assert [o[0][0][1] for o in out] == ["Amanita muscaria", "Bubo virginianus", "Aramus guarauna"]
        assert out[1][0][1] == pytest.approx(softmax_p([0.0, 20.0, 0.0], 1), abs=2e-3)

    def test_probabilities_over_all_species_sum_to_one(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        zs = tz.ZeroShot(self._index(tmp_path), NpTorch(), "cuda:0")
        (preds,) = zs.predict(np.array([[0.5, 0.5, 0.7]]), scale=5.0, top_k=3)
        assert sum(p for _, p in preds) == pytest.approx(1.0, abs=2e-3)

    def test_an_index_with_a_different_number_of_vectors_and_species_is_refused(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        with pytest.raises(ValueError, match=r"index: 2 text vectors for 3 species"):
            tz.ZeroShot(self._index(tmp_path, vecs=np.eye(3, dtype=np.float16)[:2]), NpTorch(), "cuda:0")
        assert tz.ZeroShot(self._index(tmp_path), NpTorch(), "cuda:0").species == self.SPECIES        # twin: matching counts load


class Scalar:
    def __init__(self, v):
        self.v = v

    def __float__(self):
        return self.v


class LogScale:
    """CLIP's learned log temperature: .detach().exp() is the scale the code must use (and float() must work on it)."""

    def __init__(self, log):
        self.log = log

    def detach(self):
        return self

    def exp(self):
        return Scalar(math.exp(self.log))


class TextFakeModel(FakeModel):
    """FakeModel plus the text side: each token row is the vector."""

    def __init__(self, scale=100.0):
        super().__init__()
        self.logit_scale = LogScale(math.log(scale))

    def encode_text(self, tokens):
        from test_bioclip_suggest import FakeOut
        self.events.append(("encode_text", len(tokens.items)))
        return FakeOut(np.array(tokens.items, dtype=np.float32))


class TextFakeOpenClip(FakeOpenClip):
    def __init__(self):
        super().__init__()
        self.model = TextFakeModel()
        self.tokenizer_ids = []

    def get_tokenizer(self, model_id):
        self.tokenizer_ids.append(model_id)
        from test_bioclip_suggest import FakeBatch
        return lambda texts: FakeBatch([[float(len(t)), 0.0] for t in texts])


class TestEncodeText:
    def _emb(self, stage):
        be = stage("bioclip_embed")
        oc = TextFakeOpenClip()
        return be.BioClipEmbedder(CFG, torch_mod=FakeTorch(free_bytes=(8 * 2**30,)), open_clip_mod=oc, load_image=lambda p: p, environ={}), oc

    def test_text_vectors_are_float16_unit_length_in_input_order_and_batched(self, stage):
        emb, oc = self._emb(stage)
        out = emb.encode_text(["abc", "de", "f"], batch=2)
        assert out.dtype == np.float16 and out.tolist() == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        assert [e for e in oc.model.events if e[0] == "encode_text"] == [("encode_text", 2), ("encode_text", 1)]

    def test_the_tokenizer_is_loaded_once_by_the_configured_model_id_and_only_when_text_is_needed(self, stage):
        emb, oc = self._emb(stage)
        assert oc.tokenizer_ids == []
        emb.encode_text(["a"]); emb.encode_text(["b"])
        assert oc.tokenizer_ids == ["hf-hub:fake/bioclip"]

    def test_logit_scale_is_the_exponentiated_learned_temperature(self, stage):
        emb, _ = self._emb(stage)
        assert emb.logit_scale() == pytest.approx(100.0, abs=1e-9)


# ---------------------------------------------------------------- pooling, calibration, iconic groups

def R(id_, name, kingdom="Animalia", phylum="Chordata", klass="Aves"):
    return [id_, name, kingdom, phylum, klass, "Order", "Family", "Genus"]


class TestPoolGroup:
    def test_pooled_probability_is_the_sum_over_the_photos_divided_by_the_photo_count(self, stage):
        tz = stage("taxon_zeroshot")
        a, b, c = R(1, "A"), R(2, "B"), R(3, "C")
        preds = [[(a, 0.6), (b, 0.3)], [(b, 0.5), (c, 0.4)], [(a, 0.2), (b, 0.1)]]        # A: 0.8/3, B: 0.9/3, C: 0.4/3
        out = tz.pool_group(preds, top=2)
        assert [(r[1], round(p, 4)) for r, p in out] == [("B", 0.3), ("A", 0.2667)]         # B wins although A is the top-1 of two photos
        assert [r[1] for r, _ in tz.pool_group(preds, top=1)] == ["B"]

    def test_equal_pooled_probabilities_are_ordered_by_species_id_so_the_result_is_deterministic(self, stage):
        tz = stage("taxon_zeroshot")
        out = tz.pool_group([[(R(9, "Z"), 0.5), (R(4, "Q"), 0.5)]])
        assert [r[1] for r, _ in out] == ["Q", "Z"]


class TestObservationBins:
    def test_accuracy_is_counted_per_observation_of_the_pooled_top_one_not_per_photo(self, stage):
        tz = stage("taxon_zeroshot")
        a, b = R(1, "Aramus guarauna"), R(2, "Bubo virginianus")
        per_photo = [[(a, 0.8)], [(a, 0.6)],         # observation 1: two photos, pooled 0.7 for A, true A -> hit in [0.7, 0.9)
                     [(b, 0.05)],                    # observation 2: pooled 0.05 for B, true A -> miss in [0, 0.1)
                     [(a, 0.05)]]                    # observation 3: pooled 0.05 for A, true A -> hit in [0, 0.1)
        bins = tz.observation_bins(per_photo, [1, 1, 2, 3], ["Aramus guarauna"] * 4)
        assert {(b["lo"], b["n"], b["accuracy"]) for b in bins} == {
            (0.0, 2, 0.5), (0.1, 0, None), (0.2, 0, None), (0.3, 0, None), (0.5, 0, None), (0.7, 1, 1.0), (0.9, 0, None)}

    def test_a_pooled_top_one_with_the_wrong_species_is_a_miss_even_when_one_photo_was_right(self, stage):
        tz = stage("taxon_zeroshot")
        a, b = R(1, "Aramus guarauna"), R(2, "Bubo virginianus")
        per_photo = [[(a, 0.9), (b, 0.05)], [(b, 0.5), (a, 0.1)]]                           # pooled: A (0.9 + 0.1) / 2 = 0.5, B (0.05 + 0.5) / 2 = 0.275
        bins = tz.observation_bins(per_photo, [1, 1], ["Bubo virginianus"] * 2)
        assert [(b["lo"], b["n"], b["accuracy"]) for b in bins if b["n"]] == [(0.5, 1, 0.0)]
        bins2 = tz.observation_bins(per_photo, [1, 1], ["Aramus guarauna"] * 2)             # twin: the same pooling scores a hit for the right label
        assert [(b["lo"], b["n"], b["accuracy"]) for b in bins2 if b["n"]] == [(0.5, 1, 1.0)]


class TestMeasured:
    BINS = [{"lo": 0.0, "hi": 0.5, "n": 4, "accuracy": 0.25}, {"lo": 0.5, "hi": 0.9, "n": 0, "accuracy": None}, {"lo": 0.9, "hi": 1.01, "n": 2, "accuracy": 1.0}]

    def test_returns_the_bin_accuracy_with_the_lower_edge_inclusive(self, stage):
        tz = stage("taxon_zeroshot")
        assert [tz.measured(p, self.BINS) for p in (0.0, 0.49, 0.9, 1.0)] == [0.25, 0.25, 1.0, 1.0]

    def test_an_empty_bin_and_a_probability_outside_every_bin_are_unmeasured_not_zero(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.measured(0.5, self.BINS) is None
        assert tz.measured(1.5, self.BINS) is None


class TestIconicOf:
    def test_the_seven_animal_classes_iNat_singles_out_are_their_own_iconic_group(self, stage):
        tz = stage("taxon_zeroshot")
        for klass in ("Aves", "Mammalia", "Reptilia", "Amphibia", "Actinopterygii", "Insecta", "Arachnida"):
            assert tz.iconic_of(R(1, "x", klass=klass)) == klass

    def test_mollusca_other_animals_and_the_non_animal_kingdoms(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.iconic_of(R(1, "x", phylum="Mollusca", klass="Gastropoda")) == "Mollusca"
        assert tz.iconic_of(R(1, "x", phylum="Arthropoda", klass="Malacostraca")) == "Animalia"
        assert [tz.iconic_of(R(1, "x", kingdom=k, klass="Insecta")) for k in ("Plantae", "Fungi", "Protozoa", "Chromista")] == [
            "Plantae", "Fungi", "Protozoa", "Chromista"]                                 # the class never matters outside Animalia


class TestCalibrate:
    def test_writes_observation_level_bins_for_species_rank_photos_only(self, stage, tmp_path):
        tz = stage("taxon_zeroshot")
        idx = TestZeroShotPredict()._index(tmp_path)
        (tmp_path / "index" / "meta.json").write_text(json.dumps({"source": "dwca.zip", "species": 3, "dim": 3}), encoding="utf-8")
        photos = [([1.0, 0.0, 0.0], {"rank": "species", "taxon": "Aramus guarauna", "obs_id": 1}),
                  ([1.0, 0.0, 0.0], {"rank": "species", "taxon": "Aramus guarauna", "obs_id": 1}),        # same observation: pooled with the one above
                  ([0.6, 0.0, 0.8], {"rank": "species", "taxon": "Bubo virginianus", "obs_id": 2}),        # predicted Amanita: a miss
                  ([0.0, 1.0, 0.0], {"rank": "genus", "taxon": "Bubo", "obs_id": 3})]                      # genus rank: not scored
        out = str(tmp_path / "zeroshot_bins.json")
        bins = tz.calibrate(idx, photos, 10.0, NpTorch(), "cuda:1", out)
        assert [(b["lo"], b["n"], b["accuracy"]) for b in bins if b["n"]] == [(0.7, 1, 0.0), (0.9, 1, 1.0)]
        written = json.load(open(out, encoding="utf-8"))
        assert (written["photos"], written["observations"], written["scale"], written["index"]["source"]) == (3, 2, 10.0, "dwca.zip")
        assert written["bins"] == bins


# ---------------------------------------------------------------- inat_suggest with the zero-shot index

ZS_BINS = [{"lo": 0.0, "hi": 0.5, "n": 10, "accuracy": 0.3}, {"lo": 0.5, "hi": 1.01, "n": 10, "accuracy": 0.9}]
SP = TestZeroShotPredict.SPECIES


class TestSuggestGroupsZeroShot:
    def _by_group(self):
        return {7: [[{"taxon": "Apis mellifera", "rank": "species", "iconic": "Insecta"}] * 2]}

    def test_each_group_gets_the_pooled_top_species_with_iconic_family_and_measured_accuracy(self, stage):
        su = stage("inat_suggest")
        zs = {7: [[(SP[0], 0.8), (SP[2], 0.1)], [(SP[0], 0.4), (SP[1], 0.3)]]}              # Aramus (0.8 + 0.4) / 2 = 0.6, Bubo 0.15, Amanita 0.05
        out = su.suggest_groups(self._by_group(), {}, [(0.0, 1.01, 0.5)], zs, ZS_BINS)
        assert out["7"]["zero_shot"] == [
            {"taxon": "Aramus guarauna", "taxon_id": 7, "iconic": "Aves", "family": "Aramidae", "prob": 0.6, "measured_accuracy": 0.9},
            {"taxon": "Bubo virginianus", "taxon_id": 10, "iconic": "Aves", "family": "Strigidae", "prob": 0.15, "measured_accuracy": 0.3},
            {"taxon": "Amanita muscaria", "taxon_id": 13, "iconic": "Fungi", "family": "Amanitaceae", "prob": 0.05, "measured_accuracy": 0.3}]

    def test_without_zero_shot_there_is_no_key_and_a_group_with_no_predictions_gets_an_empty_list(self, stage):
        su = stage("inat_suggest")
        assert "zero_shot" not in su.suggest_groups(self._by_group(), {}, [(0.0, 1.01, 0.5)])["7"]
        assert su.suggest_groups(self._by_group(), {}, [(0.0, 1.01, 0.5)], {}, ZS_BINS)["7"]["zero_shot"] == []


class FakeIndexedEmbedder:
    device = "cuda:1"

    def logit_scale(self):
        return 10.0


class TestLoadZeroShot:
    def _cfg(self, tmp_path):
        return {"zeroshot_index": TestZeroShotPredict()._index(tmp_path)}

    def test_no_index_in_the_config_means_no_zero_shot(self, stage, tmp_path):
        su = stage("inat_suggest")
        assert su.load_zeroshot(FakeIndexedEmbedder(), {}, torch_mod=NpTorch()) is None

    def test_the_old_backend_never_loads_it_because_its_vectors_are_not_bioclip_vectors(self, stage, tmp_path, monkeypatch):
        su = stage("inat_suggest")
        (tmp_path / "data" / "zeroshot_bins.json").write_text(json.dumps({"bins": ZS_BINS}), encoding="utf-8")
        monkeypatch.setattr(su, "BACKEND", "siglip")
        assert su.load_zeroshot(FakeIndexedEmbedder(), self._cfg(tmp_path), torch_mod=NpTorch()) is None

    def test_a_configured_index_without_its_calibration_is_refused(self, stage, tmp_path):
        su = stage("inat_suggest")
        with pytest.raises(SystemExit, match=r"zeroshot_bins\.json does not exist; run taxon_zeroshot\.py calibrate first"):
            su.load_zeroshot(FakeIndexedEmbedder(), self._cfg(tmp_path), torch_mod=NpTorch())

    def test_with_the_calibration_it_returns_the_index_on_the_embedders_device_its_scale_and_the_bins(self, stage, tmp_path):
        su = stage("inat_suggest")
        (tmp_path / "data" / "zeroshot_bins.json").write_text(json.dumps({"bins": ZS_BINS}), encoding="utf-8")
        t = NpTorch()
        zs, scale, bins = su.load_zeroshot(FakeIndexedEmbedder(), self._cfg(tmp_path), torch_mod=t)
        assert (zs.species, scale, bins, t.log) == (SP, 10.0, ZS_BINS, ["cuda:1"])


class FakeZeroShot:
    """Records the image vectors and scale predict() is asked about; every photo's top species is Aramus (0.6) then Bubo (0.2)."""

    def __init__(self):
        self.calls = []

    def predict(self, vecs, scale):
        self.calls.append((np.asarray(vecs).tolist(), scale))
        return [[(SP[0], 0.6), (SP[1], 0.2)] for _ in vecs]


class TestMainWithZeroShot:
    def _run(self, tmp_path, stage, monkeypatch, zsl):
        from test_bioclip_suggest import TestInatSuggestMain
        su, emb, out, calls = TestInatSuggestMain()._setup(tmp_path, stage, monkeypatch, with_copy=True, reference=[
            ((1.0, 0.0), {"source": "inat_photo", "key": "p1", "obs_id": 1, "taxon": "Amanita muscaria", "rank": "species", "iconic": "Fungi"})])
        monkeypatch.setattr(su, "load_zeroshot", lambda e, cfg, torch_mod=None: zsl)
        if zsl:
            monkeypatch.setitem(stage("embed_index").CFG["bioclip"], "zeroshot_index", "taxonomy-index-dir")
        su.main()
        return json.loads(out.read_text(encoding="utf-8")), json.loads((tmp_path / "data" / "inat_suggestions.meta.json").read_text(encoding="utf-8"))

    def test_the_photo_vectors_go_to_the_zero_shot_index_and_its_candidates_and_index_are_written(self, stage, tmp_path, monkeypatch):
        zs = FakeZeroShot()
        out, meta = self._run(tmp_path, stage, monkeypatch, (zs, 100.0, ZS_BINS))
        assert zs.calls == [([[1.0, 0.0]], 100.0)]                                   # the one photo with a copy, at the embedder's own scale
        assert out["7"]["zero_shot"] == [
            {"taxon": "Aramus guarauna", "taxon_id": 7, "iconic": "Aves", "family": "Aramidae", "prob": 0.6, "measured_accuracy": 0.9},
            {"taxon": "Bubo virginianus", "taxon_id": 10, "iconic": "Aves", "family": "Strigidae", "prob": 0.2, "measured_accuracy": 0.3}]
        assert (meta["zero_shot_index"], meta["groups"]) == ("taxonomy-index-dir", 1)

    def test_the_same_run_without_a_zero_shot_index_writes_no_zero_shot_key_and_no_index_in_meta(self, stage, tmp_path, monkeypatch):
        out, meta = self._run(tmp_path, stage, monkeypatch, None)
        assert "zero_shot" not in out["7"] and meta["zero_shot_index"] is None


# ---------------------------------------------------------------- drift: the earlier calibration against observations posted since

def _bins(acc, n):
    return [{"lo": lo, "hi": hi, "n": n if i == 6 else 0, "accuracy": acc if i == 6 else None}
            for i, (lo, hi) in enumerate(((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)))]


class TestDriftWarnings:
    def test_a_bin_that_is_worse_than_calibrated_by_more_than_the_tolerance_is_named_with_both_numbers(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.drift_warnings(_bins(0.95, 40), _bins(0.60, 12)) == [
            "pooled p 0.9-1.0: calibrated 0.95, but 0.60 on 12 newly posted observations"]

    def test_exactly_at_the_tolerance_is_not_a_warning_and_just_past_it_is(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.drift_warnings(_bins(0.75, 40), _bins(0.50, 12), tolerance=0.25) == []              # a drop of exactly 0.25 (exact in binary)
        assert len(tz.drift_warnings(_bins(0.75, 40), _bins(0.4375, 12), tolerance=0.25)) == 1        # 0.3125 is past it
        assert tz.drift_warnings(_bins(0.90, 40), _bins(0.80, 12)) == []                              # and the default tolerance is 0.10

    def test_too_few_observations_on_either_side_or_an_unmeasured_bin_never_warn(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.drift_warnings(_bins(0.95, 40), _bins(0.0, 9)) == []                  # only 9 recent observations
        assert tz.drift_warnings(_bins(0.95, 9), _bins(0.0, 40)) == []                  # only 9 in the old calibration
        assert tz.drift_warnings(_bins(None, 0), _bins(0.0, 40)) == []                  # the old bin was empty

    def test_a_better_than_calibrated_bin_is_not_a_warning(self, stage):
        tz = stage("taxon_zeroshot")
        assert tz.drift_warnings(_bins(0.70, 40), _bins(1.0, 12)) == []


class TestCalibrateDrift:
    def _run(self, tmp_path, stage, new_true, prev=None):
        tz = stage("taxon_zeroshot")
        idx = TestZeroShotPredict()._index(tmp_path)
        (tmp_path / "index" / "meta.json").write_text(json.dumps({"source": "dwca.zip", "species": 3, "dim": 3}), encoding="utf-8")
        photos = [([1.0, 0.0, 0.0], {"rank": "species", "taxon": "Aramus guarauna", "obs_id": 1})]            # an old observation: right
        photos += [([0.0, 0.0, 1.0], {"rank": "species", "taxon": new_true, "obs_id": 100 + i}) for i in range(10)]   # ten new ones, all predicted Amanita
        prev_path = None
        if prev is not None:
            prev_path = str(tmp_path / "prev.json")
            json.dump(prev, open(prev_path, "w", encoding="utf-8"))
        out = str(tmp_path / "zeroshot_bins.json")
        tz.calibrate(idx, photos, 10.0, NpTorch(), "cuda:1", out, prev_path=prev_path)
        return json.load(open(out, encoding="utf-8"))

    PREV = {"max_obs_id": 50, "bins": _bins(0.95, 40)}

    def test_new_observations_that_are_all_wrong_produce_the_drift_warning_and_are_scored_on_their_own(self, stage, tmp_path):
        r = self._run(tmp_path, stage, "Bubo virginianus", self.PREV)
        assert r["max_obs_id"] == 109
        assert r["recent"]["since_obs_id"] == 50 and (r["recent"]["photos"], r["recent"]["observations"]) == (10, 10)
        assert [(b["lo"], b["n"], b["accuracy"]) for b in r["recent"]["bins"] if b["n"]] == [(0.9, 10, 0.0)]
        assert r["recent"]["warnings"] == ["pooled p 0.9-1.0: calibrated 0.95, but 0.00 on 10 newly posted observations"]

    def test_the_same_run_with_new_observations_that_are_all_right_has_no_warning(self, stage, tmp_path):
        r = self._run(tmp_path, stage, "Amanita muscaria", self.PREV)
        assert [(b["lo"], b["n"], b["accuracy"]) for b in r["recent"]["bins"] if b["n"]] == [(0.9, 10, 1.0)]
        assert r["recent"]["warnings"] == []

    def test_without_an_earlier_calibration_there_is_no_recent_block_and_the_high_water_mark_is_still_recorded(self, stage, tmp_path):
        r = self._run(tmp_path, stage, "Bubo virginianus", None)
        assert "recent" not in r and r["max_obs_id"] == 109
        assert r["observations"] == 11

    def test_observations_not_newer_than_the_earlier_calibration_are_not_counted_as_recent(self, stage, tmp_path):
        r = self._run(tmp_path, stage, "Bubo virginianus", {"max_obs_id": 105, "bins": _bins(0.95, 40)})
        assert (r["recent"]["observations"], r["recent"]["warnings"]) == (4, [])        # obs 106..109 only: below the 10-observation minimum
