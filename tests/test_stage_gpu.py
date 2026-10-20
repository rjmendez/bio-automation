"""pipeline/stages GPU handling (EXT-2 embed_index providers, EXT-4 Ollama main_gpu pin). No GPU, no network, no real model."""
from __future__ import annotations

import collections
import importlib
import json
import logging
import sys
import urllib.error
from pathlib import Path

import pytest
from fake_ort import FakeOrt

STAGES = Path(__file__).resolve().parents[1] / "pipeline" / "stages"
CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
_FLAT = ("embed_index", "ort_providers", "vision_files", "visionlib", "hands_check", "redescribe_vision", "screen_inat_copies",
         "triage_vision")


@pytest.fixture
def stage(monkeypatch):
    """Import flat stage modules the way the scripts do (their own directory on sys.path), without leaking them into other tests."""
    monkeypatch.syspath_prepend(str(STAGES))
    for name in _FLAT:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delenv("BIO_ORT_CUDA_DEVICE", raising=False)
    monkeypatch.delenv("BIO_OLLAMA_MAIN_GPU", raising=False)
    yield importlib.import_module
    for name in _FLAT:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------- EXT-2: embed_index

def _embed_cfg(tmp_path, monkeypatch, **extra):
    md = tmp_path / "model"
    md.mkdir()
    (md / "preprocessor_config.json").write_text(json.dumps({"size": 224}), encoding="utf-8")
    cfg = {"model_dir": str(md), "onnx_file": "vision_model.onnx", "output_name": "pooler_output", "threads": 4, **extra}
    path = tmp_path / "embed_config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("EMBED_CONFIG", str(path))
    return md


class TestEmbedderProviders:
    def test_default_config_asks_for_cuda_then_cpu_on_device_zero_with_four_threads(self, stage, tmp_path, monkeypatch):
        md = _embed_cfg(tmp_path, monkeypatch)
        ei = stage("embed_index")
        ort = FakeOrt()
        monkeypatch.setattr(ei, "ort", ort)
        emb = ei.Embedder()
        (path, opts, providers), = ort.session_calls
        assert path == str(md / "vision_model.onnx")
        assert opts.intra_op_num_threads == 4
        assert providers == [(CUDA, {"device_id": 0}), CPU]
        assert (emb.inp, emb.out) == ("pixel_values", "pooler_output")

    def test_config_providers_and_device_are_used_and_env_overrides_the_device(self, stage, tmp_path, monkeypatch):
        _embed_cfg(tmp_path, monkeypatch, providers=[CUDA, CPU], cuda_device_id=1)
        ei = stage("embed_index")
        ort = FakeOrt()
        monkeypatch.setattr(ei, "ort", ort)
        ei.Embedder()
        assert ort.session_calls[0][2] == [(CUDA, {"device_id": 1}), CPU]
        monkeypatch.setenv("BIO_ORT_CUDA_DEVICE", "0")
        ei.Embedder()
        assert ort.session_calls[1][2] == [(CUDA, {"device_id": 0}), CPU]

    def test_config_can_pin_cpu_without_any_cuda_warning(self, stage, tmp_path, monkeypatch, caplog):
        _embed_cfg(tmp_path, monkeypatch, providers=[CPU])
        ei = stage("embed_index")
        ort = FakeOrt(got=[CPU])
        monkeypatch.setattr(ei, "ort", ort)
        with caplog.at_level(logging.WARNING):
            ei.Embedder()
        assert ort.session_calls[0][2] == [CPU]
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_cpu_only_onnxruntime_warns_and_counts_when_cuda_is_not_required(self, stage, tmp_path, monkeypatch, caplog):
        _embed_cfg(tmp_path, monkeypatch)
        ei = stage("embed_index")
        monkeypatch.setattr(ei, "ort", FakeOrt(available=[CPU], got=[CPU]))
        with caplog.at_level(logging.WARNING, logger="bio_automation.ort_providers"):
            ei.Embedder()
        assert sys.modules["ort_providers"].degraded_counts() == {
            "embed_index:CUDAExecutionProvider-unavailable": 1, "embed_index:cuda-not-obtained": 1}
        assert any("embed_index: CUDA was requested (device_id=0) but the session runs on ['CPUExecutionProvider']" in r.getMessage()
                   for r in caplog.records)

    def test_require_cuda_refuses_to_run_on_cpu(self, stage, tmp_path, monkeypatch):
        _embed_cfg(tmp_path, monkeypatch, require_cuda=True)
        ei = stage("embed_index")
        monkeypatch.setattr(ei, "ort", FakeOrt(got=[CPU]))
        with pytest.raises(RuntimeError, match=r"embed_index: CUDA was requested \(device_id=0\)"):
            ei.Embedder()

    def test_require_cuda_runs_when_cuda_is_obtained(self, stage, tmp_path, monkeypatch):
        _embed_cfg(tmp_path, monkeypatch, require_cuda=True)
        ei = stage("embed_index")
        monkeypatch.setattr(ei, "ort", FakeOrt())
        assert ei.Embedder().sess.get_providers() == [CUDA, CPU]


# ---------------------------------------------------------------- EXT-4: Ollama pin

class FakeHttp:
    """Records every POST body (deep-copied) and answers from a script of results / exceptions, then repeats the last one."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, path, body=None, timeout=300):
        self.calls.append((path, json.loads(json.dumps(body))))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def _reply(obj):
    return {"message": {"content": json.dumps(obj)}}


@pytest.fixture
def vf(stage, monkeypatch):
    mod = stage("vision_files")
    slept = []
    monkeypatch.setattr(mod, "SLEEP", slept.append)
    monkeypatch.setattr(mod, "RETRY_COUNT", 0)
    mod.slept = slept
    return mod


class TestGpuOptions:
    def test_default_is_no_pin(self, vf):
        assert vf.gpu_options({}) == {}
        assert vf.gpu_options({"BIO_OLLAMA_MAIN_GPU": ""}) == {}

    def test_env_pin(self, vf):
        assert vf.gpu_options({"BIO_OLLAMA_MAIN_GPU": "0"}) == {"main_gpu": 0}

    def test_process_env_is_read_by_default(self, vf, monkeypatch):
        monkeypatch.setenv("BIO_OLLAMA_MAIN_GPU", "3")
        assert vf.gpu_options() == {"main_gpu": 3}

    def test_garbage_raises(self, vf):
        with pytest.raises(ValueError, match="BIO_OLLAMA_MAIN_GPU must be an integer Ollama device index, got 'two'"):
            vf.gpu_options({"BIO_OLLAMA_MAIN_GPU": "two"})

    def test_negative_raises(self, vf):
        with pytest.raises(ValueError, match="BIO_OLLAMA_MAIN_GPU must be >= 0, got -1"):
            vf.gpu_options({"BIO_OLLAMA_MAIN_GPU": "-1"})


OPTS = {"temperature": 0, "main_gpu": 1}


def _body():
    return {"model": "m", "stream": False, "think": False, "options": dict(OPTS), "messages": []}


class TestChat:
    def test_success_makes_one_call_and_never_sleeps(self, vf, monkeypatch):
        http = FakeHttp([_reply({"a": 1})])
        monkeypatch.setattr(vf, "http", http)
        assert vf.chat(_body()) == _reply({"a": 1})
        assert [(p, b["options"]) for p, b in http.calls] == [("/api/chat", OPTS)]
        assert vf.slept == []
        assert vf.RETRY_COUNT == 0

    def test_transient_errors_are_retried_with_the_same_options_and_reported(self, vf, monkeypatch, caplog):
        http = FakeHttp([TimeoutError("cold load"), urllib.error.URLError("reset"), _reply({"a": 1})])
        monkeypatch.setattr(vf, "http", http)
        with caplog.at_level(logging.WARNING, logger="vision_files"):
            assert vf.chat(_body()) == _reply({"a": 1})
        assert [b["options"] for _, b in http.calls] == [OPTS, OPTS, OPTS]
        assert vf.slept == [2.0, 4.0]
        assert vf.RETRY_COUNT == 2
        assert [r.getMessage() for r in caplog.records] == [
            ("ollama /api/chat failed (TimeoutError: cold load); retry 1 of 3 with options "
             "{'temperature': 0, 'main_gpu': 1} (total retries 1)"),
            ("ollama /api/chat failed (URLError: <urlopen error reset>); retry 2 of 3 with options "
             "{'temperature': 0, 'main_gpu': 1} (total retries 2)")]

    def test_errors_beyond_the_budget_raise_and_still_carry_the_pin(self, vf, monkeypatch):
        http = FakeHttp([TimeoutError("down")])
        monkeypatch.setattr(vf, "http", http)
        with pytest.raises(TimeoutError, match="down"):
            vf.chat(_body())
        assert [b["options"] for _, b in http.calls] == [OPTS] * 4
        assert vf.slept == [2.0, 4.0, 8.0]

    def test_a_model_rejecting_think_is_retried_once_without_it_and_keeps_the_pin(self, vf, monkeypatch):
        err = urllib.error.HTTPError("u", 400, "does not support thinking", {}, None)
        http = FakeHttp([err, _reply({"a": 1})])
        monkeypatch.setattr(vf, "http", http)
        assert vf.chat(_body()) == _reply({"a": 1})
        assert ["think" in b for _, b in http.calls] == [True, False]
        assert [b["options"] for _, b in http.calls] == [OPTS, OPTS]
        assert vf.slept == []

    def test_ask_sends_the_pin_it_was_given_on_every_attempt(self, vf, monkeypatch):
        http = FakeHttp([TimeoutError("x"), _reply({"release": "ok"})])
        monkeypatch.setattr(vf, "http", http)
        assert vf.ask("m", "IMG", {"main_gpu": 0}) == {"release": "ok"}
        assert [b["options"] for _, b in http.calls] == [{"temperature": 0, "main_gpu": 0}] * 2


def _run_main(mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(mod.vf, "pick_model", lambda: "some-model")
    monkeypatch.setattr(mod.vf, "small", lambda p: "IMG:" + str(p))
    mod.main()


@pytest.fixture(params=[(None, None), ("0", 0), ("3", 3)], ids=["default", "env0", "env3"])
def pin(request, monkeypatch):
    """The options every request must carry: no pin by default, or BIO_OLLAMA_MAIN_GPU."""
    env, expected = request.param
    if env is not None:
        monkeypatch.setenv("BIO_OLLAMA_MAIN_GPU", env)
    return {"temperature": 0} if expected is None else {"temperature": 0, "main_gpu": expected}


class TestScriptsKeepThePinAfterAnError:
    """The old code set gpu = {} after an error in the first items, silently moving the rest of the batch off the card."""

    def test_hands_check(self, stage, vf, tmp_path, monkeypatch, pin):
        hc = stage("hands_check")
        out = tmp_path / "hands.json"
        monkeypatch.setattr(hc, "OUT", str(out))
        monkeypatch.setattr(hc, "candidates", lambda: [("a.jpg", "pa"), ("b.jpg", "pb"), ("c.jpg", "pc")])
        answer = _reply({"hands_or_fingers": True, "face_or_head": False, "other_body": False, "clothing_or_accessory": False,
                         "tattoo_or_mark": False, "shadow_of_person": False, "other_person_note": ""})
        http = FakeHttp([TimeoutError("cold") ] * 4 + [answer])      # item a exhausts its 4 attempts, b and c succeed
        monkeypatch.setattr(vf, "http", http)
        _run_main(hc, monkeypatch, ["hands_check.py", "run"])
        assert [b["options"] for _, b in http.calls] == [pin] * 6
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["a.jpg"] == {"error": "TimeoutError: cold"}
        assert [saved[k]["only_hands"] for k in ("b.jpg", "c.jpg")] == [True, True]

    def test_redescribe_vision(self, stage, vf, tmp_path, monkeypatch, pin):
        rv = stage("redescribe_vision")
        out = tmp_path / "redescribed.json"
        monkeypatch.setattr(rv, "OUT", str(out))
        tax = {"name": "Auricularia", "iconic": "Fungi"}
        monkeypatch.setattr(rv, "build", lambda: ([("a.jpg", "pa", tax), ("b.jpg", "pb", tax)], collections.Counter(), 0, 2))
        http = FakeHttp([TimeoutError("cold")] * 4 + [_reply({"id_plausible": "yes"})])
        monkeypatch.setattr(vf, "http", http)
        _run_main(rv, monkeypatch, ["redescribe_vision.py", "run"])
        assert [b["options"] for _, b in http.calls] == [pin] * 5
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["a.jpg"] == {"error": "TimeoutError: cold"}
        assert saved["b.jpg"] == {"tags": {"id_plausible": "yes"}, "taxon": "Auricularia", "iconic": "Fungi", "model": "some-model"}

    def test_screen_inat_copies(self, stage, vf, tmp_path, monkeypatch, pin):
        sic = stage("screen_inat_copies")
        copies = []
        for n in ("a", "b"):
            p = tmp_path / f"{n}.jpg"
            p.write_bytes(b"x")
            copies.append(str(p))
        (tmp_path / "inat_queue.json").write_text(json.dumps([{"copy": c} for c in copies]), encoding="utf-8")
        monkeypatch.setattr(sic, "ROOT", str(tmp_path))
        monkeypatch.setattr(sic, "TAG", "")
        monkeypatch.setattr(sic, "OUT", str(tmp_path / "vision_inat.json"))
        http = FakeHttp([TimeoutError("cold")] * 4 + [_reply({"person_visible": False})])
        monkeypatch.setattr(vf, "http", http)
        _run_main(sic, monkeypatch, ["screen_inat_copies.py"])
        assert [b["options"] for _, b in http.calls] == [pin] * 5
        saved = json.loads((tmp_path / "vision_inat.json").read_text(encoding="utf-8"))
        assert saved[copies[0]] == {"error": "TimeoutError: cold"}
        assert saved[copies[1]] == {"tags": {"person_visible": False}, "model": "some-model"}

    def test_vision_files_main(self, vf, tmp_path, monkeypatch, pin):
        d = tmp_path / "ready"
        d.mkdir()
        for n in ("a.jpg", "b.jpg"):
            (d / n).write_bytes(b"x")
        monkeypatch.setattr(vf, "DIRS", [str(d)])
        monkeypatch.setattr(vf, "OUT", str(tmp_path / "vision_files.json"))
        monkeypatch.setattr(vf, "pick_model", lambda: "some-model")
        monkeypatch.setattr(vf, "small", lambda p: "IMG")
        http = FakeHttp([TimeoutError("cold")] * 4 + [_reply({"person_visible": False})])
        monkeypatch.setattr(vf, "http", http)
        vf.main()
        assert [b["options"] for _, b in http.calls] == [pin] * 5
        saved = json.loads((tmp_path / "vision_files.json").read_text(encoding="utf-8"))
        assert saved["a.jpg"] == {"error": "TimeoutError: cold"}
        assert saved["b.jpg"] == {"tags": {"person_visible": False}, "model": "some-model"}

    def test_triage_vision(self, stage, vf, tmp_path, monkeypatch, pin):
        _embed_cfg(tmp_path, monkeypatch)
        tv = stage("triage_vision")
        thumbs = tmp_path / "thumbs"
        thumbs.mkdir()
        for n in (1, 2):
            (thumbs / f"{n}.jpg").write_bytes(b"x")
        (tmp_path / "triage_index.json").write_text(json.dumps([1, 2]), encoding="utf-8")
        monkeypatch.setattr(tv, "ROOT", str(tmp_path))
        monkeypatch.setattr(tv, "THUMBS", str(thumbs))
        monkeypatch.setattr(tv, "OUT", str(tmp_path / "triage_vision.json"))
        monkeypatch.setattr(tv, "VEC", str(tmp_path / "triage_vectors.npz"))

        class FakeEmbedder:
            def embed(self, paths):
                return [[0.5, 0.5]]

        monkeypatch.setattr(tv, "Embedder", FakeEmbedder)
        http = FakeHttp([TimeoutError("cold")] * 4 + [_reply({"person_visible": False})])
        monkeypatch.setattr(vf, "http", http)
        monkeypatch.setattr(vf, "pick_model", lambda: "some-model")
        monkeypatch.setattr(vf, "small", lambda p: "IMG")
        tv.main()
        # frame 1 fails all 4 attempts, frame 2 succeeds, then the loop retries the errored frame 1 and it succeeds: 6 requests
        assert [b["options"] for _, b in http.calls] == [pin] * 6
        saved = json.loads((tmp_path / "triage_vision.json").read_text(encoding="utf-8"))
        assert saved == {"1": {"tags": {"person_visible": False}}, "2": {"tags": {"person_visible": False}}}
