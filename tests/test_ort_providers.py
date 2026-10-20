"""bio_automation.ort_providers and daclip.load_session: the GPU is requested, checked and a miss is reported (EXT-1)."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pytest
from fake_ort import FakeOrt

from bio_automation import daclip, ort_providers

CUDA, CPU = "CUDAExecutionProvider", "CPUExecutionProvider"
LOGGER = "bio_automation.ort_providers"


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(ort_providers, "_COUNTS", Counter())
    monkeypatch.setattr(ort_providers, "_LAST_WARN", {})
    monkeypatch.delenv(ort_providers.DEVICE_ENV, raising=False)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == LOGGER]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestOpenSession:
    def test_cuda_obtained_requests_cuda_then_cpu_after_preload_and_is_silent(self, caplog):
        ort = FakeOrt()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            sess = ort_providers.open_session(ort, Path("m.onnx"), sess_options="OPTS", label="daclip")
        assert sess.get_providers() == [CUDA, CPU]
        assert ort.events == ["preload", "session"]
        assert ort.session_calls == [("m.onnx", "OPTS", [(CUDA, {"device_id": 0}), CPU])]
        assert _warnings(caplog) == []
        assert ort_providers.degraded_counts() == {}

    def test_cuda_missed_logs_the_real_provider_and_counts_but_still_returns_the_session(self, caplog, monkeypatch):
        monkeypatch.setenv(ort_providers.DEVICE_ENV, "1")
        ort = FakeOrt(got=[CPU])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = ort_providers.open_session(ort, "m.onnx", label="daclip")
        assert sess.get_providers() == [CPU]
        assert _warnings(caplog) == [(
            "daclip: CUDA was requested (device_id=1) but the session runs on ['CPUExecutionProvider']; expect ~10-50x slower "
            "inference (cuDNN/CUDA DLLs missing or CPU-only onnxruntime) (occurrence 1 of this kind in this process)")]
        assert ort_providers.degraded_counts() == {"daclip:cuda-not-obtained": 1}

    def test_require_cuda_raises_when_cpu_was_obtained(self):
        with pytest.raises(RuntimeError, match=r"CUDA was requested \(device_id=0\) but the session runs on \['CPUExecutionProvider'\]"):
            ort_providers.open_session(FakeOrt(got=[CPU]), "m.onnx", require_cuda=True)

    def test_require_cuda_returns_the_session_when_cuda_was_obtained(self):
        sess = ort_providers.open_session(FakeOrt(), "m.onnx", require_cuda=True)
        assert sess.get_providers() == [CUDA, CPU]

    def test_cpu_only_onnxruntime_skips_cuda_and_preload_and_warns_twice(self, caplog):
        ort = FakeOrt(available=[CPU], got=[CPU])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ort_providers.open_session(ort, "m.onnx", label="emb")
        assert ort.events == ["session"]
        assert ort.session_calls[0][2] == [CPU]
        assert ort_providers.degraded_counts() == {"emb:CUDAExecutionProvider-unavailable": 1, "emb:cuda-not-obtained": 1}
        assert _warnings(caplog)[0].startswith(
            "emb: requested CUDAExecutionProvider but this onnxruntime offers only ['CPUExecutionProvider']; install onnxruntime-gpu")

    def test_cpu_only_onnxruntime_with_require_cuda_raises(self):
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            ort_providers.open_session(FakeOrt(available=[CPU], got=[CPU]), "m.onnx", require_cuda=True)

    def test_explicit_cpu_request_is_not_a_degradation(self, caplog):
        ort = FakeOrt(got=[CPU])
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            ort_providers.open_session(ort, "m.onnx", requested=[CPU])
        assert ort.events == ["session"]
        assert ort.session_calls[0][2] == [CPU]
        assert _warnings(caplog) == []
        assert ort_providers.degraded_counts() == {}

    def test_preload_failure_is_reported_and_the_session_is_still_attempted(self, caplog):
        ort = FakeOrt(preload_error=OSError("boom"))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = ort_providers.open_session(ort, "m.onnx", label="d")
        assert sess.get_providers() == [CUDA, CPU]
        assert ort.events == ["preload", "session"]
        assert ort_providers.degraded_counts() == {"d:preload-dlls-failed": 1}
        assert _warnings(caplog) == ["d: ort.preload_dlls() raised OSError: boom (occurrence 1 of this kind in this process)"]

    def test_onnxruntime_without_preload_dlls_still_opens(self):
        ort = FakeOrt(with_preload=False)
        ort_providers.open_session(ort, "m.onnx")
        assert ort.events == ["session"]

    def test_device_id_argument_beats_env(self, monkeypatch):
        monkeypatch.setenv(ort_providers.DEVICE_ENV, "1")
        ort = FakeOrt()
        ort_providers.open_session(ort, "m.onnx", device_id=3)
        assert ort.session_calls[0][2][0] == (CUDA, {"device_id": 3})


class TestRateLimit:
    def test_one_warning_per_interval_but_every_occurrence_counted(self, caplog):
        clock = Clock()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            for dt in (0.0, 59.9, 60.0):
                clock.t = 1000.0 + dt
                ort_providers.open_session(FakeOrt(got=[CPU]), "m.onnx", label="d", clock=clock)
        msgs = _warnings(caplog)
        assert [m.rsplit("(", 1)[1] for m in msgs] == ["occurrence 1 of this kind in this process)",
                                                        "occurrence 3 of this kind in this process)"]
        assert ort_providers.degraded_counts() == {"d:cuda-not-obtained": 3}


class TestRetry:
    def test_transient_creation_errors_are_retried_with_exponential_backoff(self, caplog):
        ort = FakeOrt(create_errors=[RuntimeError("cuda busy"), RuntimeError("cuda busy")])
        slept = []
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            sess = ort_providers.open_session(ort, "m.onnx", label="d", sleep=slept.append)
        assert sess.get_providers() == [CUDA, CPU]
        assert slept == [0.5, 1.0]
        assert len(ort.session_calls) == 3
        assert ort_providers.degraded_counts() == {"d:session-create-retry": 2}
        assert _warnings(caplog)[0].startswith("d: InferenceSession creation failed (RuntimeError: cuda busy); retry 1 of 2")

    def test_errors_beyond_the_retry_budget_propagate(self):
        ort = FakeOrt(create_errors=[RuntimeError("cuda oom")] * 3)
        slept = []
        with pytest.raises(RuntimeError, match="cuda oom"):
            ort_providers.open_session(ort, "m.onnx", sleep=slept.append)
        assert len(ort.session_calls) == 3
        assert slept == [0.5, 1.0]


class TestConfiguredDeviceId:
    def test_default_is_zero(self):
        assert ort_providers.configured_device_id(environ={}) == 0

    def test_config_value_used_when_env_absent(self):
        assert ort_providers.configured_device_id(2, environ={}) == 2

    def test_env_beats_config(self):
        assert ort_providers.configured_device_id(2, environ={ort_providers.DEVICE_ENV: "1"}) == 1

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="BIO_ORT_CUDA_DEVICE must be an integer CUDA device index, got 'gpu1'"):
            ort_providers.configured_device_id(environ={ort_providers.DEVICE_ENV: "gpu1"})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="config cuda_device_id must be >= 0, got -1"):
            ort_providers.configured_device_id(-1, environ={})


class TestDaclipLoadSession:
    def _model(self, tmp_path):
        d = tmp_path / "daclip-v2-fp32-224x224-1x-ox"
        d.mkdir()
        (d / "daclip-v2-fp32-224x224-1x.onnx").write_bytes(b"x")
        return d / "daclip-v2-fp32-224x224-1x.onnx"

    def test_requests_cuda_then_cpu_with_all_optimisations_and_the_env_device(self, tmp_path, monkeypatch):
        model = self._model(tmp_path)
        ort = FakeOrt()
        monkeypatch.setattr(daclip, "ort", ort)
        monkeypatch.setenv(ort_providers.DEVICE_ENV, "1")
        sess = daclip.load_session(tmp_path)
        assert sess.get_providers() == [CUDA, CPU]
        (path, opts, providers), = ort.session_calls
        assert path == str(model)
        assert opts.graph_optimization_level == "ENABLE_ALL"
        assert providers == [(CUDA, {"device_id": 1}), CPU]

    def test_cpu_result_is_reported_not_silent(self, tmp_path, monkeypatch, caplog):
        self._model(tmp_path)
        monkeypatch.setattr(daclip, "ort", FakeOrt(got=[CPU]))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            daclip.load_session(tmp_path)
        assert ort_providers.degraded_counts() == {"daclip:cuda-not-obtained": 1}
        assert "daclip: CUDA was requested (device_id=0) but the session runs on ['CPUExecutionProvider']" in _warnings(caplog)[0]

    def test_missing_model_still_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="DA-CLIP model not found"):
            daclip.load_session(tmp_path)


def test_stage_copy_is_identical_to_the_package_module():
    root = Path(__file__).resolve().parents[1]
    assert (root / "pipeline" / "stages" / "ort_providers.py").read_bytes() == \
        (root / "src" / "bio_automation" / "ort_providers.py").read_bytes()
