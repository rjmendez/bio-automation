"""A recording fake of the onnxruntime module surface that bio_automation.ort_providers uses. No GPU, no real model."""
from __future__ import annotations

from types import SimpleNamespace


class FakeSession:
    def __init__(self, got, inputs=("pixel_values",), outputs=("last_hidden_state", "pooler_output")):
        self._got = list(got)
        self._inputs = inputs
        self._outputs = outputs

    def get_providers(self):
        return list(self._got)

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self._inputs]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self._outputs]


class FakeOrt:
    """available: what get_available_providers() reports. got: what the created session reports from get_providers().
    create_errors: exceptions raised by the first N InferenceSession(...) calls, in order."""

    def __init__(self, available=("CUDAExecutionProvider", "CPUExecutionProvider"), got=("CUDAExecutionProvider", "CPUExecutionProvider"),
                 create_errors=(), preload_error=None, with_preload=True):
        self.available = list(available)
        self.got = list(got)
        self.create_errors = list(create_errors)
        self.preload_error = preload_error
        self.events = []            # ordered: "preload", "session"
        self.session_calls = []     # (path, sess_options, providers) per InferenceSession call
        if with_preload:
            self.preload_dlls = self._preload
        self.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="ENABLE_ALL")

    def _preload(self):
        self.events.append("preload")
        if self.preload_error is not None:
            raise self.preload_error

    def get_available_providers(self):
        return list(self.available)

    def SessionOptions(self):
        return SimpleNamespace(intra_op_num_threads=None, graph_optimization_level=None)

    def InferenceSession(self, path, sess_options=None, providers=None):
        self.events.append("session")
        self.session_calls.append((path, sess_options, providers))
        if self.create_errors:
            raise self.create_errors.pop(0)
        return FakeSession(self.got)
