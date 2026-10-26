"""Shared helper for lce gallery/demo scripts: force onnxruntime onto CPU,
bypassing its automatic CUDA attempt entirely.

History: CUDA initially silently fell back to CPU (onnxruntime's CUDA 13
provider couldn't find cublas64_13.dll/cudart64_13.dll). Adding the system
CUDA Toolkit v13.0 + CUDNN v9.13 DLL dirs made it load (get_providers() ==
['CUDAExecutionProvider', ...]) -- the same recipe that gets Topaz's
dnt_beta model a real ~28x GPU speedup (35ms/512px tile, see Loci finding
44ae881f-13be-4aca-b158-1e8d36405125). For lce's models it does NOT help,
confirmed with real profiling, not just wall-clock timing:

  HAT upscale (hat-stock-lora-x2): ONNX Runtime's own profiler
  (SessionOptions.enable_profiling=True) plus nvidia-smi sampled
  CONCURRENTLY (not before/after) across 2 independent 15-real-tile runs:
    - 99.9% of node time correctly attributed to CUDAExecutionProvider --
      not a silent CPU fallback, CUDA genuinely is active.
    - Tile 0 takes ~1.6s; every tile after that takes 72-97s. Reproducible,
      both runs.
    - GPU0 (2080 Ti, 11GB) sits at ~10900-10989MB used (97%) throughout --
      even on a SECOND run with Ollama's /api/ps confirmed showing zero
      models loaded (ruling out Ollama contention directly, not just
      inferred from an idle nvidia-smi snapshot beforehand, which was an
      earlier, weaker check using Loci's model_lease_acquire that turned
      out not to have evicted anything -- GPU1's free-memory reporting bug
      let it report "granted" without touching GPU0 at all).
    - GPU utilization oscillates 0-100% second to second rather than
      sustained high -- a stall/launch-overhead signature, not genuine
      sustained compute.
    - Per-op breakdown: Conv (51-64% of total time) and Gelu (20-25%)
      dominate, with durations architecturally implausible for 256px
      tiles (Gelu on a tensor this size should be microseconds).
    Root mechanism not fully identified (would need Nsight Systems-level
    tracing); most likely WDDM kernel-launch/sync overhead specific to
    this GPU/driver/model combination. Full finding stored in Loci,
    investigation "bio-automation-standing-rules",
    finding 6d01200e-8cfe-43df-ba59-d5503d87d7df.

  NAFNet sharpen (nafnet-fp32): crashes outright on CUDA EP --
  DequantizeLinear: "Unsupported quantization type". The cuDNN dir was
  present when this happened, so not a DLL-path issue -- more likely a
  real CUDA EP op-coverage gap for this model's quantization scheme.

This does NOT mean GPU is broken on this machine in general -- Topaz's
model gets a real win with the identical DLL setup, and native PyTorch
CUDA works fine elsewhere (e.g. the isolated D:\\venvs\\sam3-image venv for
SAM3 image inference). The pathology is specific to lce's current HAT/
NAFNet ONNX exports under onnxruntime's CUDA EP on this box. Force CPU for
these models instead of fighting it further.
"""
import os

os.environ.setdefault("LCE_CUDA_DEVICE", "")  # harmless if unused; see force_cpu()

_forced = False


def force_cpu() -> None:
    """Call before the first lce model-cache load. Idempotent. Monkeypatches
    onnxruntime.get_available_providers() to report CPU only, so lce's own
    _open_onnx() takes its existing, already-correct CPU-only code path
    (the same path it took before onnxruntime-gpu was even installed) --
    no lce source changes, no DLL-path fragility."""
    global _forced
    if _forced:
        return
    import onnxruntime as ort

    real_available = ort.get_available_providers
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    _forced = True
    return real_available
