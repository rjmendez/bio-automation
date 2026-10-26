"""Shared helper for lce gallery/demo scripts: force onnxruntime onto CPU,
bypassing its automatic CUDA attempt entirely.

History, in order: (1) CUDA silently fell back to CPU with warnings --
onnxruntime's CUDA provider (a CUDA 13 build) couldn't find cublas64_13.dll/
cudart64_13.dll. (2) Added the system CUDA Toolkit v13.0 + CUDNN v9.13 DLL
dirs (os.add_dll_directory + PATH) -- CUDA then loaded and
get_providers() == ['CUDAExecutionProvider', ...]. (3) Measured the REAL
effect on actual gallery runs, 3 times, on 2 different model families:
  - HAT upscale (hat-stock-lora-x2): a short isolated 5-tile test suggested
    ~0.77s/tile (vs ~5.6-7.9s/tile CPU, ~10x) -- but the real full run came
    back at 709.5s and 1240.2s, matching or EXCEEDING the 709.8s CPU
    baseline. Verified this wasn't GPU contention from Ollama (used Loci's
    model_lease_acquire to evict resident models and confirmed GPU0 was
    idle, 6.6GB free, 0% util beforehand -- the GPU run was still slower).
  - NAFNet sharpen: crashed outright (DequantizeLinear: "Unsupported
    quantization type" -- a model whose quantized ops this CUDA EP build
    doesn't support, a failure mode CPU doesn't hit).
Conclusion: CUDA via onnxruntime is not viable for lce's current ONNX
models on this machine -- not a DLL-path problem to keep chasing, a real
dead end measured three times. Force CPU instead of fighting it further.
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
