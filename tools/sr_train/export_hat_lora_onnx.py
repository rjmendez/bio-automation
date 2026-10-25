"""Export the M3b HAT-S + LoRA fine-tune (trained on real stock nature photos,
see gallery_upscale_examples.py / gallery_fullsize_examples.py for the proof)
to ONNX, for registration in lce's local-only model registry.

Not merging LoRA into the base weights manually -- LoRALinear.forward already
computes base_out + lora_out*scaling as plain tensor ops, so torch.onnx.export
traces the whole live model (base + adapters) as one graph. Dynamic H/W axes:
lce's engine tiles at 256x256 by default when a model has no fixed shape
(see lce/engine.py:static_hw), matching how this fine-tune was trained/run.

Verifies the exported graph's output against the live PyTorch model on a real
tile before writing anything final -- HAT's windowed/overlapping attention
(relative position bias, padding-to-window-size) is exactly the kind of
architecture that can silently diverge on ONNX export.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "hat_vendor"))

from lora import inject_lora_into_hat  # noqa: E402
from train_lora_hat import load_hat_s_x2  # noqa: E402

ZERO_SHOT_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat-weights\HAT-S_SRx2.pth")
LORA_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat_s_lora_r8.pt")
OUT_ONNX = Path(r"D:\bio-cache\sr-train\hat_s_lora_r8_stock_x2.onnx")
TILE = 256


def build_model(device):
    model = load_hat_s_x2(ZERO_SHOT_WEIGHTS, device)
    ckpt = torch.load(LORA_WEIGHTS, map_location="cpu")
    inject_lora_into_hat(model, r=ckpt.get("rank", 8))
    model.load_state_dict(ckpt["lora_state"], strict=False)
    return model.to(device).eval()


def main():
    # CPU, not CUDA: the TorchScript tracer OOM'd on the contended local 12GB GPU
    # ("Tried to allocate 384.00 MiB ... 0 bytes is free") even though a plain forward
    # pass at the same 256px tile size works fine -- tracing's extra bookkeeping is
    # the problem, not inference itself. This is a one-time export, not a perf path.
    device = torch.device("cpu")
    print(f"Building HAT-S + LoRA model on {device}...")
    model = build_model(device)

    rng = np.random.RandomState(0)
    dummy = torch.from_numpy(rng.rand(1, 3, TILE, TILE).astype(np.float32)).to(device)

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    print(f"PyTorch output: {torch_out.shape}, range [{torch_out.min():.4f}, {torch_out.max():.4f}]")

    print(f"Exporting to {OUT_ONNX}...")
    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=True (torch 2.11's new default exporter) hits an internal bug converting
    # this model's type_as() buffer op (ValueError: Key 'c_mean' does not match the
    # name of the value 'type_as') -- a real torch.onnx bug, not our model. Falling
    # back to the older, battle-tested TorchScript-tracing exporter.
    torch.onnx.export(
        model,
        (dummy,),
        str(OUT_ONNX),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h2", 3: "w2"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Verifying exported ONNX against PyTorch output on the same tile...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.cpu().numpy()})[0]
    max_abs_diff = np.abs(onnx_out - torch_out).max()
    print(f"ONNX output: {onnx_out.shape}, max abs diff vs PyTorch: {max_abs_diff:.6f}")
    if max_abs_diff > 1e-3:
        raise RuntimeError(
            f"ONNX export diverges from PyTorch (max abs diff {max_abs_diff:.6f} > 1e-3) -- "
            "do not register this file, the export is not faithful."
        )

    print("Verifying a second, non-square/different-size tile (dynamic axes sanity check)...")
    dummy2 = torch.from_numpy(rng.rand(1, 3, 192, 320).astype(np.float32)).to(device)
    with torch.no_grad():
        torch_out2 = model(dummy2).cpu().numpy()
    onnx_out2 = sess.run(None, {"input": dummy2.cpu().numpy()})[0]
    max_abs_diff2 = np.abs(onnx_out2 - torch_out2).max()
    print(f"  shapes: torch {torch_out2.shape}, onnx {onnx_out2.shape}, max abs diff: {max_abs_diff2:.6f}")
    if onnx_out2.shape != torch_out2.shape or max_abs_diff2 > 1e-3:
        raise RuntimeError(
            f"ONNX export fails on a non-training-size tile (shapes {torch_out2.shape} vs "
            f"{onnx_out2.shape}, max abs diff {max_abs_diff2:.6f}) -- dynamic axes are not faithful."
        )

    size_bytes = OUT_ONNX.stat().st_size
    sha256 = hashlib.sha256(OUT_ONNX.read_bytes()).hexdigest()
    print(f"\nOK -- wrote {OUT_ONNX} ({size_bytes} bytes)")
    print(f"sha256 = {sha256}")
    print(f"size_bytes = {size_bytes}")


if __name__ == "__main__":
    main()
