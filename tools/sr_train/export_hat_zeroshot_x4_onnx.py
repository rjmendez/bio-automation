"""Export the official, unmodified HAT-S 4x checkpoint (zero-shot, no LoRA) to
ONNX, for registration in lce's model registry.

Why zero-shot and not the LoRA fine-tune: the HAT-S 4x LoRA attempt (rank 8,
lr 1.5e-4, 1500 steps, see cloud/user-data-lora-x4.sh) failed its M3b gate --
the best checkpoint was step 0, i.e. training never beat the zero-shot
baseline. But the zero-shot baseline itself is a real, measured win over
bicubic (PSNR 45.77dB vs 44.46dB, +1.31dB; LPIPS 0.1469 vs higher for bicubic;
sharper deployment Laplacian-variance 2.89 vs 1.02 -- see that run's own
"zero-shot baseline (reproduced in this run)" log line). So there is nothing
to merge here, no adapter -- this just exports the plain official checkpoint,
same as export_hat_lora_onnx.py minus inject_lora_into_hat/the adapter
load_state_dict.

Verifies the exported graph's output against the live PyTorch model on two
different tile sizes before writing anything final, same as the x2 LoRA
export -- HAT's windowed/overlapping attention is exactly the kind of
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

from train_lora_hat import load_hat_s  # noqa: E402

ZERO_SHOT_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat-weights\HAT-S_SRx4.pth")
OUT_ONNX = Path(r"D:\bio-cache\sr-train\hat_s_zeroshot_x4.onnx")
SCALE = 4
TILE = 256


def main():
    # CPU, not CUDA: tracing costs more memory than plain inference and OOM'd
    # on the contended local GPU for the x2 export; same precaution here.
    device = torch.device("cpu")
    print(f"Building HAT-S x{SCALE} (zero-shot, no LoRA) on {device}...")
    model = load_hat_s(ZERO_SHOT_WEIGHTS, device, scale=SCALE).eval()

    rng = np.random.RandomState(0)
    dummy = torch.from_numpy(rng.rand(1, 3, TILE, TILE).astype(np.float32)).to(device)

    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()
    print(f"PyTorch output: {torch_out.shape}, range [{torch_out.min():.4f}, {torch_out.max():.4f}]")

    print(f"Exporting to {OUT_ONNX}...")
    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: the new dynamo exporter hits a real torch.onnx bug on this
    # model's type_as() buffer op (ValueError: Key 'c_mean' does not match the
    # name of the value 'type_as'), same as the x2 export.
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
