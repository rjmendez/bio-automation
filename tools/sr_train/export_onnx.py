"""Export a trained TinySR2x checkpoint to ONNX, fixed [1,3,128,128] -> [1,3,256,256],
matching bio_automation.topaz._run_tiled's tiling contract (128x128 tiles, scale 2) so
it drops into the same tiler Topaz's ghqv2 model already uses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TinySR2x  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = TinySR2x()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3, 128, 128)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["netInput"], output_names=["netOutput"],
        opset_version=17, dynamic_axes=None,  # fixed shape: matches the fixed-tile OpenVINO models it sits beside
        dynamo=False,  # the default dynamo exporter needs onnxscript; legacy path doesn't
    )
    print(f"exported: {args.out}")
