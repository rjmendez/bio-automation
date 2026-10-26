"""Run SAM 3's promptable concept segmentation (text prompt, e.g. "spider") on a real
test photo and dump the resulting mask(s)/boxes/scores, for comparison against the
existing Topaz s_mask_l_flexible-based subject_mask() (see topaz_subject_mask_port.py).

Loads from the local ungated-mirror checkpoint (D:\\bio-cache\\sr-train\\sam3\\sam3.pt) --
no Hugging Face access, no Meta account, confirmed via build_sam3_image_model's
checkpoint_path= parameter in sam3_vendor/sam3/model_builder.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CHECKPOINT = Path(r"D:\bio-cache\sr-train\sam3\sam3.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="spider")
    ap.add_argument("--out-mask-png", type=Path, required=True)
    ap.add_argument("--out-stats-json", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Official usage (examples/sam3_image_predictor_example.ipynb): the checkpoint's
    # weights are bf16 and a plain fp32 forward pass crashes on a dtype mismatch
    # (measured directly: "mat1 and mat2 must have the same dtype, got BFloat16 and
    # Float") -- autocast is required, not optional, for this checkpoint.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(checkpoint_path=str(CHECKPOINT))
    processor = Sam3Processor(model)

    img = Image.open(args.image).convert("RGB")
    state = processor.set_image(img)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    n = len(scores) if hasattr(scores, "__len__") else 0
    stats = {
        "prompt": args.prompt,
        "image": str(args.image),
        "image_size": [img.width, img.height],
        "n_instances": int(n),
        "scores": [float(s) for s in scores] if n else [],
        "boxes": [[float(v) for v in b] for b in boxes] if n else [],
    }

    if n:
        combined = np.zeros((img.height, img.width), dtype=np.float32)
        for m in masks:
            m_arr = np.asarray(m)
            if m_arr.ndim > 2:
                m_arr = m_arr.squeeze()
            combined = np.maximum(combined, m_arr.astype(np.float32))
        stats["mask_area_frac"] = float((combined > 0.5).mean())
        Image.fromarray((combined * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(args.out_mask_png)
    else:
        stats["mask_area_frac"] = 0.0
        Image.new("L", (img.width, img.height), 0).save(args.out_mask_png)

    args.out_stats_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
