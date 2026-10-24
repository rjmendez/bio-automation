"""Batch-run SAM3 promptable concept segmentation over the resolved spider set, one model
load for the whole batch. Prompts "spider" and "arachnid" per image (the set includes true
spiders as well as ticks/mites, which genuinely should NOT score high on "spider" -- that's
a correct non-match, not a model failure, tracked via each item's real iNat taxon).
Run with F:\\venvs\\sam3\\Scripts\\python.exe.
"""
import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
SPIDER_SET = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\spider_set.json")
OUT_DIR = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\results\sam3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS = ["spider", "arachnid"]


def main():
    items = json.loads(SPIDER_SET.read_text(encoding="utf-8"))
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    results = []
    for i, item in enumerate(items):
        t0 = time.time()
        try:
            img = Image.open(item["path"]).convert("RGB")
            state = processor.set_image(img)
            row = {**item, "prompts": {}, "error": None}
            for prompt in PROMPTS:
                out = processor.set_text_prompt(prompt=prompt, state=state)
                scores = out["scores"]
                n = scores.numel()
                row["prompts"][prompt] = {
                    "n_instances": int(n),
                    "top_score": float(scores.max().item()) if n else 0.0,
                }
                if n:
                    mask = out["masks"][scores.argmax()]
                    m = np.asarray(mask.float().cpu()).squeeze()
                    row["prompts"][prompt]["mask_area_frac"] = float(m.mean())
                    thumb = Image.fromarray((m * 255).clip(0, 255).astype(np.uint8), mode="L")
                    thumb.thumbnail((400, 400))
                    thumb.save(OUT_DIR / f"{item['obs_id']}_{prompt}_mask.png")
                else:
                    row["prompts"][prompt]["mask_area_frac"] = 0.0
            row["elapsed_s"] = time.time() - t0
            results.append(row)
        except Exception as e:
            results.append({**item, "prompts": {}, "error": str(e)})
        summary = {p: results[-1]["prompts"].get(p, {}).get("top_score") for p in PROMPTS}
        print(f"[{i+1}/{len(items)}] obs_id={item['obs_id']} taxon={item['taxon']!r} "
              f"-> {summary} ({time.time()-t0:.1f}s)")

    (OUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} results to {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
