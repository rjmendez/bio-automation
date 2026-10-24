"""Batch-run SAM3 over the cross-taxon general_set.json. Two prompts per photo:
"informed" = the photo's own common name (or taxon name as fallback) -- the realistic
production case, since bio_automation already knows the subject from its iNat match --
and "generic" = a single fixed concept phrase that doesn't assume any identification,
chosen per iconic group so it's a meaningful noun phrase for that kind of subject
(testing whether SAM3 is useful even BEFORE an ID is known, e.g. during triage).
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
ITEMS = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\general_set.json")
OUT_DIR = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\results_general\sam3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERIC_PROMPT_BY_GROUP = {
    "Plantae": "plant",
    "Fungi": "mushroom",
    "Insecta": "insect",
    "Arachnida": "spider",
    "Mammalia": "animal",
    "Aves": "bird",
    "Mollusca": "snail",
    "Reptilia": "lizard",
    "Amphibia": "frog",
    "Actinopterygii": "fish",
    "Protozoa": "organism",
    "Animalia": "animal",
}


def main():
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    results = []
    for i, item in enumerate(items):
        t0 = time.time()
        informed_prompt = (item.get("common") or item.get("taxon") or "organism").lower()
        generic_prompt = GENERIC_PROMPT_BY_GROUP.get(item["iconic"], "organism")
        try:
            img = Image.open(item["path"]).convert("RGB")
            state = processor.set_image(img)
            row = {**item, "informed_prompt": informed_prompt, "generic_prompt": generic_prompt,
                   "prompts": {}, "error": None}
            for key, prompt in [("informed", informed_prompt), ("generic", generic_prompt)]:
                out = processor.set_text_prompt(prompt=prompt, state=state)
                scores = out["scores"]
                n = scores.numel()
                entry = {"n_instances": int(n), "top_score": float(scores.max().item()) if n else 0.0}
                if n:
                    mask = out["masks"][scores.argmax()]
                    m = np.asarray(mask.float().cpu()).squeeze()
                    entry["mask_area_frac"] = float(m.mean())
                else:
                    entry["mask_area_frac"] = 0.0
                row["prompts"][key] = entry
            row["elapsed_s"] = time.time() - t0
            results.append(row)
        except Exception as e:
            results.append({**item, "informed_prompt": informed_prompt, "generic_prompt": generic_prompt,
                             "prompts": {}, "error": str(e)})
        inf = results[-1]["prompts"].get("informed", {}).get("top_score")
        gen = results[-1]["prompts"].get("generic", {}).get("top_score")
        print(f"[{i+1}/{len(items)}] {item['iconic']:12s} obs_id={item['obs_id']} "
              f"informed={informed_prompt!r}->{inf} generic={generic_prompt!r}->{gen} ({time.time()-t0:.1f}s)")

    (OUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} results to {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
