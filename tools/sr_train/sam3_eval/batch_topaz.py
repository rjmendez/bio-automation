"""Batch-run the ported Topaz s_mask_l_flexible subject_mask() over the resolved spider
set (spider_set.json), one OpenVINO model load for the whole batch (not per-image).
Run with F:\\venvs\\bio-sr-train\\Scripts\\python.exe (has openvino + numpy + Pillow).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # tools/sr_train
from topaz_subject_mask_port import ModelCache, subject_mask  # noqa: E402

import numpy as np
from PIL import Image

SPIDER_SET = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\spider_set.json")
OUT_DIR = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\results\topaz")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    items = json.loads(SPIDER_SET.read_text(encoding="utf-8"))
    cache = ModelCache()
    results = []
    for i, item in enumerate(items):
        t0 = time.time()
        try:
            img = Image.open(item["path"]).convert("RGB")
            img_chw_f32 = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
            prob = subject_mask(img_chw_f32, cache)
            mask_area_frac = float((prob > 0.5).mean())
            thumb = Image.fromarray((prob * 255).clip(0, 255).astype(np.uint8), mode="L")
            thumb.thumbnail((400, 400))
            thumb.save(OUT_DIR / f"{item['obs_id']}_mask.png")
            results.append({**item, "mask_area_frac": mask_area_frac,
                             "mask_mean": float(prob.mean()), "elapsed_s": time.time() - t0,
                             "error": None})
        except Exception as e:
            results.append({**item, "mask_area_frac": None, "error": str(e)})
        print(f"[{i+1}/{len(items)}] obs_id={item['obs_id']} taxon={item['taxon']!r} "
              f"-> {results[-1].get('mask_area_frac')} ({time.time()-t0:.1f}s)")

    (OUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} results to {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
