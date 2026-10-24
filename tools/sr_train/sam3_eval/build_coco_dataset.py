"""Build a COCO-format instance-segmentation dataset for the 3 weak categories (Fungi,
Plantae, Protozoa). Box seeding is content-derived, never guessed: for each image, SAM3's
own text-prompted detection runs first, with the per-group generic word at a low
threshold (0.1), to find a real candidate box. If that genuinely finds nothing, the photo
is DROPPED -- no centered-box or other composition-guess fallback. An earlier version of
this script used a centered-box guess as the primary seed; that was rejected (correctly)
as a methodological flaw: it risks SAM3 confidently segmenting whatever prominent object
sits in the center, which the confidence filter below would not catch, since that
confidently-wrong object would still score high. Text-detection-first avoids that failure
mode because the seed itself already required the model to recognize the right CONCEPT,
not just "find something in the middle."

Saves incrementally (every 25 images) so a crash on a 4800+ image run doesn't lose
completed work -- an earlier version of this script only wrote output at the very end.

Run with F:\\venvs\\sam3\\Scripts\\python.exe.
"""
import json
import random
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

import argparse
import os

_ap = argparse.ArgumentParser()
_ap.add_argument("--pool-file", default=r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\weak_train_pool.json")
_ap.add_argument("--dataset-root", default=r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\coco_dataset")
_args, _ = _ap.parse_known_args()

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
POOL = Path(_args.pool_file)
DATASET_ROOT = Path(_args.dataset_root)
SUPERCATEGORY = "weak_categories"
TRAIN_DIR = DATASET_ROOT / SUPERCATEGORY / "train"
TEST_DIR = DATASET_ROOT / SUPERCATEGORY / "test"
# Images are written ONCE to a shared folder, never copied/moved: the train/test split
# is recomputed by reshuffling the growing kept_meta list at every checkpoint (same seed,
# different length -> different boundaries each time), so physically placing a file in
# train/ or test/ at save time would orphan it the moment a later checkpoint reassigns
# it to the other split. Both COCO jsons reference the shared folder via a relative path.
IMAGES_DIR = DATASET_ROOT / SUPERCATEGORY / "images"
for d in (TRAIN_DIR, TEST_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)
MIN_BOX_PROMPT_SCORE = 0.6
MIN_DETECT_SCORE = 0.1  # low bar for the SEED box -- refinement, not the final answer
TEST_FRACTION = 0.15
CHECKPOINT_EVERY = 25

GENERIC_BY_GROUP = {"Fungi": "mushroom", "Plantae": "plant", "Protozoa": "slime mold"}


def _save_checkpoint(kept_meta, dropped, final=False):
    rng = random.Random(20261023)
    rows = list(kept_meta)
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * TEST_FRACTION)) if rows else 0
    splits = {"test": rows[:n_test], "train": rows[n_test:]}

    for split_name, split_rows in splits.items():
        out_dir = TRAIN_DIR if split_name == "train" else TEST_DIR
        images, annotations, categories = [], [], {}
        next_cat_id, next_ann_id = 1, 1
        for r in split_rows:
            images.append({"id": r["obs_id"], "file_name": f"../images/{r['file_name']}",
                            "width": r["w"], "height": r["h"]})
            for cat_name in r["cat_names"]:
                if cat_name not in categories:
                    categories[cat_name] = next_cat_id
                    next_cat_id += 1
                annotations.append({
                    "id": next_ann_id, "image_id": r["obs_id"], "category_id": categories[cat_name],
                    "bbox": r["coco_bbox"], "segmentation": r["rle"], "area": r["area"], "iscrowd": 0,
                })
                next_ann_id += 1
        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": cid, "name": name} for name, cid in categories.items()]}
        (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    (DATASET_ROOT / "dropped.json").write_text(json.dumps(dropped, indent=2))
    tag = "FINAL" if final else "checkpoint"
    print(f"[{tag}] saved {len(kept_meta)} kept, {len(dropped)} dropped "
          f"(train={len(splits['train'])}, test={len(splits['test'])})")


def main():
    items = json.loads(POOL.read_text(encoding="utf-8"))
    model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    kept_meta = []
    dropped = []

    for i, item in enumerate(items):
        t0 = time.time()
        try:
            img = Image.open(item["path"]).convert("RGB")
            w, h = img.size
            state = processor.set_image(img)

            # Step 1: content-derived seed box via low-threshold text detection. No
            # composition-guess fallback: if SAM3 itself finds no real evidence of the
            # concept even at a low bar, the photo is dropped, not guessed at.
            generic_prompt = GENERIC_BY_GROUP[item["iconic"]]
            processor.set_confidence_threshold(MIN_DETECT_SCORE, state=state)
            det = processor.set_text_prompt(prompt=generic_prompt, state=state)
            det_scores = det["scores"]
            if det_scores.numel() == 0:
                dropped.append({"obs_id": item["obs_id"], "reason": "no detection (below 0.1)"})
                processor.set_confidence_threshold(MIN_BOX_PROMPT_SCORE, state=state)
                continue
            top_det = det_scores.argmax()
            det_box_xyxy = det["boxes"][top_det].float().cpu().tolist()  # already in pixel coords
            x0, y0, x1, y1 = det_box_xyxy
            box_cxcywh = ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
            processor.set_confidence_threshold(MIN_BOX_PROMPT_SCORE, state=state)

            # Step 2: refine with box-prompting.
            out = processor.add_geometric_prompt(box=list(box_cxcywh), label=True, state=state)
            scores = out["scores"]
            if scores.numel() == 0:
                dropped.append({"obs_id": item["obs_id"], "reason": "no box-prompt instance"})
                continue
            top_idx = scores.argmax()
            top_score = float(scores[top_idx].item())
            if top_score < MIN_BOX_PROMPT_SCORE:
                dropped.append({"obs_id": item["obs_id"], "reason": f"low score {top_score:.3f}"})
                continue

            mask = out["masks"][top_idx]
            m = np.asarray(mask.cpu()).squeeze().astype(np.uint8)
            if m.sum() == 0:
                dropped.append({"obs_id": item["obs_id"], "reason": "empty mask"})
                continue

            ys, xs = np.where(m > 0)
            bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            coco_bbox = [bx0, by0, bx1 - bx0, by1 - by0]
            rle = mask_util.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")

            local_name = f"{item['obs_id']}.jpg"
            img.save(IMAGES_DIR / local_name, quality=92)

            cat_names = {generic_prompt, (item.get("common") or item["taxon"]).lower()}
            kept_meta.append({
                "obs_id": item["obs_id"], "file_name": local_name, "w": w, "h": h,
                "coco_bbox": coco_bbox, "rle": rle, "area": float(m.sum()),
                "cat_names": cat_names, "score": top_score,
            })
            print(f"[{i+1}/{len(items)}] {item['iconic']:10s} obs_id={item['obs_id']} "
                  f"score={top_score:.3f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            dropped.append({"obs_id": item["obs_id"], "reason": str(e)})
            print(f"[{i+1}/{len(items)}] obs_id={item['obs_id']} ERROR: {e}")
        finally:
            # First full run dropped 1402/4864 (28.8%) to CUDA OOM, not genuine misses --
            # VRAM fragmentation accumulating across thousands of sequential forward passes
            # in one long-lived process. Clearing the allocator cache every image is the
            # direct fix the OOM error itself suggested (PYTORCH_CUDA_ALLOC_CONF is set at
            # launch for the complementary expandable_segments fix).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(kept_meta, dropped)

    _save_checkpoint(kept_meta, dropped, final=True)
    print(f"\nKept {len(kept_meta)} of {len(items)} (all content-detected, no composition "
          f"guesses). Dropped {len(dropped)} (see dropped.json for reasons).")


if __name__ == "__main__":
    main()
