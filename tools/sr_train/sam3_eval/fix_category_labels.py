"""One-time fix for the already-built weak_categories COCO dataset: each image got
TWO annotations for the SAME box -- one tagged with the generic per-group word
(GENERIC_BY_GROUP in build_coco_dataset.py: "mushroom"/"plant"/"slime mold"), one
tagged with the species-level common name/taxon. The species-level duplicate ballooned
the category count to 690 (one per distinct species encountered), which is what the AWS
training OOM traced back to: Sam3ImageDataset's train loader has no category_chunk_size
override (val does), so COCO_FROM_JSON defaults to chunk_size=len(categories)=690,
unchunked, and every training image queried all 690 categories at once (measured: a
5184x690-scale attention allocation of ~593GB). It's also not what we want trained at
all -- the fine-tune target is the 3 broad groups, not 690 individual species.

Fix: drop the species-level annotation per image, keep only the generic-category one,
and renumber categories down to exactly the 3 intended ones (same ids/order in both
train and test, matching build_coco_dataset.py's GENERIC_BY_GROUP: Fungi=1, Plantae=2,
Protozoa=3). No re-run of SAM3 detection needed -- the boxes/masks are already correct,
only the category labeling was wrong.

Run with F:\\venvs\\sam3\\Scripts\\python.exe.
"""
import json
from pathlib import Path

DATASET_ROOT = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\coco_dataset\weak_categories")
# Same order as build_coco_dataset.py's GENERIC_BY_GROUP = {"Fungi": "mushroom", "Plantae": "plant", "Protozoa": "slime mold"}
NEW_CATEGORIES = [
    {"id": 1, "name": "mushroom"},
    {"id": 2, "name": "plant"},
    {"id": 3, "name": "slime mold"},
]
GENERIC_NAME_TO_NEW_ID = {c["name"]: c["id"] for c in NEW_CATEGORIES}


def fix_split(split: str) -> None:
    path = DATASET_ROOT / split / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    old_cat_name = {c["id"]: c["name"] for c in coco["categories"]}

    kept = []
    for ann in coco["annotations"]:
        name = old_cat_name[ann["category_id"]]
        if name in GENERIC_NAME_TO_NEW_ID:
            ann["category_id"] = GENERIC_NAME_TO_NEW_ID[name]
            kept.append(ann)
        # else: species-level duplicate for the same box -- dropped.

    n_images = len(coco["images"])
    assert len(kept) == n_images, (
        f"{split}: expected exactly one generic annotation per image "
        f"({n_images} images) after filtering, got {len(kept)} -- did not touch the file."
    )

    coco["annotations"] = kept
    coco["categories"] = NEW_CATEGORIES
    path.write_text(json.dumps(coco))
    print(f"{split}: {n_images} images, {len(kept)} annotations (was {len(old_cat_name)} categories, now 3)")


if __name__ == "__main__":
    for split in ("train", "test"):
        fix_split(split)
