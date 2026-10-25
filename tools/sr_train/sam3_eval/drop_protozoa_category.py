"""Drop the Protozoa/"slime mold" category from the weak_categories dataset: only 4
training images and 1 test image total (severe class imbalance, 852 mushroom / 1194
plant / 4 slime mold in train) -- the likely cause of a recurring, eventually permanent
NaN cascade during training (reproduced with the backbone fully frozen too, ruling out
general backbone-instability and pointing at the rare category's own scoring-head
parameters instead). Each image has exactly one annotation (see fix_category_labels.py),
so dropping slime-mold annotations means dropping those images outright, not leaving
zero-annotation orphans.

Run with F:\\venvs\\sam3\\Scripts\\python.exe.
"""
import json
from pathlib import Path

DATASET_ROOT = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\coco_dataset\weak_categories")
DROP_CATEGORY_NAME = "slime mold"


def drop_split(split: str) -> None:
    path = DATASET_ROOT / split / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    drop_id = next(cid for cid, name in cat_name.items() if name == DROP_CATEGORY_NAME)

    kept_anns = [a for a in coco["annotations"] if a["category_id"] != drop_id]
    kept_image_ids = {a["image_id"] for a in kept_anns}
    n_images_before = len(coco["images"])
    coco["images"] = [im for im in coco["images"] if im["id"] in kept_image_ids]
    coco["annotations"] = kept_anns
    coco["categories"] = [c for c in coco["categories"] if c["id"] != drop_id]

    path.write_text(json.dumps(coco))
    print(
        f"{split}: {n_images_before} -> {len(coco['images'])} images "
        f"(dropped {n_images_before - len(coco['images'])} {DROP_CATEGORY_NAME!r}), "
        f"{len(coco['annotations'])} annotations, categories now {cat_name_after(coco)}"
    )


def cat_name_after(coco) -> list:
    return [c["name"] for c in coco["categories"]]


if __name__ == "__main__":
    for split in ("train", "test"):
        drop_split(split)
