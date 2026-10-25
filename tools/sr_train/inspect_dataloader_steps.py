"""Fast (no model forward, pure data loading) scan of the real train dataloader, same
seed/epoch as the AWS runs, looking for the specific example that caused a loss spike to
7.80e+02 at step 210 (then NaN from step 211/265 onward in two separate runs with
different LR -- same step region both times, strongly suggesting a specific bad example
in the shuffled order rather than a generic LR-sensitivity issue).

Prints every step's post-augmentation box (normalized cxcywh, what the model actually
sees) plus image id and category, flagging anything degenerate.

Run with F:\\venvs\\sam3\\Scripts\\python.exe.
"""
import os
import sys
import tempfile
from pathlib import Path

os.chdir(Path(__file__).resolve().parent / "sam3_vendor")
sys.path.insert(0, str(Path(__file__).resolve().parent / "sam3_vendor"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from hydra import compose, initialize_config_module  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from sam3.train.train import register_omegaconf_resolvers  # noqa: E402

N_STEPS = 320

register_omegaconf_resolvers()
initialize_config_module("sam3.train", version_base="1.2")
cfg = compose(
    config_name="configs/custom/weak_categories_aws.yaml",
    overrides=[
        r"paths.roboflow_vl_100_root=D:\dev\bio-sr-train\tools\sr_train\sam3_eval\coco_dataset",
        "scratch.num_train_workers=0",  # avoid Windows multiprocessing spawn (needs __main__ guard)
    ],
)

store_file = Path(tempfile.gettempdir()) / "sam3_inspect_dataloader.store"
store_file.unlink(missing_ok=True)
dist.init_process_group(backend="gloo", rank=0, world_size=1,
                         init_method=f"file:///{store_file.as_posix()}")

print("Building the real training dataset + dataloader (seed matches trainer.py)...")
train_dataset = instantiate(cfg.trainer.data.train)
dataloader = train_dataset.get_loader(epoch=0)

cats = {1: "mushroom", 2: "plant", 3: "slime mold"}

for step, raw_batch in enumerate(dataloader):
    if step >= N_STEPS:
        break
    _, batch = raw_batch.popitem()
    img_id = int(batch.find_metadatas[0].original_image_id[0].item())
    cat_id = int(batch.find_metadatas[0].original_category_id[0].item())
    boxes = batch.find_targets[0].boxes  # normalized cxcywh
    n_boxes = boxes.shape[0] if boxes is not None else 0
    flags = []
    if n_boxes == 0:
        flags.append("ZERO BOXES")
    else:
        for b in boxes.reshape(-1, 4).tolist():
            cx, cy, w, h = b
            if not all(torch.isfinite(torch.tensor(b))):
                flags.append(f"NON-FINITE box={b}")
            elif w <= 0 or h <= 0:
                flags.append(f"DEGENERATE box={b}")
            elif w > 1.5 or h > 1.5 or cx < -0.5 or cx > 1.5 or cy < -0.5 or cy > 1.5:
                flags.append(f"OUT-OF-RANGE box={b}")
    marker = "  <<<< " + " | ".join(flags) if flags else ""
    print(f"[{step}] img_id={img_id} cat={cats.get(cat_id, cat_id)} n_boxes={n_boxes} "
          f"boxes={boxes.reshape(-1, 4).tolist() if boxes is not None else None}{marker}",
          flush=True)

print(f"\nScanned {min(N_STEPS, step+1)} steps.")
