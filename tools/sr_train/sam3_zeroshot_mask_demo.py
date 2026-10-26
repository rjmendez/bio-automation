"""Real text-prompted segmentation masks from SAM3 (Meta's Segment Anything
Model 3), zero-shot -- no fine-tuning. Demonstrates the one thing lce's
MobileSAM-based `segment` tool cannot do: prompt by what the subject actually
IS ("mushroom", "wasp"), not just by a point/grid location, so the mask
follows the subject rather than whatever the grid happens to agree on.

NOT the weak-iNat-category SAM3 fine-tune that bio-sr-train's cloud/user-
data-sam3.sh was training: that training never completed (see the SAM3
training-stability investigation finding in Loci, investigation
"bio-automation-standing-rules") and no fine-tuned checkpoint was ever
saved anywhere -- confirmed by checking S3 (only logs/configs/tensorboard
events, no .pt file). This uses the official pretrained SAM3 checkpoint,
unmodified.

LICENSE NOTE (read before touching this file in a public context): SAM3
code and weights are under Meta's "SAM License" (sam3_vendor/LICENSE),
NOT Apache-2.0/MIT -- it permits use, reproduction, distribution and
modification, but redistributing SAM Materials (code or weights) requires
including a copy of that license, and publishing research results using
SAM3 requires acknowledging it. This repo does neither: the vendored sam3
package (tools/sr_train/sam3_vendor/) is git-ignored (not committed, not
redistributed -- a local clone only) and the checkpoint
(D:\bio-cache\sr-train\sam3\sam3.pt) is never committed either, same
pattern as every other model weight referenced by these gallery scripts.
Only this wrapper script -- which contains none of Meta's code or
weights -- is committed. The mask IMAGES this script produces are outputs
on the owner's own photos, not SAM Materials themselves.

Run with the isolated venv (NOT the global Python -- see the GPU profiling
session's own side-effect lesson about touching a shared interpreter):
  D:\venvs\sam3-image\Scripts\python.exe sam3_zeroshot_mask_demo.py
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_VENDOR = Path(__file__).resolve().parent / "sam3_vendor"
sys.path.insert(0, str(_VENDOR))

CHECKPOINT = Path(r"D:\bio-cache\sr-train\sam3\sam3.pt")
SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")
GALLERY_OUT = Path(r"D:\bio-results\gallery\_examples\sam3-zeroshot-demo")

# Picked by actually looking at the photos (held-out indices 13-15 of the
# same seeded split every other gallery script in this file uses), not
# guessed blind -- a text prompt only means something if it is accurate.
EXAMPLES = [
    (13, "mushroom"),
    (14, "wasp"),
    (14, "hand"),  # same photo, second prompt -- shows SAM3 follows the
                   # PROMPT, not just "whatever's salient", on one image
    # Honest finding, not cherry-picked: "slime mold", "fungus", and "mold"
    # all scored 0 objects on this photo (below the 0.5 confidence_threshold
    # -- "lichen" got closest at 0.39). A plain descriptive phrase instead
    # of jargon did clear the threshold (0.55), which is itself useful to
    # know about SAM3's zero-shot vocabulary.
    (15, "growth on tree bark"),
]

_COLORS = [(255, 60, 60), (60, 140, 255), (60, 220, 90), (230, 200, 40)]


def _held_out_files():
    all_files = sorted(SOURCE_DIR.glob("*.jpg"))
    rng = random.Random(1234)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    return shuffled[:80]


def _overlay(hwc_u8: np.ndarray, masks, color_offset=0) -> np.ndarray:
    overlay = hwc_u8.copy().astype(np.float32)
    for i, mask in enumerate(masks):
        color = np.array(_COLORS[(i + color_offset) % len(_COLORS)], dtype=np.float32)
        sel = mask > 0.5
        overlay[sel] = overlay[sel] * 0.5 + color * 0.5
    return overlay.round().clip(0, 255).astype(np.uint8)


def main():
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    model = build_sam3_image_model(
        checkpoint_path=str(CHECKPOINT), load_from_HF=False, device=device,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=0.5)

    held_out = _held_out_files()
    GALLERY_OUT.mkdir(parents=True, exist_ok=True)

    cur_idx, cur_img, cur_hwc, cur_state = None, None, None, None
    for idx, prompt in EXAMPLES:
        if idx != cur_idx:
            cur_idx = idx
            p = held_out[idx]
            cur_img = Image.open(p).convert("RGB")
            cur_hwc = np.asarray(cur_img)
            cur_state = processor.set_image(cur_img)
            out_dir = GALLERY_OUT / p.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            h, w = cur_hwc.shape[:2]
            Image.fromarray(cur_hwc).save(out_dir / f"0_original_{w}x{h}.jpg", quality=95)
            print(f"[{p.stem}] ({w}x{h})", flush=True)

        processor.reset_all_prompts(cur_state)
        state = processor.set_text_prompt(prompt=prompt, state=cur_state)
        n = len(state.get("scores", []))
        scores = [float(s) for s in state["scores"]] if n else []
        print(f"  prompt='{prompt}': {n} object(s), scores={[f'{s:.2f}' for s in scores]}", flush=True)

        out_dir = GALLERY_OUT / held_out[idx].stem
        safe_prompt = prompt.replace(" ", "_")
        if n == 0:
            print(f"  (no detections above confidence_threshold for '{prompt}')", flush=True)
            continue
        masks = [state["masks"][i].squeeze(0).float().cpu().numpy() for i in range(n)]
        overlay = _overlay(cur_hwc, masks)
        Image.fromarray(overlay).save(out_dir / f"overlay_{safe_prompt}.jpg", quality=92)
        Image.fromarray((masks[0] * 255).astype(np.uint8)).save(
            out_dir / f"mask_{safe_prompt}_top.png"
        )
        print(f"  -> {out_dir}", flush=True)

    print(f"\nWrote SAM3 zero-shot examples to {GALLERY_OUT}")


if __name__ == "__main__":
    main()
