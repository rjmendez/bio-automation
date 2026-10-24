import json
import torch
import numpy as np
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.5)  # see raw box-prompt output, no text gate

OUT_DIR = r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\results_general\box_prompt_test"
import os
os.makedirs(OUT_DIR, exist_ok=True)

cases = [
    # (path, box_cxcywh_norm, label)
    (r"F:\Bio-export\adobe-stock\ready\2024-10-14_DSC_0249-Enhanced-NR (2025_06_03 19_23_22 UTC).jpg",
     (0.5, 0.5, 0.6, 0.6), "lycogala_topaz_seeded"),
    (r"F:\Bio-export\adobe-stock\ready\2025-05-31_DSC_3576-Enhanced-NR (2025_06_03 19_23_22 UTC).jpg",
     (0.5, 0.5, 0.5, 0.5), "tremella_centered"),
    (r"F:\Bio-export\adobe-stock\ready\2024-12-01_DSC_4563-Enhanced-NR (2025_06_03 19_23_22 UTC).jpg",
     (0.5, 0.5, 0.5, 0.5), "exidia_centered"),
]

for path, box, name in cases:
    img = Image.open(path).convert("RGB")
    state = processor.set_image(img)
    out = processor.add_geometric_prompt(box=list(box), label=True, state=state)
    scores = out["scores"]
    n = scores.numel()
    print(f"{name}: n_instances={n} top_score={scores.max().item() if n else 0:.4f}")
    if n:
        mask = out["masks"][scores.argmax()]
        m = np.asarray(mask.float().cpu()).squeeze()
        thumb = Image.fromarray((m * 255).clip(0, 255).astype(np.uint8), mode="L")
        thumb.thumbnail((500, 500))
        thumb.save(f"{OUT_DIR}/{name}_mask.png")
        img_thumb = img.copy(); img_thumb.thumbnail((500, 500))
        img_thumb.save(f"{OUT_DIR}/{name}_orig.jpg")
