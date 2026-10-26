import torch
from PIL import Image
from pathlib import Path

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
IMG = r"F:\Bio-export\adobe-stock\ready\2024-11-23_DSC_4412-Enhanced-NR (2025_06_03 19_23_22 UTC).jpg"

model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.0)  # see everything

img = Image.open(IMG).convert("RGB")
state = processor.set_image(img)

for prompt in ["spider", "a spider", "insect", "spider web", "orb weaver"]:
    out = processor.set_text_prompt(prompt=prompt, state=state)
    scores = out["scores"]
    n = scores.numel()
    top = scores.max().item() if n else 0.0
    n_above_half = (scores > 0.5).sum().item() if n else 0
    print(f"prompt={prompt!r:16} n_candidates={n:4d} max_score={top:.4f} n_above_0.5={n_above_half}")
