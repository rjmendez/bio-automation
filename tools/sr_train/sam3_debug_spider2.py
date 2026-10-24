import torch
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
IMG = r"F:\Bio-export\adobe-stock\ready\2024-11-23_DSC_4412-Enhanced-NR (2025_06_03 19_23_22 UTC).jpg"

model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.5)

img = Image.open(IMG).convert("RGB")
state = processor.set_image(img)

with torch.inference_mode():
    for prompt in ["spider", "a spider", "insect", "spider web", "orb weaver", "bug"]:
        text_outputs = model.backbone.forward_text([prompt], device=processor.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = model._get_dummy_prompt()
        outputs = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )
        out_logits = outputs["pred_logits"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)
        top5 = torch.topk(out_probs.flatten(), k=5).values.tolist()
        print(f"prompt={prompt!r:16} n_queries={out_probs.numel():5d} top5={[round(v,4) for v in top5]} presence_max={presence_score.max().item():.4f}")
