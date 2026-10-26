import torch
import numpy as np
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
torch.autocast(device_type, dtype=torch.bfloat16).__enter__()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate

CHECKPOINT = r"D:\bio-cache\sr-train\sam3\sam3.pt"
IMG = r"F:\Bio-export\adobe-stock\rendered\2025\2025-06-10\DSC_4412.jpg"
OUT_DIR = r"D:\dev\bio-sr-train\tools\sr_train\results\sam3_spider"

model = build_sam3_image_model(checkpoint_path=CHECKPOINT)
processor = Sam3Processor(model, confidence_threshold=0.5)

img = Image.open(IMG).convert("RGB")
state = processor.set_image(img)
img_h, img_w = state["original_height"], state["original_width"]

with torch.inference_mode():
    for prompt in ["spider", "a spider", "insect", "spider web", "orb weaver", "bug"]:
        text_outputs = model.backbone.forward_text([prompt], device=processor.device)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = model._get_dummy_prompt()
        outputs = model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=processor.find_stage,
            geometric_prompt=state["geometric_prompt"], find_target=None,
        )
        out_logits = outputs["pred_logits"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        top1_idx = out_probs.flatten().argmax()
        top_score = out_probs.flatten()[top1_idx].item()
        n_above_half = (out_probs > 0.5).sum().item()

        out_bbox = outputs["pred_boxes"].view(-1, 4)[top1_idx:top1_idx+1]
        out_masks = outputs["pred_masks"].view(-1, *outputs["pred_masks"].shape[-2:])[top1_idx:top1_idx+1]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(processor.device)
        boxes = (boxes * scale_fct[None, :]).float().cpu().tolist()[0]

        mask = interpolate(out_masks.unsqueeze(1).float(), (img_h, img_w), mode="bilinear", align_corners=False).sigmoid()
        mask_np = mask[0, 0].float().cpu().numpy()

        Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode="L").save(
            f"{OUT_DIR}/sam3_v2_top1_{prompt.replace(' ', '_')}.png"
        )
        print(f"prompt={prompt!r:12} top_score={top_score:.4f} n_above_0.5={n_above_half} box_xyxy={[round(v) for v in boxes]} mask_area_frac={(mask_np>0.5).mean():.4f}")
