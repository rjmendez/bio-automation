"""Full-size (not cropped) before/after examples for the gallery: original, upscaled
(HAT-S+LoRA 2x, tiled), and enhanced (Topaz sharpen -- the same production slb path,
called read-only as a library import, production code never modified or restarted).
Tiled upscale is slow on this desktop (measured: 713ms/256px tile), so this is deliberately
run on just a couple of photos, not a batch.

Run with F:\\venvs\\bio-sr-train\\Scripts\\python.exe (has openvino for the Topaz call).
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "hat_vendor"))
sys.path.insert(0, r"D:\apps\bio-automation\src")  # read-only library import, no writes

from lora import inject_lora_into_hat  # noqa: E402
from train_lora_hat import load_hat_s_x2  # noqa: E402
from hat_tiled import upscale_2x_tiled  # noqa: E402
from bio_automation.topaz import ModelCache, sharpen  # noqa: E402

ZERO_SHOT_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat-weights\HAT-S_SRx2.pth")
LORA_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat_s_lora_r8.pt")
SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")
GALLERY_OUT = Path(r"D:\bio-results\gallery\_examples\hat-s-fullsize-demo")
N_EXAMPLES = 2


def load_fine_tuned(device):
    model = load_hat_s_x2(ZERO_SHOT_WEIGHTS, device)
    ckpt = torch.load(LORA_WEIGHTS, map_location="cpu")
    inject_lora_into_hat(model, r=ckpt.get("rank", 8))
    model.load_state_dict(ckpt["lora_state"], strict=False)
    return model.to(device).eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU-only (standing rule)")

    all_files = sorted(SOURCE_DIR.glob("*.jpg"))
    rng = random.Random(1234)
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    held_out = shuffled[:80]  # same held-out split as the crop demo and LoRA training

    print("Loading HAT-S + LoRA fine-tune...")
    model = load_fine_tuned(device)
    topaz_cache = ModelCache(max_models=1)

    GALLERY_OUT.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in held_out:
        if made >= N_EXAMPLES:
            break
        t0 = time.time()
        with Image.open(p) as im:
            img_rgb = np.asarray(im.convert("RGB"))
        w, h = img_rgb.shape[1], img_rgb.shape[0]
        print(f"[{made+1}/{N_EXAMPLES}] {p.stem} ({w}x{h}, {w*h/1e6:.1f}MP) -- tiling upscale, this is slow...")

        upscaled = upscale_2x_tiled(img_rgb, model, device, tile=256, overlap=32)
        print(f"  upscale done in {time.time()-t0:.0f}s ({upscaled.shape[1]}x{upscaled.shape[0]})")

        t1 = time.time()
        img_bgr = img_rgb[:, :, ::-1].copy()
        enhanced_bgr = sharpen(img_bgr, topaz_cache, auto_params=True, gate=True)
        enhanced_rgb = enhanced_bgr[:, :, ::-1]
        same_as_input = enhanced_bgr is img_bgr
        print(f"  Topaz sharpen done in {time.time()-t1:.1f}s (gated/no-op: {same_as_input})")

        out_dir = GALLERY_OUT / p.stem
        out_dir.mkdir(exist_ok=True)
        Image.fromarray(img_rgb).save(out_dir / "0_original.jpg", quality=95)
        Image.fromarray(upscaled).save(out_dir / "1_upscaled_hat_s_lora_2x.jpg", quality=95)
        Image.fromarray(enhanced_rgb).save(out_dir / "2_enhanced_topaz_sharpen.jpg", quality=95)
        (out_dir / "info.txt").write_text(
            f"source: {p}\noriginal: {w}x{h} ({w*h/1e6:.1f}MP)\n"
            f"upscaled: {upscaled.shape[1]}x{upscaled.shape[0]} (HAT-S+LoRA r8, tiled 256px/32px overlap)\n"
            f"enhanced: Topaz slb sharpen (t_param=0.25, gated); no-op if gate rejected = {same_as_input}\n"
            f"total time: {time.time()-t0:.0f}s\n"
        )
        print(f"  -> {out_dir} (total {time.time()-t0:.0f}s)")
        made += 1

    print(f"\nWrote {made} full-size examples to {GALLERY_OUT}")


if __name__ == "__main__":
    main()
