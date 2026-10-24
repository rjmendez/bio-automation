"""Produce real before/after upscale examples (bicubic vs HAT-S zero-shot vs the M3b
HAT-S+LoRA fine-tune, the only model that's actually won in this whole project) on
real stock photo crops, and publish them into the gallery so the owner can see the
result rather than just trust the PSNR numbers. Crop size 256 (the exact size used
throughout this project's eval work -- 512 OOMs on this desktop's contended 12GB cards:
HAT's overlap cross-attention relative-position-bias term scales badly with input size,
measured directly at "Tried to allocate 3.38 GiB" on a 512px input). Models are loaded
and run one at a time, never both resident at once, for the same memory reason.

Run with F:\\venvs\\bio-sr-train\\Scripts\\python.exe.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "hat_vendor"))
from lora import inject_lora_into_hat  # noqa: E402
from train_lora_hat import load_hat_s_x2  # noqa: E402

ZERO_SHOT_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat-weights\HAT-S_SRx2.pth")
LORA_WEIGHTS = Path(r"D:\bio-cache\sr-train\hat_s_lora_r8.pt")
SOURCE_DIR = Path(r"F:\Bio-export\adobe-stock\ready")
GALLERY_OUT = Path(r"D:\bio-results\gallery\_examples\hat-s-upscale-demo")
CROP = 256
N_EXAMPLES = 6


def load_zero_shot(device):
    return load_hat_s_x2(ZERO_SHOT_WEIGHTS, device)


def load_fine_tuned(device):
    ckpt = torch.load(LORA_WEIGHTS, map_location="cpu")
    lora_sd = ckpt["lora_state"]
    rank = ckpt.get("rank", 8)
    inject_lora_into_hat(model := load_hat_s_x2(ZERO_SHOT_WEIGHTS, device), r=rank)
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)
    real_missing = [k for k in missing if "lora" in k.lower()]
    if real_missing:
        raise RuntimeError(f"LoRA weights didn't load onto the expected keys: {real_missing[:5]}")
    print(f"LoRA weights loaded (rank={rank}): {len(lora_sd)} tensors, {len(unexpected)} unexpected keys ignored")
    return model.to(device).eval()


def run_model(model, crop_u8, device):
    t = torch.from_numpy(crop_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (out * 255.0).round().astype(np.uint8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU-only (standing rule)")

    all_files = sorted(SOURCE_DIR.glob("*.jpg"))
    rng = random.Random(1234)  # same seed convention as this project's eval split
    shuffled = all_files[:]
    rng.shuffle(shuffled)
    held_out = shuffled[:80]  # matches train_lora_hat.py's --held-out 80 -- these were
    # never trained on, so the examples are a fair held-out demo, not cherry-picked train data.

    GALLERY_OUT.mkdir(parents=True, exist_ok=True)

    # Pick crops first (cheap, CPU-only), then run each model fully, one at a time, so
    # only one HAT-S instance is ever resident on this contended desktop GPU.
    crops = []
    for p in held_out:
        if len(crops) >= N_EXAMPLES:
            break
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < CROP or h < CROP:
                continue
            x0, y0 = (w - CROP) // 2, (h - CROP) // 2
            crop = np.asarray(im.crop((x0, y0, x0 + CROP, y0 + CROP)))
        crops.append((p.stem, crop))
    print(f"Selected {len(crops)} held-out crops ({CROP}px, center crop, not cherry-picked)")

    print("Loading zero-shot HAT-S...")
    model_zs = load_zero_shot(device)
    zs_outs = [run_model(model_zs, c, device) for _, c in crops]
    del model_zs
    torch.cuda.empty_cache()

    print("Loading HAT-S + LoRA fine-tune...")
    model_ft = load_fine_tuned(device)
    ft_outs = [run_model(model_ft, c, device) for _, c in crops]
    del model_ft
    torch.cuda.empty_cache()

    for (name, crop), zs_out, ft_out in zip(crops, zs_outs, ft_outs):
        bicubic = np.asarray(Image.fromarray(crop).resize((CROP * 2, CROP * 2), Image.BICUBIC))
        out_dir = GALLERY_OUT / name
        out_dir.mkdir(exist_ok=True)
        Image.fromarray(crop).save(out_dir / f"0_original_{CROP}px_crop.jpg", quality=95)
        Image.fromarray(bicubic).save(out_dir / "1_bicubic_2x.jpg", quality=95)
        Image.fromarray(zs_out).save(out_dir / "2_hat_s_zeroshot_2x.jpg", quality=95)
        Image.fromarray(ft_out).save(out_dir / "3_hat_s_lora_finetuned_2x.jpg", quality=95)

        # Side-by-side composite for quick visual comparison in the gallery grid view.
        strip = Image.new("RGB", (CROP * 2 * 2, CROP * 2))
        strip.paste(Image.fromarray(bicubic), (0, 0))
        strip.paste(Image.fromarray(ft_out), (CROP * 2, 0))
        strip.save(out_dir / "4_bicubic_vs_lora_sidebyside.jpg", quality=92)

        print(f"{name} -> {out_dir}")

    print(f"\nWrote {len(crops)} examples to {GALLERY_OUT}")


if __name__ == "__main__":
    main()
