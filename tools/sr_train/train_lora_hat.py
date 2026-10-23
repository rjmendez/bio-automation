"""M3b: LoRA fine-tune the pretrained HAT-S backbone on the owner's own
crop-upscale data, and gate the result against the HAT-S ZERO-SHOT baseline
benchmark_hat_s.py already measured (+1.35 dB PSNR vs bicubic, LPIPS 0.0441,
DISTS 0.0285, deployment Laplacian-var 18.23).

Whole backbone frozen by default (hat_vendor/hat_arch.py's HAT class). LoRA
adapters (tools/sr_train/lora.py) go on:
  (a) the MLP linears inside every HAB block (the attention-block MLPs), and
  (b) the OCAB module's output projection + MLP linears,
NOT on any Q/K/V projection -- AdaptSR's SwinIR ablation (cited in the M3b
task) found (a) and (b) drive domain-adaptation gain more than Q/K/V tuning.

Same HR/LR construction as benchmark_hat_s.py and validate_real.py
(random_hr_crop + degrade_hr_to_lr on F:\\Bio-export\\adobe-stock\\ready),
same held-out split convention (Random(1234).shuffle, first --held-out files
held out), so the "old" (zero-shot) and "new" (fine-tuned) numbers this script
prints are directly comparable -- the zero-shot numbers are even re-measured
in this same run (should reproduce benchmark_hat_s.py's numbers) rather than
just quoted, so there is one apples-to-apples report instead of two runs that
might have silently drifted apart.

SR/restoration is ill-posed: train L1 keeps falling regardless of whether the
model is overfitting the ~800-ish source images in the crop-upscale pool, so
this script early-stops on a held-out composite score, not on train loss.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "hat_vendor"))
from degrade import degrade_hr_to_lr, random_hr_crop  # noqa: E402
from train import pick_device, PATCH_SIZE  # noqa: E402
from validate_real import psnr, perceptual_scores, _laplacian_var  # noqa: E402
from hat_arch import HAT  # noqa: E402
from lora import inject_lora_into_hat, freeze_backbone_except_lora, lora_state_dict  # noqa: E402


def load_hat_s_x2(weights_path: Path, device) -> HAT:
    model = HAT(
        upscale=2, in_chans=3, img_size=64, window_size=16,
        compress_ratio=24, squeeze_factor=24, conv_scale=0.01, overlap_ratio=0.5,
        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=144,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    )
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt["params_ema"] if "params_ema" in ckpt else ckpt["params"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def split_files(source_dir: Path, held_out: int) -> tuple[list[Path], list[Path]]:
    all_files = sorted(source_dir.glob("*.jpg"))
    shuffled = all_files[:]
    random.Random(1234).shuffle(shuffled)  # same split convention as validate_real.py / benchmark_hat_s.py
    held_out_files = shuffled[:held_out]
    train_files = shuffled[held_out:]
    return held_out_files, train_files


def run_full_eval(model, held_out_files: list[Path], device, seed: int) -> dict:
    """Byte-for-byte port of benchmark_hat_s.py's two eval loops, so the crops
    drawn are identical run to run (fresh rng, same seed, same iteration order)
    and the numbers it prints are directly comparable to the already-measured
    zero-shot baseline. Returns a flat dict of scalars."""
    rng = random.Random(seed)
    model.eval()

    # --- 1. synthetic-bicubic-degradation benchmark ---
    model_psnrs, bicubic_psnrs = [], []
    model_preds, bicubics, hrs = [], [], []
    with torch.no_grad():
        for p in held_out_files:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                hr = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
            lr = degrade_hr_to_lr(hr, rng)
            lr_t = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(lr_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(Image.fromarray(lr).resize((PATCH_SIZE, PATCH_SIZE), Image.BICUBIC))
            model_psnrs.append(psnr(pred_u8, hr))
            bicubic_psnrs.append(psnr(bicubic, hr))
            model_preds.append(pred_u8)
            bicubics.append(bicubic)
            hrs.append(hr)

    model_avg = sum(model_psnrs) / len(model_psnrs)
    bicubic_avg = sum(bicubic_psnrs) / len(bicubic_psnrs)
    model_lpips, model_dists = perceptual_scores(model_preds, hrs, device)
    bicubic_lpips, bicubic_dists = perceptual_scores(bicubics, hrs, device)

    # --- 2. deployment sanity check: genuinely native, undegraded crops ---
    model_vars, bicubic_vars = [], []
    with torch.no_grad():
        for p in held_out_files[:40]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if im.width < PATCH_SIZE or im.height < PATCH_SIZE:
                    continue
                native = random_hr_crop(np.asarray(im), PATCH_SIZE, rng)
            inp_t = torch.from_numpy(native.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            pred = model(inp_t)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            pred_u8 = (pred * 255.0).round().astype(np.uint8)
            bicubic = np.asarray(Image.fromarray(native).resize((PATCH_SIZE * 2, PATCH_SIZE * 2), Image.BICUBIC))
            model_vars.append(_laplacian_var(np.asarray(Image.fromarray(pred_u8).convert("L"))))
            bicubic_vars.append(_laplacian_var(np.asarray(Image.fromarray(bicubic).convert("L"))))
    model_sharp = sum(model_vars) / len(model_vars)
    bicubic_sharp = sum(bicubic_vars) / len(bicubic_vars)

    return dict(
        model_psnr=model_avg, bicubic_psnr=bicubic_avg,
        model_lpips=model_lpips, model_dists=model_dists,
        bicubic_lpips=bicubic_lpips, bicubic_dists=bicubic_dists,
        model_sharp=model_sharp, bicubic_sharp=bicubic_sharp,
        n_synthetic=len(model_psnrs), n_deploy=len(model_vars),
    )


def composite_score(ev: dict) -> float:
    """Single scalar for early-stop plateau detection only -- NOT the gate
    criterion (the gate needs strictly better on PSNR AND LPIPS AND DISTS,
    checked separately at the end on the saved best checkpoint)."""
    return ev["model_psnr"] - 50.0 * ev["model_lpips"] - 50.0 * ev["model_dists"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("hat_s_lora.pt"))
    ap.add_argument("--steps", type=int, default=20_000, help="hard upper cap; --time-budget-s is the real stop")
    ap.add_argument("--time-budget-s", type=float, default=5400,
                     help="stop training after this many seconds regardless of step count, leaving headroom "
                          "under the external process timeout for staging/install/eval/upload -- same pattern "
                          "as aws_train.py.")
    ap.add_argument("--batch-size", type=int, default=8,
                     help="AWS g4dn.xlarge's T4 is dedicated (16GB, no shared-desktop Ollama contention), so "
                          "this can be larger than the local smoke test's bs=2.")
    ap.add_argument("--train-patch-size", type=int, default=256,
                     help="HR patch edge for TRAINING; matches the 256/128 eval patch size (PATCH_SIZE, "
                          "imported from train.py) by default, removing the crop-size mismatch the local smoke "
                          "test had (it used 128 to fit this shared desktop's contended VRAM; AWS's dedicated "
                          "T4 doesn't need that compromise).")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=None, help="default: equal to --rank")
    ap.add_argument("--held-out", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6, help="evals with no composite improvement before early stop")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                     help="if set, write the current best LoRA checkpoint here on every improvement (atomic "
                          "tmp-then-rename), so an external kill (timeout, spot interruption) still leaves a "
                          "usable checkpoint, not just whatever --out would have been at clean exit.")
    ap.add_argument("--result-json", type=Path, default=None,
                     help="dump the final gate numbers as JSON (native float/bool, not numpy) alongside --out.")
    ap.add_argument("--cuda-name-substr", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.cuda_name_substr)
    if device.type != "cuda":
        raise SystemExit("GPU-only training (owner standing rule) -- pass --cuda-name-substr")

    held_out_files, train_files = split_files(args.source_dir, args.held_out)
    print(f"train pool: {len(train_files)} files   held-out: {len(held_out_files)} files   device: {device}")

    model = load_hat_s_x2(args.weights, device)

    # --- reproduce the zero-shot baseline inside THIS run, same code path/seed ---
    zero_shot = run_full_eval(model, held_out_files, device, seed=args.seed)
    print("\n=== zero-shot baseline (reproduced in this run) ===")
    print(f"  PSNR {zero_shot['model_psnr']:.2f} dB (bicubic {zero_shot['bicubic_psnr']:.2f} dB, "
          f"delta {zero_shot['model_psnr'] - zero_shot['bicubic_psnr']:+.2f} dB)")
    print(f"  LPIPS {zero_shot['model_lpips']:.4f}  DISTS {zero_shot['model_dists']:.4f}")
    print(f"  deployment Laplacian-var {zero_shot['model_sharp']:.2f} (bicubic {zero_shot['bicubic_sharp']:.2f})")

    inserted = inject_lora_into_hat(model, r=args.rank, alpha=args.alpha)
    model = model.to(device)  # LoRALinear's lora_A/lora_B are freshly created on CPU; move them to device too
    n_trainable, n_total = freeze_backbone_except_lora(model)
    print(f"\nLoRA adapters inserted: {len(inserted)}  "
          f"(6 RHAGs x [6 HAB blocks x 2 MLP linears + OCAB.proj + OCAB.mlp x 2] = 6x15)")
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)  rank={args.rank}")
    model.eval()  # keep the frozen backbone's drop_path/dropout off; LoRA grads are unaffected by .training

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)

    rng = random.Random(args.seed + 1)  # distinct stream from the eval rng

    train_ps = args.train_patch_size

    def sample_hr_batch(n: int) -> np.ndarray:
        out = np.empty((n, train_ps, train_ps, 3), dtype=np.uint8)
        for i in range(n):
            for _ in range(8):
                p = rng.choice(train_files)
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    if im.width >= train_ps and im.height >= train_ps:
                        out[i] = random_hr_crop(np.asarray(im), train_ps, rng)
                        break
            else:
                raise RuntimeError(f"no file >= {train_ps}px found near {p}")
        return out

    best_composite = composite_score(zero_shot)  # LoRA must beat zero-shot's own composite to count as "improved"
    best_step = 0
    best_lora_state = lora_state_dict(model)  # identity at init (B=0), so this equals "no LoRA effect"
    best_eval = zero_shot
    patience_counter = 0
    stopped_early = False

    def save_checkpoint(lora_state: dict, step: int, elapsed: float) -> None:
        if args.checkpoint_dir is None:
            return
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = args.checkpoint_dir / "hat_s_lora.pt.tmp"
        torch.save({"lora_state": lora_state, "rank": args.rank, "alpha": args.alpha or args.rank,
                    "step": step, "lr": args.lr}, tmp)
        tmp.replace(args.checkpoint_dir / "hat_s_lora.pt")
        (args.checkpoint_dir / "progress.json").write_text(json.dumps({"step": step, "elapsed_s": elapsed}))

    losses: list[float] = []
    t0 = time.time()
    actual_steps = 0
    stopped_on_time_budget = False
    for step in range(args.steps):
        elapsed = time.time() - t0
        if elapsed >= args.time_budget_s:
            print(f"\ntime budget ({args.time_budget_s:.0f}s) reached at step {step}")
            stopped_on_time_budget = True
            break
        actual_steps = step + 1
        hr_uint8 = sample_hr_batch(args.batch_size)
        lr_list = [degrade_hr_to_lr(hr_uint8[i], rng) for i in range(hr_uint8.shape[0])]
        lr = np.stack(lr_list).astype(np.float32) / 255.0
        hr = hr_uint8.astype(np.float32) / 255.0
        lr_t = torch.from_numpy(lr).permute(0, 3, 1, 2).to(device)
        hr_t = torch.from_numpy(hr).permute(0, 3, 1, 2).to(device)

        pred = model(lr_t)
        loss = F.l1_loss(pred, hr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  l1 {loss.item():.4f}  elapsed {time.time() - t0:.1f}s")

        if (step + 1) % args.eval_every == 0:
            ev = run_full_eval(model, held_out_files, device, seed=args.seed)
            score = composite_score(ev)
            print(f"  [eval @ step {step + 1}] PSNR {ev['model_psnr']:.2f}dB  LPIPS {ev['model_lpips']:.4f}  "
                  f"DISTS {ev['model_dists']:.4f}  Laplacian {ev['model_sharp']:.2f}  composite {score:.3f}  "
                  f"(best {best_composite:.3f} @ step {best_step})")
            if score > best_composite:
                best_composite = score
                best_step = step + 1
                best_lora_state = lora_state_dict(model)
                best_eval = ev
                patience_counter = 0
                save_checkpoint(best_lora_state, best_step, time.time() - t0)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  early stop: no composite improvement for {args.patience} evals "
                          f"(train loss alone is not trusted -- SR/restoration is ill-posed)")
                    stopped_early = True
                    break
            model.eval()
            torch.cuda.empty_cache()  # perceptual_scores() builds fresh LPIPS/DISTS (VGG) nets every call

    stop_reason = "early stop" if stopped_early else ("time budget" if stopped_on_time_budget else "max steps reached")
    print(f"\ntraining stopped after {actual_steps} steps ({stop_reason}), best checkpoint @ step {best_step}")

    # Restore the best checkpoint's LoRA weights before the final gate eval.
    sd = model.state_dict()
    for k, v in best_lora_state.items():
        sd[k] = v.to(device)
    model.load_state_dict(sd, strict=True)
    final = run_full_eval(model, held_out_files, device, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"lora_state": best_lora_state, "rank": args.rank, "alpha": args.alpha or args.rank,
                "step": best_step, "lr": args.lr}, args.out)
    print(f"saved LoRA adapter weights ({sum(v.numel() for v in best_lora_state.values()):,} params): {args.out}")

    print("\n=== M3b GATE: zero-shot (old) vs LoRA fine-tune (new) ===")
    print(f"{'metric':<22}{'zero-shot':>14}{'fine-tuned':>14}{'delta':>12}")
    print(f"{'PSNR (dB, higher=better)':<22}{zero_shot['model_psnr']:>14.2f}{final['model_psnr']:>14.2f}"
          f"{final['model_psnr'] - zero_shot['model_psnr']:>+12.2f}")
    print(f"{'LPIPS (lower=better)':<22}{zero_shot['model_lpips']:>14.4f}{final['model_lpips']:>14.4f}"
          f"{final['model_lpips'] - zero_shot['model_lpips']:>+12.4f}")
    print(f"{'DISTS (lower=better)':<22}{zero_shot['model_dists']:>14.4f}{final['model_dists']:>14.4f}"
          f"{final['model_dists'] - zero_shot['model_dists']:>+12.4f}")
    print(f"{'Laplacian-var (deploy)':<22}{zero_shot['model_sharp']:>14.2f}{final['model_sharp']:>14.2f}"
          f"{final['model_sharp'] - zero_shot['model_sharp']:>+12.2f}")

    psnr_pass = final["model_psnr"] > zero_shot["model_psnr"]
    lpips_pass = final["model_lpips"] < zero_shot["model_lpips"]
    dists_pass = final["model_dists"] < zero_shot["model_dists"]
    gate_pass = psnr_pass and lpips_pass and dists_pass
    print(f"\n  PSNR beats zero-shot:  {psnr_pass}")
    print(f"  LPIPS beats zero-shot: {lpips_pass}")
    print(f"  DISTS beats zero-shot: {dists_pass}")
    if gate_pass:
        print("\n  GATE PASSED: LoRA fine-tune beats HAT-S zero-shot on PSNR AND LPIPS AND DISTS.")
    else:
        print("\n  GATE FAILED: LoRA fine-tune does NOT clear HAT-S zero-shot on all three metrics. "
              "This is a real negative result for this rank/LR/step budget, not a thing to hide.")

    if args.result_json:
        result = {
            "actual_steps": actual_steps, "best_step": best_step, "stop_reason": stop_reason,
            "train_elapsed_s": time.time() - t0, "rank": args.rank, "alpha": float(args.alpha or args.rank),
            "lr": args.lr, "batch_size": args.batch_size, "train_patch_size": args.train_patch_size,
            "zero_shot_psnr_db": float(zero_shot["model_psnr"]), "zero_shot_lpips": float(zero_shot["model_lpips"]),
            "zero_shot_dists": float(zero_shot["model_dists"]), "zero_shot_deploy_sharp": float(zero_shot["model_sharp"]),
            "final_psnr_db": float(final["model_psnr"]), "final_lpips": float(final["model_lpips"]),
            "final_dists": float(final["model_dists"]), "final_deploy_sharp": float(final["model_sharp"]),
            "psnr_pass": bool(psnr_pass), "lpips_pass": bool(lpips_pass), "dists_pass": bool(dists_pass),
            "gate_pass": bool(gate_pass),
        }
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2))
        print(f"\nresult written: {args.result_json}")


if __name__ == "__main__":
    main()
