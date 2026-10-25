#!/bin/bash
# M3b follow-up: LoRA fine-tune HAT-S 4x on AWS, same recipe as the proven 2x run
# (cloud/user-data-lora.sh) -- train_lora_hat.py and degrade.py are now scale-generic
# (verified: HAT-S x2/x4 share identical architecture hyperparameters per
# XPixelGroup/HAT's own options/test/HAT-S_SRx{2,4}.yml, only `upscale` differs).
# Same guardrails: the trap pushes logs+checkpoint to S3 and shuts the instance down
# (terminating it, launched with instance-initiated-shutdown-behavior terminate) on
# ANY exit path.
set -uo pipefail
BUCKET="bio-automation-cloud-399511413068"
WORK=/opt/bio-sr-train
LOG="$WORK/train.log"
mkdir -p "$WORK/code/hat_vendor" "$WORK/data" "$WORK/weights" "$WORK/out" "$WORK/checkpoint"

finish() {
  aws s3 cp "$LOG" "s3://$BUCKET/results/lora-x4-train.log" --only-show-errors || true
  aws s3 sync "$WORK/out" "s3://$BUCKET/results/lora-x4/" --only-show-errors || true
  aws s3 sync "$WORK/checkpoint" "s3://$BUCKET/results/lora-x4-checkpoint/" --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

{
  echo "=== $(date -u) staging code+data+weights ==="
  aws s3 sync "s3://$BUCKET/code/" "$WORK/code/" --only-show-errors
  aws s3 sync "s3://$BUCKET/data/" "$WORK/data/" --only-show-errors
  aws s3 cp "s3://$BUCKET/weights/HAT-S_SRx4.pth" "$WORK/weights/HAT-S_SRx4.pth" --only-show-errors

  echo "=== $(date -u) installing torch (cu128) in a clean venv ==="
  python3 -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
  pip install --quiet numpy Pillow einops
  # piq (LPIPS/DISTS) silently downgrades torch to a CPU wheel unless --no-deps
  # (measured in the 2x run, same failure class as the cuDNN bug it's not): keep
  # it from touching torch, pin torchvision to the matching cu128 build instead.
  pip install --quiet --no-deps piq
  pip install --quiet --no-deps torchvision --index-url https://download.pytorch.org/whl/cu128

  echo "=== $(date -u) GPU check ==="
  nvidia-smi
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

  echo "=== $(date -u) training ==="
  cd "$WORK/code"
  # Same batch/patch as the proven 2x run (bs=4, patch=128 on a dedicated T4) --
  # not re-probing VRAM headroom for x4: HAT-S's architecture (param count,
  # activation shapes through the backbone) is IDENTICAL between x2/x4, only the
  # final pixelshuffle upsampling stage differs, which is a small fraction of the
  # model's total memory footprint.
  timeout 7200 python3 train_lora_hat.py \
    --source-dir "$WORK/data" --weights "$WORK/weights/HAT-S_SRx4.pth" --scale 4 \
    --out "$WORK/out/hat_s_lora_x4.pt" --checkpoint-dir "$WORK/checkpoint" \
    --result-json "$WORK/out/result.json" \
    --time-budget-s 5400 --batch-size 4 --train-patch-size 128 \
    --lr 1.5e-4 --rank 8 --held-out 80 --eval-every 250 --patience 6 \
    --cuda-name-substr "T4"

  echo "=== $(date -u) done ==="
} > "$LOG" 2>&1
