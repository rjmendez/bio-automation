#!/bin/bash
# M3b: LoRA fine-tune of HAT-S on AWS. Same guardrails as user-data.sh: the trap
# pushes logs+checkpoint to S3 and shuts the instance down (terminating it, since
# launched with instance-initiated-shutdown-behavior terminate) on ANY exit path.
set -uo pipefail
BUCKET="bio-automation-cloud-399511413068"
WORK=/opt/bio-sr-train
LOG="$WORK/train.log"
mkdir -p "$WORK/code/hat_vendor" "$WORK/data" "$WORK/weights" "$WORK/out" "$WORK/checkpoint"

finish() {
  aws s3 cp "$LOG" "s3://$BUCKET/results/lora-train.log" --only-show-errors || true
  aws s3 sync "$WORK/out" "s3://$BUCKET/results/lora/" --only-show-errors || true
  aws s3 sync "$WORK/checkpoint" "s3://$BUCKET/results/lora-checkpoint/" --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

{
  echo "=== $(date -u) staging code+data+weights ==="
  aws s3 sync "s3://$BUCKET/code/" "$WORK/code/" --only-show-errors
  aws s3 sync "s3://$BUCKET/data/" "$WORK/data/" --only-show-errors
  aws s3 cp "s3://$BUCKET/weights/HAT-S_SRx2.pth" "$WORK/weights/HAT-S_SRx2.pth" --only-show-errors

  echo "=== $(date -u) installing torch (cu128) in a clean venv ==="
  python3 -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
  pip install --quiet numpy Pillow einops
  # piq (LPIPS/DISTS) silently downgrades torch to a CPU wheel unless --no-deps
  # (measured this session, same failure class as the cuDNN bug it's not): keep
  # it from touching torch, pin torchvision to the matching cu128 build instead.
  pip install --quiet --no-deps piq
  pip install --quiet --no-deps torchvision --index-url https://download.pytorch.org/whl/cu128

  echo "=== $(date -u) GPU check ==="
  nvidia-smi
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

  echo "=== $(date -u) training ==="
  cd "$WORK/code"
  # batch=4, patch=128 (NOT the 256 eval-matching size originally planned): local
  # VRAM probes to estimate headroom at patch=256 were unreliable (this desktop's
  # GPUs stayed contended by other processes even after explicit unload attempts,
  # "0 bytes free" readings that didn't match nvidia-smi moments earlier), and a
  # rough extrapolation from the one clean local measurement (bs=2/patch=128
  # ~5.6GB) suggested patch=256 could approach or exceed the T4's 16GB even at
  # low batch. Patch=128 is the exact configuration the local smoke test already
  # proved safe and effective (+0.16dB gate pass); batch=4 is a modest, not
  # aggressive, increase leveraging the T4 being fully dedicated (no desktop
  # contention) rather than assuming unverified headroom.
  timeout 7200 python3 train_lora_hat.py \
    --source-dir "$WORK/data" --weights "$WORK/weights/HAT-S_SRx2.pth" \
    --out "$WORK/out/hat_s_lora.pt" --checkpoint-dir "$WORK/checkpoint" \
    --result-json "$WORK/out/result.json" \
    --time-budget-s 5400 --batch-size 4 --train-patch-size 128 \
    --lr 1.5e-4 --rank 8 --held-out 80 --eval-every 250 --patience 6

  echo "=== $(date -u) done ==="
} > "$LOG" 2>&1
