#!/bin/bash
# bio-automation-cloud training run. Runs once as root via cloud-init. The trap
# guarantees logs+checkpoint are pushed to S3 and the instance shuts down (and,
# because the instance was launched with instance-initiated-shutdown-behavior
# terminate, is torn down) on ANY exit path -- success, training crash, or the
# 2-hour timeout cap -- so nothing can be left running and billing by accident.
set -uo pipefail
BUCKET="bio-automation-cloud-399511413068"
WORK=/opt/bio-sr-train
LOG="$WORK/train.log"
mkdir -p "$WORK/code" "$WORK/patches" "$WORK/held-out" "$WORK/out"

finish() {
  aws s3 cp "$LOG" "s3://$BUCKET/results/train.log" --only-show-errors || true
  aws s3 sync "$WORK/out" "s3://$BUCKET/results/" --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

{
  echo "=== $(date -u) staging code+data ==="
  aws s3 sync "s3://$BUCKET/code/" "$WORK/code/" --only-show-errors
  aws s3 sync "s3://$BUCKET/patches/" "$WORK/patches/" --only-show-errors
  aws s3 sync "s3://$BUCKET/held-out/" "$WORK/held-out/" --only-show-errors

  echo "=== $(date -u) installing torch (cu128) in a clean venv ==="
  # The system pip (RPM-managed on Amazon Linux) refuses to self-upgrade, which
  # silently left an ancient pip in place that failed to pull in torch's cudnn
  # companion package. A fresh venv has its own unmanaged pip, same fix as the
  # local F:\venvs\bio-sr-train CPU-wheel issue.
  python3 -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128
  pip install --quiet numpy Pillow
  # piq (for LPIPS/DISTS) declares its own torch/torchvision version constraints;
  # installing it normally silently downgrades torch to a CPU wheel to satisfy
  # them (measured today, locally: same CPU-wheel failure mode as the cuDNN bug
  # above, from a different cause). --no-deps keeps it from ever touching torch;
  # torchvision is pinned to the matching cu128 build explicitly instead.
  pip install --quiet --no-deps piq
  pip install --quiet --no-deps torchvision --index-url https://download.pytorch.org/whl/cu128

  echo "=== $(date -u) GPU check ==="
  nvidia-smi
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

  echo "=== $(date -u) training ==="
  cd "$WORK/code"
  timeout 7200 python3 aws_train.py \
    --patch-array "$WORK/patches/patches_train.npy" --held-out-dir "$WORK/held-out" \
    --time-budget-s 5400 --batch-size 32 --workers 3 --checkpoint-every 250 --out-dir "$WORK/out"

  echo "=== $(date -u) done ==="
} > "$LOG" 2>&1
