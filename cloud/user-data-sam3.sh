#!/bin/bash
# M-sam3: fine-tune SAM3 on the 3 weak categories (Fungi/Plantae/Protozoa) found this
# session. Same guardrails as the other cloud/user-data*.sh scripts in this worktree: the
# trap pushes logs+checkpoints to S3 and shuts the instance down (terminating it, since
# launched with instance-initiated-shutdown-behavior terminate) on ANY exit path.
set -uo pipefail
BUCKET="bio-automation-cloud-399511413068"
WORK=/opt/sam3-train
LOG="$WORK/train.log"
mkdir -p "$WORK/code" "$WORK/data" "$WORK/weights" "$WORK/out"

finish() {
  aws s3 cp "$LOG" "s3://$BUCKET/results/sam3-train.log" --only-show-errors || true
  aws s3 sync "$WORK/out" "s3://$BUCKET/results/sam3/" --only-show-errors || true
  shutdown -h now
}
trap finish EXIT

{
  echo "=== $(date -u) staging code+data+weights ==="
  aws s3 sync "s3://$BUCKET/code-sam3/" "$WORK/code/" --only-show-errors
  aws s3 sync "s3://$BUCKET/data-sam3-coco/" "$WORK/data/coco_dataset/" --only-show-errors
  aws s3 cp "s3://$BUCKET/weights-sam3/sam3.pt" "$WORK/weights/sam3.pt" --only-show-errors

  # First attempt used the plain al2023-ami-kernel-default AMI, which has NO NVIDIA driver
  # at all -- measured directly: "nvidia-smi: command not found", torch.cuda.is_available()
  # False. Installing the driver manually on a running instance needs a reboot before
  # nvidia-smi works (confirmed against AWS's own AL2023 NVIDIA driver docs), which doesn't
  # fit this single-pass script -- using "Deep Learning Base OSS Nvidia Driver GPU AMI
  # (Amazon Linux 2023)" instead (ami-096c37309634b0873, drivers pre-loaded, no reboot
  # needed) rather than fight that. See the AMI choice in the launch command, not here.
  echo "=== $(date -u) installing Python 3.12 + deps ==="
  dnf install -y python3.12 python3.12-pip python3.12-devel >/dev/null 2>&1 || \
    (amazon-linux-extras install -y python3.12 2>&1 | tail -5)
  python3.12 -m venv "$WORK/venv"
  source "$WORK/venv/bin/activate"
  pip install --quiet --upgrade pip

  # torch 2.11+cu128: same version proven to work this session (also fixes an unrelated
  # Windows-only gloo bug that doesn't apply on Linux, but no reason to pin older).
  pip install --quiet torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  pip install --quiet einops triton psutil

  echo "=== $(date -u) installing sam3 package ==="
  cd "$WORK/code"
  # [train,dev] alone was missing psutil (measured: ModuleNotFoundError importing
  # sam3.model.sam3_video_predictor via model_builder.py's eager import chain --
  # same failure class as einops/triton on Windows, a real gap in sam3's own extras
  # declarations). [train,dev,notebooks] is the exact combo already verified to import
  # cleanly end-to-end on Windows this session; use it here too rather than discover
  # more missing deps one AWS round-trip at a time.
  pip install --quiet -e ".[train,dev,notebooks]"
  pip install --quiet psutil  # belt and suspenders in case notebooks' own deps shift

  echo "=== $(date -u) GPU check ==="
  nvidia-smi
  python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" || \
    { echo "FATAL: CUDA still unavailable after driver install -- aborting (GPU-only standing rule)"; exit 1; }

  echo "=== $(date -u) training ==="
  # Patches already baked into the synced code tree (vitdet.py Mlp.forward grad-fallback,
  # train.py MASTER_ADDR -- the latter harmless on Linux, kept for parity). Config:
  # configs/custom/weak_categories_aws.yaml (nccl backend, 8 epochs x 200 steps, real
  # checkpoint saving every 2 epochs). 3-hour wall-clock safety cap.
  mkdir -p "$WORK/out"
  timeout 10800 python3 sam3/train/train.py \
    -c configs/custom/weak_categories_aws.yaml \
    --use-cluster 0 --num-gpus 1

  echo "=== $(date -u) done ==="
} > "$LOG" 2>&1
