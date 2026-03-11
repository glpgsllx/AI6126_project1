#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_train.sh
#   ARCH=deeplab EPOCHS=120 BATCH_SIZE=10 ./run_train.sh

ARCH="${ARCH:-attention_unet}"         # attention_unet | deeplab | segnet
DATA_DIR="${DATA_DIR:-./data}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
NUM_CLASSES="${NUM_CLASSES:-19}"
IMG_SIZE="${IMG_SIZE:-512}"
BASE_CH="${BASE_CH:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
COMPILE_MODE="${COMPILE_MODE:-default}" # default | reduce-overhead | max-autotune
LOG_INTERVAL="${LOG_INTERVAL:-20}"

echo "[train] arch=${ARCH} epochs=${EPOCHS} batch_size=${BATCH_SIZE} img_size=${IMG_SIZE}"

python3 -u train.py \
  --arch "${ARCH}" \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --num_classes "${NUM_CLASSES}" \
  --img_size "${IMG_SIZE}" \
  --base_ch "${BASE_CH}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --persistent_workers \
  --no_compile \
  --compile_mode "${COMPILE_MODE}" \
  --matmul_precision high \
  --log_interval "${LOG_INTERVAL}"
