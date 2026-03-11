#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Attention U-Net:
#     ./run_train.sh
#   DeepLab:
#     ARCH=deeplab ./run_train.sh
#   SegNet:
#     ARCH=segnet ./run_train.sh
#   With local validation split:
#     VAL_SPLIT=0.15 ./run_train.sh
#   Save a separate experiment:
#     ARCH=deeplab EXP_NAME=lr5e4 LR=5e-4 ./run_train.sh
#   Override defaults:
#     ARCH=deeplab EPOCHS=120 BATCH_SIZE=12 NUM_WORKERS=2 ./run_train.sh

ARCH="${ARCH:-attention_unet}"         # attention_unet | deeplab | segnet
DATA_DIR="${DATA_DIR:-./data}"
SAVE_DIR="${SAVE_DIR:-./checkpoints}"
NUM_CLASSES="${NUM_CLASSES:-19}"
COMPILE_MODE="${COMPILE_MODE:-default}" # default | reduce-overhead | max-autotune
LOG_INTERVAL="${LOG_INTERVAL:-20}"
VAL_SPLIT="${VAL_SPLIT:-0.0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EXP_NAME="${EXP_NAME:-}"
WEIGHTED_CE="${WEIGHTED_CE:-none}"
WEIGHTED_CE_TRANSFORM="${WEIGHTED_CE_TRANSFORM:-none}"
WEIGHTED_CE_CLIP_MAX="${WEIGHTED_CE_CLIP_MAX:-0}"
DICE_WEIGHT="${DICE_WEIGHT:-0.5}"
CE_WEIGHT="${CE_WEIGHT:-0.5}"

case "${ARCH}" in
  attention_unet)
    IMG_SIZE="${IMG_SIZE:-512}"
    BASE_CH="${BASE_CH:-15}"
    BATCH_SIZE="${BATCH_SIZE:-24}"
    EPOCHS="${EPOCHS:-100}"
    LR="${LR:-1e-3}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    ;;
  deeplab)
    IMG_SIZE="${IMG_SIZE:-512}"
    BASE_CH="${BASE_CH:-32}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    EPOCHS="${EPOCHS:-100}"
    LR="${LR:-1e-3}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    ;;
  segnet)
    IMG_SIZE="${IMG_SIZE:-512}"
    BASE_CH="${BASE_CH:-32}"
    BATCH_SIZE="${BATCH_SIZE:-24}"
    EPOCHS="${EPOCHS:-100}"
    LR="${LR:-1e-3}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    ;;
  *)
    echo "Unknown ARCH: ${ARCH}" >&2
    exit 1
    ;;
esac

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
  --log_interval "${LOG_INTERVAL}" \
  --val_split "${VAL_SPLIT}" \
  --split_seed "${SPLIT_SEED}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --exp_name "${EXP_NAME}" \
  --weighted_ce "${WEIGHTED_CE}" \
  --weighted_ce_transform "${WEIGHTED_CE_TRANSFORM}" \
  --weighted_ce_clip_max "${WEIGHTED_CE_CLIP_MAX}" \
  --dice_weight "${DICE_WEIGHT}" \
  --ce_weight "${CE_WEIGHT}"
