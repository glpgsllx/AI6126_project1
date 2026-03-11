#!/usr/bin/env bash
set -euo pipefail

# Edit these variables before running.
arch="deeplab"
checkpoint="./checkpoints/deeplab/weighted_sqrt_d7c3_st20/best_model.pth"
test_dir="./data/val/images"
work_dir="./submission_work"
zip_name="./weighted_sqrt_d7c3_st20.zip"
num_classes="19"
img_size="512"
base_ch="32"
device="auto"

pred_dir="${work_dir}/masks"
zip_dir="$(dirname "${zip_name}")"
zip_file="$(basename "${zip_name}")"

rm -rf "${work_dir}"
mkdir -p "${pred_dir}"
mkdir -p "${zip_dir}"

python predict.py \
  --arch "${arch}" \
  --checkpoint "${checkpoint}" \
  --test_dir "${test_dir}" \
  --output_dir "${pred_dir}" \
  --num_classes "${num_classes}" \
  --img_size "${img_size}" \
  --base_ch "${base_ch}" \
  --device "${device}"

rm -f "${zip_name}"
(
  cd "${work_dir}"
  zip -r "../${zip_file}" masks
)

if [ "${zip_dir}" != "." ]; then
  mv "${zip_file}" "${zip_name}"
fi

echo "Created zip: ${zip_name}"
