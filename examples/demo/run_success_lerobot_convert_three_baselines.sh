#!/usr/bin/env bash
set -euo pipefail
cd /mnt/home/zhengyixin/starVLA/examples/Droid/RoboLab

OUT_ROOT=/mnt/project/world_model/Robolab_dataset
mkdir -p "${OUT_ROOT}"

ROOTS=(
  output/robolab120_pi05_jointpos_specific_10ep_20260603
  output/robolab120_dreamzero_specific_10ep_torchattn_20260603
  output/robolab120_cosmos3_official_specific_10ep_20260603
)
NAMES=(
  pi05_jointpos
  dreamzero
  cosmos3_official
)

for i in "${!ROOTS[@]}"; do
  root="${ROOTS[$i]}"
  name="${NAMES[$i]}"
  out="${OUT_ROOT}/${name}"
  if [[ -f "${out}/meta/info.json" && -f "${out}/meta/stats.json" ]]; then
    echo "==== skip ${out}; info.json and stats.json already exist ===="
    continue
  fi
  echo "==== convert ${root} -> ${out} ===="
  uv run --with pyarrow --with h5py python scripts/convert_to_lerobot.py \
    --input "${root}" \
    --output "${out}" \
    --robot-type droid \
    --fps 15 \
    --success-only \
    --require-complete-videos \
    --required-cameras over_shoulder_left_camera over_shoulder_right_camera wrist_cam
done
