#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7777}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_RUNS="${NUM_RUNS:-3}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-smartworld_ablation16_$(date +%Y%m%d_%H%M%S)}"
VIDEO_MODE="${VIDEO_MODE:-all}"

uv run python examples/policy/run_eval.py \
  --policy smartworld \
  --remote-host "${HOST}" \
  --remote-port "${PORT}" \
  --headless \
  --num-envs "${NUM_ENVS}" \
  --num-runs "${NUM_RUNS}" \
  --output-folder-name "${OUTPUT_FOLDER_NAME}" \
  --video-mode "${VIDEO_MODE}" \
  "$@" \
  --task \
    BananaInBowlTask \
    RubiksCubeTask \
    PickDrillTask \
    PickUpGreenObjectTask \
    SauceBottlesCrateTask \
    MustardInRightBinTask \
    RubiksCubeBehindBowlTask \
    WhiteMugInCenterOfTableTask \
    RubiksCubeAndBananaTask \
    BananaThenRubiksCubeTask \
    RedItemsInBinTask \
    BananasInBinThreeTotalTask \
    SpoonInMugTask \
    ReorientRedMugTask \
    Stack3RubiksCubeTask \
    ToolsPickingHammerTask
