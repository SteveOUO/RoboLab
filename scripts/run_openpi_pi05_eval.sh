#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robolab_root="$(cd "${script_dir}/.." && pwd)"
cd "${robolab_root}"

POLICY="${POLICY:-pi05}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_RUNS="${NUM_RUNS:-1}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-openpi_${POLICY}_$(date +%Y%m%d_%H%M%S)}"
VIDEO_MODE="${VIDEO_MODE:-all}"
OPENPI_ACTION_MODE="${OPENPI_ACTION_MODE:-joint_position}"
OPENPI_CONTROL_DT="${OPENPI_CONTROL_DT:-0.06666666666666667}"
HEADLESS="${HEADLESS:-1}"
TASKS="${TASKS:-BananaInBowlTask RubiksCubeAndBananaTask}"

eval_args=(
  --policy "${POLICY}"
  --remote-host "${HOST}"
  --remote-port "${PORT}"
  --num-envs "${NUM_ENVS}"
  --num-runs "${NUM_RUNS}"
  --output-folder-name "${OUTPUT_FOLDER_NAME}"
  --video-mode "${VIDEO_MODE}"
  --openpi-action-mode "${OPENPI_ACTION_MODE}"
  --openpi-control-dt "${OPENPI_CONTROL_DT}"
)

if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" ]]; then
  eval_args+=(--headless)
fi

if [[ -n "${TASKS}" ]]; then
  read -r -a task_array <<< "${TASKS}"
  eval_args+=(--task "${task_array[@]}")
fi

echo "[run_openpi_pi05_eval] POLICY=${POLICY}"
echo "[run_openpi_pi05_eval] remote=${HOST}:${PORT}"
echo "[run_openpi_pi05_eval] NUM_ENVS=${NUM_ENVS} NUM_RUNS=${NUM_RUNS} HEADLESS=${HEADLESS}"
echo "[run_openpi_pi05_eval] OPENPI_ACTION_MODE=${OPENPI_ACTION_MODE} OPENPI_CONTROL_DT=${OPENPI_CONTROL_DT}"
echo "[run_openpi_pi05_eval] OUTPUT_FOLDER_NAME=${OUTPUT_FOLDER_NAME}"
echo "[run_openpi_pi05_eval] TASKS=${TASKS}"

uv run python examples/policy/run_eval.py "${eval_args[@]}" "$@"
