#!/usr/bin/env bash
# Run the same tasks through every harness and print one comparison table.
#
#   scripts/run-matrix.sh                       # smoke slice, all harnesses
#   scripts/run-matrix.sh --dataset swebench_verified
#   scripts/run-matrix.sh --config configs/dataset/swebench-verified.yaml \
#       --num-tasks 5 --seeds 3 --max-steps 150 --max-time 10800
#
# The model comes from the environment, so the same script re-runs any other
# served checkpoint:
#
#   EVAL_MODEL_NAME=qwen3.8-27b-splash scripts/run-matrix.sh
set -euo pipefail

DATASET=""
CONFIG=""
NUM_TASKS=""
SEEDS=""
MAX_STEPS=""
MAX_TIME=""
HARNESSES=(leaf opencode pi)
ROOT="${EVAL_OUTPUT_ROOT:-$(pwd)/eval-results}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --dataset-config) CONFIG="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --num-tasks) NUM_TASKS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-time) MAX_TIME="$2"; shift 2 ;;
    --harnesses) IFS=, read -r -a HARNESSES <<< "$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# `--dataset` accepts the CLI name or the config file stem, so both
# `terminal_bench_2_verified` and `terminal-bench-2-verified` work.
if [[ -z "$CONFIG" ]]; then
  stem="${DATASET:-swebench_verified}"
  stem="${stem//_/-}"
  CONFIG="configs/dataset/${stem}.yaml"
fi

overrides=()
[[ -n "$NUM_TASKS" ]] && overrides+=("--set" "num_tasks=$NUM_TASKS")
[[ -n "$SEEDS" ]] && overrides+=("--set" "seeds_per_task=$SEEDS")
[[ -n "$MAX_STEPS" ]] && overrides+=("--set" "max_steps=$MAX_STEPS")
[[ -n "$MAX_TIME" ]] && overrides+=("--set" "max_total_time_sec=$MAX_TIME")

echo "== preflight =="
cae doctor --config "$CONFIG"

for harness in "${HARNESSES[@]}"; do
  out="$ROOT/${harness}"
  echo
  echo "== $harness -> $out =="
  cae run --config "$CONFIG" --harness "$harness" \
    --set "output_dir=$out" "${overrides[@]}" || {
      echo "  $harness failed; continuing with the rest of the matrix" >&2
      continue
    }
done

echo
echo "== comparison =="
cae compare "$ROOT"