#!/usr/bin/env bash
# Smoke run: two tasks, one seed, through all three harnesses.
#
# Proves the wiring works without a multi-day run. Use this first after changing
# a model, a harness, or this repository.
set -euo pipefail

: "${EVAL_MODEL_NAME:=frognano-4b-2609}"
: "${EVAL_MODEL_BASE_URL:=http://127.0.0.1:1234/v1}"
export EVAL_MODEL_NAME EVAL_MODEL_BASE_URL

echo "model: $EVAL_MODEL_NAME @ $EVAL_MODEL_BASE_URL"
exec "$(dirname "$0")/run-matrix.sh" \
  --num-tasks 2 --seeds 1 --max-steps 10 --max-time 900 "$@"