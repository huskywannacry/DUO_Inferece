#!/usr/bin/env bash
# DUO paper §4.1 Ring-A-Bell evaluation (pre-trained inverse prompts + DSR).
#
# Nudity (paper: 95 prompts from HF Chia15/RingABell-Nudity):
#   RING_PROMPTS=/path/to/nudity.csv \
#   UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500/Nudity \
#   bash scripts/eval-ring-a-bell.sh
#
# Violence (paper: 250 prompts shipped in Ring-A-Bell clone):
#   EXP_TYPE=violence \
#   UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500 \
#   bash scripts/eval-ring-a-bell.sh

set -euo pipefail

EXP_TYPE=${EXP_TYPE:-"nudity"}
UNLEARN_MODEL_PATH=${UNLEARN_MODEL_PATH:-"outputs/unlearn/SD-train/dpo/500/Nudity"}
RING_REPO=${RING_REPO:-"$HOME/Code/Ring-A-Bell"}
RING_PROMPTS=${RING_PROMPTS:-""}   # required for nudity
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/ring_a_bell_${EXP_TYPE}"}
DEVICE=${DEVICE:-"cuda"}
NUM_PROMPTS=${NUM_PROMPTS:-""}     # empty = paper default 95/250

base_dir=$(pwd)
if [[ ! -f "$base_dir/evaluation/ring_a_bell.py" ]]; then
  echo "Run from DUO repo root"; exit 1
fi

extra=()
if [[ -n "$RING_PROMPTS" ]]; then
  extra+=(--ring_a_bell_prompts "$RING_PROMPTS")
fi
if [[ -n "$NUM_PROMPTS" ]]; then
  extra+=(--num_prompts "$NUM_PROMPTS")
fi

echo "==> Ring-A-Bell paper protocol exp=$EXP_TYPE"
python3 -m evaluation.ring_a_bell \
  --unlearn_model_path "$UNLEARN_MODEL_PATH" \
  --exp_type "$EXP_TYPE" \
  --ring_a_bell_repo "$RING_REPO" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --run_dsr \
  "${extra[@]}"

echo "==> Done. See $OUTPUT_DIR"
