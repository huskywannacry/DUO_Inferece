#!/usr/bin/env bash
# FIXED_IDEA Stage 5b: identity metrics for a trained person LoRA.
#
# Usage:
#   UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500/Obama_FaceInpaint \
#     bash scripts/eval-person.sh
#   UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500/Obama_SDEdit \
#     OUTPUT_DIR=eval_results/obama_sdedit_b500 bash scripts/eval-person.sh

set -euo pipefail

UNLEARN_MODEL_PATH=${UNLEARN_MODEL_PATH:-"outputs/unlearn/SD-train/dpo/500/Obama_FaceInpaint"}
PERSON=${PERSON:-"Barack Obama"}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/person_$(basename "$UNLEARN_MODEL_PATH")"}
REF_DIR=${REF_DIR:-"evaluation/face_recognition/reference_embeddings/obama"}
NUM_PER_PROMPT=${NUM_PER_PROMPT:-2}
DEVICE=${DEVICE:-"cuda"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/evaluation/person_metrics.py" ]]; then
  echo "Run from repo root"; exit 1
fi

mkdir -p "$REF_DIR" "$OUTPUT_DIR"

build_flag=()
if [[ -z "$(find "$REF_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | head -1)" ]]; then
  build_flag=(--build_ref_from_model)
fi

echo "==> person_metrics unlearn=$UNLEARN_MODEL_PATH"
python3 -m evaluation.person_metrics \
  --unlearn_model_path "$UNLEARN_MODEL_PATH" \
  --person "$PERSON" \
  --output_dir "$OUTPUT_DIR" \
  --ref_dir "$REF_DIR" \
  --num_per_prompt "$NUM_PER_PROMPT" \
  --device "$DEVICE" \
  --baseline \
  "${build_flag[@]}"

echo "==> $OUTPUT_DIR/metrics.json"
