#!/usr/bin/env bash
# Evaluate Obama unlearned LoRA with ArcFace DSR / FPR / cross-ID.
#
# Usage (repo root):
#   bash scripts/eval-obama.sh
#   UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500/Obama bash scripts/eval-obama.sh

set -euo pipefail

UNLEARN_MODEL_PATH=${UNLEARN_MODEL_PATH:-"outputs/unlearn/SD-train/dpo/500/Obama"}
PERSON=${PERSON:-"Barack Obama"}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/obama_b500"}
REF_DIR=${REF_DIR:-"eval_results/refs/obama"}
NUM_PER_PROMPT=${NUM_PER_PROMPT:-2}
DEVICE=${DEVICE:-"cuda"}
BUILD_REF=${BUILD_REF:-"1"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/evaluation/person_metrics.py" && -f "$base_dir/../evaluation/person_metrics.py" ]]; then
  base_dir=$(cd "$base_dir/.." && pwd)
fi
cd "$base_dir"

mkdir -p "$REF_DIR" "$OUTPUT_DIR"

ref_flag=()
if [[ "$BUILD_REF" == "1" ]]; then
  if [[ -z "$(find "$REF_DIR" -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | head -1)" ]]; then
    echo "==> Building generated reference faces (prefer real public-domain photos later)"
    ref_flag=(--build_ref_from_model)
  fi
fi

echo "==> person_metrics"
echo "    unlearn : $UNLEARN_MODEL_PATH"
echo "    output  : $OUTPUT_DIR"

python3 -m evaluation.person_metrics \
  --unlearn_model_path "$UNLEARN_MODEL_PATH" \
  --person "$PERSON" \
  --output_dir "$OUTPUT_DIR" \
  --ref_dir "$REF_DIR" \
  --num_per_prompt "$NUM_PER_PROMPT" \
  --device "$DEVICE" \
  --baseline \
  "${ref_flag[@]}"

echo "==> Done. See $OUTPUT_DIR/metrics.json"
