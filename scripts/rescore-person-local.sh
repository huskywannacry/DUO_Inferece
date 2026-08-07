#!/usr/bin/env bash
# Re-score an already-downloaded eval folder (no GPU gen required if insightface OK).
# Uses thr=1.0 by default and prints mean_distance comparison.
#
# Usage:
#   bash scripts/rescore-person-local.sh /home/kientt44/Downloads/all_obama_eval/Obama_SDEdit_b500
#   THRESHOLD=1.0 bash scripts/rescore-person-local.sh path/to/Obama_FaceInpaint_b500

set -euo pipefail

OUT=${1:-}
THRESHOLD=${THRESHOLD:-"1.0"}
PERSON=${PERSON:-"Barack Obama"}
REF_DIR=${REF_DIR:-"evaluation/face_recognition/reference_embeddings/obama"}

if [[ -z "$OUT" || ! -d "$OUT" ]]; then
  echo "Usage: bash scripts/rescore-person-local.sh <eval_output_dir>"
  exit 1
fi

base_dir=$(pwd)
if [[ ! -f "$base_dir/evaluation/person_metrics.py" ]]; then
  echo "Run from repo root"; exit 1
fi

# Prefer ref images; else ref_embedding.npy inside OUT
extra_ref=()
if [[ ! -d "$REF_DIR" ]] || [[ -z "$(find "$REF_DIR" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) 2>/dev/null | head -1)" ]]; then
  if [[ -f "$OUT/ref_embedding.npy" ]]; then
    echo "Using $OUT/ref_embedding.npy (no ref images in $REF_DIR)"
  else
    echo "WARNING: no ref images and no ref_embedding.npy — will fail unless you point REF_DIR"
  fi
fi

python3 -m evaluation.person_metrics \
  --person "$PERSON" \
  --output_dir "$OUT" \
  --ref_dir "$REF_DIR" \
  --threshold "$THRESHOLD" \
  --score_existing \
  --baseline

echo "Updated $OUT/metrics.json"
