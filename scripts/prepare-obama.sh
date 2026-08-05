#!/usr/bin/env bash
# Step 1: generate Obama paired data for DUO (SDEdit baseline by default).
#
# Usage (repo root):
#   bash scripts/prepare-obama.sh
#   METHOD=face_inpaint NUM_IMAGES=64 bash scripts/prepare-obama.sh
#   METHOD=sdedit CONCEPT=Obama_SDEdit bash scripts/prepare-obama.sh
#   METHOD=face_inpaint CONCEPT=Obama_FaceEdit bash scripts/prepare-obama.sh

set -euo pipefail

METHOD=${METHOD:-"sdedit"}
CONCEPT=${CONCEPT:-"Obama"}
PERSON=${PERSON:-"Barack Obama"}
NUM_IMAGES=${NUM_IMAGES:-64}
SAVE_DIR=${SAVE_DIR:-"datasets/SD"}
VERIFY=${VERIFY:-"0"}
DEVICE=${DEVICE:-"cuda"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/datasets/SD/generate_person_data.py" && -f "$base_dir/../datasets/SD/generate_person_data.py" ]]; then
  base_dir=$(cd "$base_dir/.." && pwd)
fi
cd "$base_dir"

verify_flag=()
if [[ "$VERIFY" == "1" ]]; then
  verify_flag=(--verify_arcface)
fi

echo "==> Generating person pairs"
echo "    concept=$CONCEPT method=$METHOD N=$NUM_IMAGES person=$PERSON"

python3 datasets/SD/generate_person_data.py \
  --concept "$CONCEPT" \
  --person "$PERSON" \
  --method "$METHOD" \
  --num_images "$NUM_IMAGES" \
  --save_dir "$SAVE_DIR" \
  --device "$DEVICE" \
  "${verify_flag[@]}"

echo "==> Next:"
echo "    bash scripts/sd-obama.sh"
echo "    # or CONCEPT=$CONCEPT bash scripts/sd-obama.sh"
