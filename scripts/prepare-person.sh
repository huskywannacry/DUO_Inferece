#!/usr/bin/env bash
# FIXED_IDEA Stages 1–3: generate person pairs + pilot report.
#
# Usage (repo root):
#   bash scripts/prepare-person.sh
#   NUM_IMAGES=50 METHODS=all bash scripts/prepare-person.sh
#   NUM_IMAGES=128 METHODS=sdedit,face_inpaint bash scripts/prepare-person.sh

set -euo pipefail

PERSON=${PERSON:-"Barack Obama"}
SLUG=${SLUG:-"obama"}
NUM_IMAGES=${NUM_IMAGES:-64}
METHODS=${METHODS:-"sdedit,face_inpaint"}
DEVICE=${DEVICE:-"cuda"}
NO_VERIFY=${NO_VERIFY:-"0"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/datasets/person_data/generate_person_data.py" ]]; then
  echo "Run from repo root"; exit 1
fi

verify_flag=()
if [[ "$NO_VERIFY" == "1" ]]; then
  verify_flag=(--no_verify)
fi

echo "==> generate_person_data person=$PERSON N=$NUM_IMAGES methods=$METHODS"
python3 datasets/person_data/generate_person_data.py \
  --person "$PERSON" \
  --slug "$SLUG" \
  --num_images "$NUM_IMAGES" \
  --methods "$METHODS" \
  --device "$DEVICE" \
  --concept_prefix "Obama" \
  "${verify_flag[@]}"

echo "==> pilot_pair_compare"
python3 -m evaluation.pilot_pair_compare \
  --slug_dir "datasets/person_data/$SLUG"

echo "==> Next train:"
echo "  CONCEPT=Obama_SDEdit bash scripts/sd-person.sh"
echo "  CONCEPT=Obama_FaceInpaint bash scripts/sd-person.sh"
