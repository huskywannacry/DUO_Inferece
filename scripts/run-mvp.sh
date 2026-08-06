#!/usr/bin/env bash
# FIXED_IDEA orchestrator — supports one or many people.
#
# Usage:
#   bash scripts/run-mvp.sh
#   PERSONS="obama" STAGE=data bash scripts/run-mvp.sh
#   PERSONS="obama,elon,trump" STAGE=train NUM_IMAGES=50 BETAS=500 bash scripts/run-mvp.sh

set -euo pipefail

STAGE=${STAGE:-"all"}
NUM_IMAGES=${NUM_IMAGES:-50}
METHODS=${METHODS:-"sdedit,face_inpaint"}
BETAS=${BETAS:-"500"}
MAX_STEPS=${MAX_STEPS:-1000}
PERSONS=${PERSONS:-"obama"}   # default 1 person for 12h; multi: obama,elon,trump
DEVICE=${DEVICE:-"cuda"}

base_dir=$(pwd)
cd "$base_dir"

run_data() {
  PERSONS="$PERSONS" NUM_IMAGES="$NUM_IMAGES" METHODS="$METHODS" DEVICE="$DEVICE" \
    bash scripts/prepare-multi-person.sh
}

run_train() {
  PERSONS="$PERSONS" BETAS="$BETAS" MAX_STEPS="$MAX_STEPS" NUM_SAMPLES="$NUM_IMAGES" \
    bash scripts/train-multi-person.sh
}

run_eval() {
  declare -A PERSON_NAME PREFIX_MAP
  PERSON_NAME[obama]="Barack Obama"; PREFIX_MAP[obama]="Obama"
  PERSON_NAME[elon]="Elon Musk";     PREFIX_MAP[elon]="Musk"
  PERSON_NAME[trump]="Donald Trump"; PREFIX_MAP[trump]="Trump"

  IFS=',' read -ra KEYS <<< "$PERSONS"
  for key in "${KEYS[@]}"; do
    key=$(echo "$key" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
    prefix="${PREFIX_MAP[$key]:-}"
    person="${PERSON_NAME[$key]:-}"
    [[ -z "$prefix" ]] && continue
    for suffix in SDEdit FaceInpaint; do
      for beta in $BETAS; do
        path="outputs/unlearn/SD-train/dpo/$beta/${prefix}_${suffix}"
        if [[ -d "$path" ]]; then
          UNLEARN_MODEL_PATH="$path" \
            OUTPUT_DIR="eval_results/${prefix}_${suffix}_b${beta}" \
            PERSON="$person" DEVICE="$DEVICE" \
            bash scripts/eval-person.sh
        else
          echo "SKIP eval $path"
        fi
      done
    done
  done
}

case "$STAGE" in
  data) run_data ;;
  train) run_train ;;
  eval) run_eval ;;
  all)
    run_data
    run_train
    run_eval
    ;;
  *)
    echo "STAGE must be all|data|train|eval"; exit 1
    ;;
esac

echo "==> MVP stage=$STAGE persons=$PERSONS done"
