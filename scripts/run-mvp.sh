#!/usr/bin/env bash
# FIXED_IDEA end-to-end MVP orchestrator (data -> pilot -> train both methods -> eval).
#
# Usage (repo root, GPU):
#   bash scripts/run-mvp.sh
#   STAGE=data bash scripts/run-mvp.sh          # only gen + pilot
#   STAGE=train bash scripts/run-mvp.sh         # only train (data must exist)
#   STAGE=eval bash scripts/run-mvp.sh          # only eval both concepts
#   NUM_IMAGES=50 MAX_STEPS=500 bash scripts/run-mvp.sh   # pilot mini

set -euo pipefail

STAGE=${STAGE:-"all"}
NUM_IMAGES=${NUM_IMAGES:-64}
METHODS=${METHODS:-"sdedit,face_inpaint"}
BETAS=${BETAS:-"500"}
MAX_STEPS=${MAX_STEPS:-1000}
PERSON=${PERSON:-"Barack Obama"}
DEVICE=${DEVICE:-"cuda"}

base_dir=$(pwd)
cd "$base_dir"

run_data() {
  NUM_IMAGES="$NUM_IMAGES" METHODS="$METHODS" PERSON="$PERSON" DEVICE="$DEVICE" \
    bash scripts/prepare-person.sh
}

run_train() {
  for concept in Obama_SDEdit Obama_FaceInpaint; do
    if [[ -d "datasets/person_data/$concept/unsafe" ]]; then
      CONCEPT="$concept" PERSON="$PERSON" BETAS="$BETAS" MAX_STEPS="$MAX_STEPS" \
        NUM_SAMPLES="$NUM_IMAGES" bash scripts/sd-person.sh
    else
      echo "SKIP train $concept (missing data)"
    fi
  done
}

run_eval() {
  for concept in Obama_SDEdit Obama_FaceInpaint; do
    for beta in $BETAS; do
      path="outputs/unlearn/SD-train/dpo/$beta/$concept"
      if [[ -d "$path" ]]; then
        UNLEARN_MODEL_PATH="$path" \
          OUTPUT_DIR="eval_results/${concept}_b${beta}" \
          PERSON="$PERSON" DEVICE="$DEVICE" \
          bash scripts/eval-person.sh
      else
        echo "SKIP eval $path"
      fi
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

echo "==> MVP stage=$STAGE done"
