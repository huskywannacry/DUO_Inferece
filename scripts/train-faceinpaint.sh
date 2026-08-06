#!/usr/bin/env bash
# Session B (Kaggle 12h): train ONLY DUO-FaceInpaint (method) for given PERSONS.
#
# Usage (repo root, after prepare-person.sh — needs FaceInpaint pair folders):
#   PERSONS="obama" bash scripts/train-faceinpaint.sh
#   PERSONS="obama,elon,trump" NUM_SAMPLES=50 BETAS=500 bash scripts/train-faceinpaint.sh
#
# Does NOT train SDEdit — use scripts/train-sdedit.sh in the other session.

set -euo pipefail

export TRAIN_METHODS="FaceInpaint"
PERSONS=${PERSONS:-"obama"}
BETAS=${BETAS:-"500"}
MAX_STEPS=${MAX_STEPS:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-50}

echo "============================================================"
echo " train-faceinpaint ONLY | PERSONS=$PERSONS | BETAS=$BETAS | steps=$MAX_STEPS"
echo "============================================================"

PERSONS="$PERSONS" \
BETAS="$BETAS" \
MAX_STEPS="$MAX_STEPS" \
NUM_SAMPLES="$NUM_SAMPLES" \
TRAIN_METHODS="FaceInpaint" \
  bash scripts/train-multi-person.sh

echo "==> FaceInpaint train done. Compare vs SDEdit LoRAs with person_metrics."
