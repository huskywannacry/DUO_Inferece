#!/usr/bin/env bash
# Session A (Kaggle 12h): train ONLY DUO-SDEdit baselines for given PERSONS.
#
# Usage (repo root, after prepare-person.sh):
#   PERSONS="obama" bash scripts/train-sdedit.sh
#   PERSONS="obama,elon,trump" NUM_SAMPLES=50 BETAS=500 bash scripts/train-sdedit.sh
#
# Does NOT train FaceInpaint — use scripts/train-faceinpaint.sh in another session.

set -euo pipefail

export TRAIN_METHODS="SDEdit"
PERSONS=${PERSONS:-"obama"}
BETAS=${BETAS:-"500"}
MAX_STEPS=${MAX_STEPS:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-50}

echo "============================================================"
echo " train-sdedit ONLY | PERSONS=$PERSONS | BETAS=$BETAS | steps=$MAX_STEPS"
echo "============================================================"

PERSONS="$PERSONS" \
BETAS="$BETAS" \
MAX_STEPS="$MAX_STEPS" \
NUM_SAMPLES="$NUM_SAMPLES" \
TRAIN_METHODS="SDEdit" \
  bash scripts/train-multi-person.sh

echo "==> SDEdit train done. Next session: bash scripts/train-faceinpaint.sh"
