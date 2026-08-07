#!/usr/bin/env bash
# Stronger person unlearn train (submit-oriented recipe).
# Same DUO loss as sd-person.sh; higher N / steps; both β; both methods.
#
# Prerequisites: person pairs already generated, e.g.
#   PERSONS=obama NUM_IMAGES=128 METHODS=sdedit,face_inpaint bash scripts/prepare-person.sh
#
# Usage (repo root):
#   bash scripts/train-person-strong.sh
#   PERSONS=obama MAX_STEPS=1500 NUM_SAMPLES=128 BETAS="250 500" bash scripts/train-person-strong.sh
#   # only FaceInpaint:
#   TRAIN_METHODS=FaceInpaint bash scripts/train-person-strong.sh

set -euo pipefail

PERSONS=${PERSONS:-"obama"}
NUM_SAMPLES=${NUM_SAMPLES:-128}
MAX_STEPS=${MAX_STEPS:-1500}
BETAS=${BETAS:-"250 500"}
TRAIN_METHODS=${TRAIN_METHODS:-"SDEdit,FaceInpaint"}
RANK=${RANK:-32}

echo "============================================================"
echo " train-person-strong"
echo "   PERSONS=$PERSONS  N=$NUM_SAMPLES  steps=$MAX_STEPS"
echo "   BETAS=$BETAS  METHODS=$TRAIN_METHODS  rank=$RANK"
echo "============================================================"
echo " NOTE: needs datasets/person_data/{Prefix}_SDEdit|FaceInpaint/{unsafe,safe}"
echo "       Prefer N_pairs >= NUM_SAMPLES (regen with NUM_IMAGES=$NUM_SAMPLES if short)."
echo "============================================================"

PERSONS="$PERSONS" \
NUM_SAMPLES="$NUM_SAMPLES" \
MAX_STEPS="$MAX_STEPS" \
BETAS="$BETAS" \
TRAIN_METHODS="$TRAIN_METHODS" \
  bash scripts/train-multi-person.sh

echo "==> Done. Eval example:"
echo "  UNLEARN_MODEL_PATH=outputs/unlearn/SD-train/dpo/500/Obama_FaceInpaint \\"
echo "    OUTPUT_DIR=eval_results/Obama_FaceInpaint_b500_s${MAX_STEPS} \\"
echo "    bash scripts/eval-person.sh"
