#!/usr/bin/env bash
# Concept Inversion - white-box red teaming attack (paper-faithful).
#
# Mirrors CCE's train.sh launcher pattern (env vars + sequential steps),
# tuned to the DUO paper:
#   - Paper Fig 5 / Sec 4.1: white-box attacks use beta=250.
#   - Paper Appendix C.2: Adam, lr=5e-3, batch_size=4, 3000 steps.
#   - Anchors from ORIGINAL SD1.4; <c> trained on the UNLEARNED pipeline;
#     <c> used as prefix of I2P sexual prompts; NudeNet 4-label, thr=0.5.
#
# Usage:
#   UNLEARN_MODEL_PATH=... I2P_REPO=... OUTPUT_DIR=... ./run.sh
#
# Each step is resumable: re-running skips anchors that already exist.

set -euo pipefail

# ---- Config (override via env) ----
# Paper recommends beta=250 for white-box (Concept Inversion) attacks.
UNLEARN_MODEL_PATH=${UNLEARN_MODEL_PATH:-"train/outputs/unlearn/SD-train/dpo/250/Nudity"}
I2P_REPO=${I2P_REPO:-"/home/kientt44/Code/i2p"}
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/ci_nudity_b250"}
EXP_TYPE=${EXP_TYPE:-"nudity"}
NUM_PROMPTS=${NUM_PROMPTS:-200}       # paper uses 741; 200 fits a T4 in ~12h
MAX_STEPS=${MAX_STEPS:-3000}          # paper Appendix C.2
BATCH_SIZE=${BATCH_SIZE:-4}           # paper Appendix C.2
LEARNING_RATE=${LEARNING_RATE:-5e-3}  # paper Appendix C.2
PLACEHOLDER_TOKEN=${PLACEHOLDER_TOKEN:-"<c>"}
INITIALIZER_TOKEN=${INITIALIZER_TOKEN:-"man"}
DEVICE=${DEVICE:-cuda}

ANCHOR_DIR="${OUTPUT_DIR}/anchor_images"
EMBED_PATH="${OUTPUT_DIR}/learned_embeds.bin"
GEN_DIR="${OUTPUT_DIR}/generated"

mkdir -p "${OUTPUT_DIR}"
echo "==> Concept Inversion attack"
echo "    unlearn model : ${UNLEARN_MODEL_PATH}"
echo "    output dir    : ${OUTPUT_DIR}"
echo "    exp_type      : ${EXP_TYPE}"

# Step 1: anchor NSFW images with ORIGINAL SD1.4
echo "==> [1/4] Generating anchor images (original SD1.4)"
python3 -m evaluation.concept_inversion.generate_anchors \
    --output_dir "${ANCHOR_DIR}" \
    --exp_type "${EXP_TYPE}" \
    --i2p_repo "${I2P_REPO}" \
    --num_prompts "${NUM_PROMPTS}" \
    --device "${DEVICE}"

# Step 2: textual inversion of <c> on the UNLEARNED pipeline
echo "==> [2/4] Training <c> via textual inversion (unlearned model)"
python3 -m evaluation.concept_inversion.concept_inversion \
    --unlearn_model_path "${UNLEARN_MODEL_PATH}" \
    --train_data_dir "${ANCHOR_DIR}" \
    --metadata_path "${ANCHOR_DIR}/metadata.json" \
    --output_dir "${OUTPUT_DIR}" \
    --exp_type "${EXP_TYPE}" \
    --device "${DEVICE}" \
    --placeholder_token "${PLACEHOLDER_TOKEN}" \
    --initializer_token "${INITIALIZER_TOKEN}" \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}"

# Step 3: generate attack images with <c> prefix
echo "==> [3/4] Generating attack images (<c> + I2P prompts)"
python3 -m evaluation.concept_inversion.generate_images \
    --unlearn_model_path "${UNLEARN_MODEL_PATH}" \
    --embed_path "${EMBED_PATH}" \
    --output_dir "${GEN_DIR}" \
    --exp_type "${EXP_TYPE}" \
    --i2p_repo "${I2P_REPO}" \
    --num_prompts "${NUM_PROMPTS}" \
    --device "${DEVICE}"

# Step 4: NudeNet DSR (paper 4-label set, thr=0.5)
echo "==> [4/4] Computing Defense Success Rate (NudeNet)"
python3 -m evaluation.defense_success_rate \
    --task "${EXP_TYPE}" \
    --image_dir "${GEN_DIR}"

echo "==> Done. Results in ${OUTPUT_DIR}"
