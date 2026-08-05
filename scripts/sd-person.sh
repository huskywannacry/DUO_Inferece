#!/usr/bin/env bash
# FIXED_IDEA Stage 4: train DUO on person pairs (reuse train/unlearn-sd.py).
#
# Prerequisites:
#   python3 datasets/person_data/generate_person_data.py --num_images 64 \
#       --methods sdedit,face_inpaint
#
# Usage (repo root):
#   # baseline DUO-SDEdit
#   CONCEPT=Obama_SDEdit bash scripts/sd-person.sh
#   # ours face inpaint
#   CONCEPT=Obama_FaceInpaint bash scripts/sd-person.sh
#   # mini-train pilot
#   CONCEPT=Obama_SDEdit MAX_STEPS=500 NUM_SAMPLES=50 BETAS=500 bash scripts/sd-person.sh
#   # enable cross-attn (RQ2)
#   CROSS_ATTN=1 CONCEPT=Obama_FaceInpaint bash scripts/sd-person.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

CONCEPT=${CONCEPT:-"Obama_SDEdit"}
PERSON=${PERSON:-"Barack Obama"}
SYNONYM=${SYNONYM:-"President Obama"}
PRIOR=${PRIOR:-"a person"}
NUM_SAMPLES=${NUM_SAMPLES:-64}
BETAS=${BETAS:-"500 250"}
MAX_STEPS=${MAX_STEPS:-1000}
CROSS_ATTN=${CROSS_ATTN:-0}
SEED=${SEED:-42}
RANK=${RANK:-32}

base_dir=$(pwd)
if [[ ! -f "$base_dir/train/unlearn-sd.py" ]]; then
  echo "Run from repo root"; exit 1
fi

config_dir="$base_dir/datasets/person_data/config.json"
data_dir="$base_dir/datasets/person_data"
save_dir="${SAVE_DIR:-$base_dir/outputs}"

if [[ ! -f "$config_dir" ]]; then
  echo "ERROR: $config_dir missing. Generate data first."
  exit 1
fi
if [[ ! -d "$data_dir/$CONCEPT/unsafe" || ! -d "$data_dir/$CONCEPT/safe" ]]; then
  echo "ERROR: missing $data_dir/$CONCEPT/{unsafe,safe}"
  exit 1
fi

cross_flag=(--no_cross_attn)
if [[ "$CROSS_ATTN" == "1" ]]; then
  cross_flag=()
fi

echo "==> DUO person train"
echo "    concept=$CONCEPT samples=$NUM_SAMPLES steps=$MAX_STEPS betas=$BETAS cross_attn=$CROSS_ATTN"

cd "$base_dir/train"
for dcoloss_beta in $BETAS; do
  lora_dir="$save_dir/unlearn/SD-train/dpo/$dcoloss_beta/$CONCEPT"
  mkdir -p "$lora_dir"
  echo "==> beta=$dcoloss_beta -> $lora_dir"
  python3 unlearn-sd.py \
      --config_dir="$config_dir" \
      --config_name="${CONCEPT}" \
      --data_dir="$data_dir" \
      --output_dir="${lora_dir}" \
      --target_prompt="${PERSON}" \
      --synonym_prompt="${SYNONYM}" \
      --prior_prompt="${PRIOR}" \
      --base_lr=3e-4 \
      --adam_weight_decay=1e-2 \
      --dcoloss_beta="$dcoloss_beta" \
      --base_lambda=1e6 \
      --rank="$RANK" \
      --method=dpo \
      --train_batch_size=1 \
      --max_train_steps="$MAX_STEPS" \
      --checkpointing_steps=250 \
      --validation_steps=250 \
      --num_validation_images=2 \
      --num_samples="$NUM_SAMPLES" \
      --t_max=750 \
      --t_min=1 \
      --dataloader_num_workers=0 \
      "${cross_flag[@]}" \
      --seed="$SEED"
done

echo "==> Done. Evaluate with: bash scripts/eval-person.sh"
