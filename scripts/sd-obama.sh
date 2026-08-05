#!/usr/bin/env bash
# Train DUO on Obama identity pairs (same recipe as sd-nudity.sh).
#
# Prerequisites:
#   python3 datasets/SD/generate_person_data.py \
#       --concept Obama --person "Barack Obama" \
#       --method sdedit --num_images 64 --save_dir datasets/SD
#
# Usage (from repo root):
#   bash scripts/sd-obama.sh
#   CONCEPT=Obama_FaceEdit bash scripts/sd-obama.sh
#   BETAS="500" bash scripts/sd-obama.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# ---- knobs (override via env) ----
CONCEPT=${CONCEPT:-"Obama"}
PERSON=${PERSON:-"Barack Obama"}
SYNONYM=${SYNONYM:-"President Obama"}
PRIOR=${PRIOR:-"a person"}
NUM_SAMPLES=${NUM_SAMPLES:-64}
# Paper: β=500 black-box, β=250 white-box. Default both.
BETAS=${BETAS:-"500 250"}
MAX_STEPS=${MAX_STEPS:-1000}
# Set CROSS_ATTN=1 to also train cross-attn (identity often needs name binding).
CROSS_ATTN=${CROSS_ATTN:-0}
SEED=${SEED:-42}

base_dir=$(pwd)
# Allow running from repo root or from scripts/
if [[ ! -f "$base_dir/train/unlearn-sd.py" && -f "$base_dir/../train/unlearn-sd.py" ]]; then
  base_dir=$(cd "$base_dir/.." && pwd)
fi

config_dir="$base_dir/datasets/SD/config_person.json"
data_dir="$base_dir/datasets/SD"
save_dir="${SAVE_DIR:-$base_dir/outputs}"

if [[ ! -f "$config_dir" ]]; then
  echo "ERROR: missing $config_dir — run generate_person_data.py first."
  exit 1
fi
if [[ ! -d "$data_dir/$CONCEPT/unsafe" || ! -d "$data_dir/$CONCEPT/safe" ]]; then
  echo "ERROR: missing $data_dir/$CONCEPT/{unsafe,safe}"
  echo "  python3 datasets/SD/generate_person_data.py --concept $CONCEPT --person \"$PERSON\" --method sdedit --num_images $NUM_SAMPLES"
  exit 1
fi

n_unsafe=$(find "$data_dir/$CONCEPT/unsafe" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
n_safe=$(find "$data_dir/$CONCEPT/safe" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
echo "==> Train DUO person unlearning"
echo "    concept     : $CONCEPT"
echo "    pairs       : unsafe=$n_unsafe safe=$n_safe (using num_samples=$NUM_SAMPLES)"
echo "    betas       : $BETAS"
echo "    cross_attn  : $CROSS_ATTN"

cross_flag=()
if [[ "$CROSS_ATTN" == "0" ]]; then
  cross_flag=(--no_cross_attn)
fi

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
      --rank=32 \
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

echo "==> Done. LoRA under $save_dir/unlearn/SD-train/dpo/{beta}/$CONCEPT"
echo "    Eval: python3 -m evaluation.person_metrics --unlearn_model_path ..."
