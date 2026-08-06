#!/usr/bin/env bash
# Train DUO for every person listed in PERSONS (or in persons_registry.jsonl).
#
# Usage (after prepare-person.sh):
#   PERSONS="obama,elon,trump" bash scripts/train-multi-person.sh
#   PERSONS="Barack Obama,Elon Musk,Donald Trump" BETAS=500 NUM_SAMPLES=50 bash scripts/train-multi-person.sh
#   # or rely on registry written by prepare-person:
#   bash scripts/train-multi-person.sh

set -euo pipefail

BETAS=${BETAS:-"500"}
MAX_STEPS=${MAX_STEPS:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-50}
# which pair methods to train (folder suffixes)
TRAIN_METHODS=${TRAIN_METHODS:-"SDEdit,FaceInpaint"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/scripts/sd-person.sh" ]]; then
  echo "Run from repo root"; exit 1
fi

preset_line() {
  case "$1" in
    obama|barack|barackobama) echo "Barack Obama|obama|Obama|President Obama" ;;
    elon|musk|elonmusk)       echo "Elon Musk|elon|Musk|Elon Reeve Musk" ;;
    trump|donald|donaldtrump) echo "Donald Trump|trump|Trump|President Trump" ;;
    *) echo "" ;;
  esac
}

declare -a PEOPLE=()

if [[ -n "${PERSONS:-}" ]]; then
  IFS=',' read -ra RAW <<< "$PERSONS"
  for raw in "${RAW[@]}"; do
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [[ -z "$raw" ]] && continue
    key=$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -d ' ._-')
    line=$(preset_line "$key")
    if [[ -z "$line" ]]; then
      key2=$(echo "$raw" | tr '[:upper:]' '[:lower:]' | awk '{print $NF}')
      line=$(preset_line "$key2")
    fi
    if [[ -n "$line" ]]; then
      PEOPLE+=("$line")
    else
      last="${raw##* }"
      slug=$(echo "$last" | tr '[:upper:]' '[:lower:]')
      PEOPLE+=("${raw}|${slug}|${last}|${raw}")
    fi
  done
elif [[ -f datasets/person_data/persons_registry.jsonl ]]; then
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    person=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['person'])" "$line")
    slug=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['slug'])" "$line")
    prefix=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['concept_prefix'])" "$line")
    synonym=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['synonym'])" "$line")
    PEOPLE+=("${person}|${slug}|${prefix}|${synonym}")
  done < datasets/person_data/persons_registry.jsonl
else
  PEOPLE+=(
    "Barack Obama|obama|Obama|President Obama"
    "Elon Musk|elon|Musk|Elon Reeve Musk"
    "Donald Trump|trump|Trump|President Trump"
  )
fi

IFS=',' read -ra SUFFIXES <<< "$TRAIN_METHODS"

for rec in "${PEOPLE[@]}"; do
  IFS='|' read -r person slug prefix synonym <<< "$rec"
  for suffix in "${SUFFIXES[@]}"; do
    suffix=$(echo "$suffix" | tr -d ' ')
    concept="${prefix}_${suffix}"
    if [[ ! -d "datasets/person_data/$concept/unsafe" ]]; then
      echo "SKIP $concept (no data — run prepare-person.sh with PERSONS=... first)"
      continue
    fi
    echo ""
    echo "============================================================"
    echo " TRAIN $concept | $person | betas=$BETAS steps=$MAX_STEPS"
    echo "============================================================"
    CONCEPT="$concept" \
    PERSON="$person" \
    SYNONYM="$synonym" \
    PRIOR="a person" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    BETAS="$BETAS" \
    MAX_STEPS="$MAX_STEPS" \
      bash scripts/sd-person.sh
  done
done

echo "==> train-multi-person done."
