#!/usr/bin/env bash
# FIXED_IDEA: generate person pairs for ONE or MANY celebrities.
#
# ---------------------------------------------------------------------------
# Multi-person (preferred):
#   PERSONS="Barack Obama,Elon Musk,Donald Trump" bash scripts/prepare-person.sh
#   PERSONS="obama,elon,trump" NUM_IMAGES=50 bash scripts/prepare-person.sh
#
# Single person (legacy):
#   PERSON="Barack Obama" bash scripts/prepare-person.sh
#   PERSON="Elon Musk" SLUG=elon CONCEPT_PREFIX=Musk bash scripts/prepare-person.sh
#
# Env:
#   PERSONS          comma-separated names OR presets (obama|elon|trump)
#   PERSON           single name if PERSONS unset
#   NUM_IMAGES       default 64 (pilot: 50)
#   METHODS          sdedit,face_inpaint | all
#   DEVICE           cuda
#   NO_VERIFY        1 = skip ArcFace filter
#   SKIP_PILOT       1 = skip pilot_pair_compare
# ---------------------------------------------------------------------------

set -euo pipefail

NUM_IMAGES=${NUM_IMAGES:-64}
METHODS=${METHODS:-"sdedit,face_inpaint"}
DEVICE=${DEVICE:-"cuda"}
NO_VERIFY=${NO_VERIFY:-"0"}
SKIP_PILOT=${SKIP_PILOT:-"0"}

base_dir=$(pwd)
if [[ ! -f "$base_dir/datasets/person_data/generate_person_data.py" ]]; then
  echo "ERROR: run from repo root (missing datasets/person_data/generate_person_data.py)"
  exit 1
fi

# --- preset map: key -> "Full Name|slug|Prefix|Synonym" ---
preset_line() {
  case "$1" in
    obama|barack|barackobama) echo "Barack Obama|obama|Obama|President Obama" ;;
    elon|musk|elonmusk)       echo "Elon Musk|elon|Musk|Elon Reeve Musk" ;;
    trump|donald|donaldtrump) echo "Donald Trump|trump|Trump|President Trump" ;;
    *) echo "" ;;
  esac
}

# slugify last word of full name -> lowercase
derive_slug() {
  local name="$1"
  local last="${name##* }"
  echo "$last" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9'
}

derive_prefix() {
  local name="$1"
  echo "${name##* }"
}

# Build array of records: full|slug|prefix|synonym
declare -a PEOPLE=()

if [[ -n "${PERSONS:-}" ]]; then
  # split on comma
  IFS=',' read -ra RAW <<< "$PERSONS"
  for raw in "${RAW[@]}"; do
    # trim spaces
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [[ -z "$raw" ]] && continue

    key=$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -d ' ._-')
    line=$(preset_line "$key")
    # also try single-word key (obama)
    if [[ -z "$line" ]]; then
      key2=$(echo "$raw" | tr '[:upper:]' '[:lower:]' | awk '{print $NF}')
      line=$(preset_line "$key2")
    fi

    if [[ -n "$line" ]]; then
      PEOPLE+=("$line")
    else
      # free-form full name
      slug=$(derive_slug "$raw")
      prefix=$(derive_prefix "$raw")
      PEOPLE+=("${raw}|${slug}|${prefix}|${raw}")
    fi
  done
elif [[ -n "${PERSON:-}" ]]; then
  slug=${SLUG:-$(derive_slug "$PERSON")}
  prefix=${CONCEPT_PREFIX:-$(derive_prefix "$PERSON")}
  synonym=${SYNONYM:-"$PERSON"}
  PEOPLE+=("${PERSON}|${slug}|${prefix}|${synonym}")
else
  # default trio for FIXED_IDEA multi-celeb
  PEOPLE+=(
    "Barack Obama|obama|Obama|President Obama"
    "Elon Musk|elon|Musk|Elon Reeve Musk"
    "Donald Trump|trump|Trump|President Trump"
  )
fi

if [[ ${#PEOPLE[@]} -eq 0 ]]; then
  echo "ERROR: no persons resolved from PERSONS/PERSON"
  exit 1
fi

verify_flag=()
if [[ "$NO_VERIFY" == "1" ]]; then
  verify_flag=(--no_verify)
fi

echo "============================================================"
echo " prepare-person: ${#PEOPLE[@]} person(s)  N=$NUM_IMAGES  methods=$METHODS"
echo "============================================================"
for rec in "${PEOPLE[@]}"; do
  IFS='|' read -r _name _slug _prefix _syn <<< "$rec"
  echo "  - $_name  (slug=$_slug  prefix=$_prefix)"
done
echo "============================================================"

# write a small registry for later train scripts
mkdir -p datasets/person_data
REG_FILE="datasets/person_data/persons_registry.jsonl"
: > "$REG_FILE"

for rec in "${PEOPLE[@]}"; do
  IFS='|' read -r person slug prefix synonym <<< "$rec"

  echo ""
  echo "############################################################"
  echo "# DATA: $person"
  echo "#   slug=$slug  concept_prefix=$prefix  synonym=$synonym"
  echo "############################################################"

  python3 datasets/person_data/generate_person_data.py \
    --person "$person" \
    --slug "$slug" \
    --num_images "$NUM_IMAGES" \
    --methods "$METHODS" \
    --device "$DEVICE" \
    --concept_prefix "$prefix" \
    "${verify_flag[@]}"

  if [[ "$SKIP_PILOT" != "1" ]]; then
    echo "==> pilot_pair_compare ($slug)"
    python3 -m evaluation.pilot_pair_compare \
      --slug_dir "datasets/person_data/$slug" || {
        echo "WARNING: pilot_pair_compare failed for $slug (continue)"
      }
  fi

  # registry line (append JSONL)
  python3 -c "
import json
print(json.dumps({
  'person': '''$person''',
  'slug': '''$slug''',
  'concept_prefix': '''$prefix''',
  'synonym': '''$synonym''',
  'concepts': {
    'sdedit': '''${prefix}_SDEdit''',
    'face_inpaint': '''${prefix}_FaceInpaint''',
    'face_crop': '''${prefix}_FaceCrop''',
  },
  'num_images': int('''$NUM_IMAGES'''),
  'methods': '''$METHODS''',
}))
" >> "$REG_FILE"

  echo "==> Train commands for $person:"
  echo "  CONCEPT=${prefix}_SDEdit PERSON=\"$person\" SYNONYM=\"$synonym\" bash scripts/sd-person.sh"
  echo "  CONCEPT=${prefix}_FaceInpaint PERSON=\"$person\" SYNONYM=\"$synonym\" bash scripts/sd-person.sh"
done

echo ""
echo "============================================================"
echo " DONE. Registry: $REG_FILE"
echo " Train all (example):"
echo "   PERSONS=\"obama,elon,trump\" NUM_SAMPLES=$NUM_IMAGES BETAS=500 bash scripts/train-multi-person.sh"
echo " Or loop registry + sd-person.sh"
echo "============================================================"
