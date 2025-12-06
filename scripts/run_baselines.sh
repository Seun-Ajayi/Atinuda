#!/usr/bin/env bash

# Variables
PROVIDER="openai"
MODEL_NAME="gpt-4.1-mini"
CUSTOM_TASKS="tasks/baselines/afriqa.py"
RESULTS_ORG="seun-ajayi"

# AfriQA languages
LANGS=(
  "yor"
#   "hau"
#   "ibo"
#   "swa"
#   "zul"
#   "twi"
#   "wol"
#   "kin"
#   "fon"
#   "bem"
)

# Variants
MODES=(
  "native"
  # "pivot"
)

# Loop through each language+mode and execute the lighteval command
for lang in "${LANGS[@]}"; do
  for mode in "${MODES[@]}"; do
    TASK_NAME="afriqa:${lang}:${mode}"
    echo "Running evaluation for ${TASK_NAME}..."

    lighteval endpoint litellm \
      "provider=${PROVIDER},model_name=${MODEL_NAME}" \
      "${TASK_NAME}" \
      --custom-tasks "${CUSTOM_TASKS}" \
      --push-to-hub \
      --public-run \
      --results-org "${RESULTS_ORG}"

    echo "✅ Finished ${TASK_NAME}"
    echo "--------------------------------------"
  done
done
