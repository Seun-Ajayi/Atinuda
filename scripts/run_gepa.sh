#!/usr/bin/env bash
#
# Runs evaluation across multiple models and languages
#

set -e  # Exit on error

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Configuration
TASK_SCRIPT="afriqa.py"
OUTPUT_BASE_DIR="./results/afriqa"

# Models to evaluate
MODELS=(
    "openai/gpt-4o-mini"
    "openai/gpt-3.5-turbo"
)

# Languages to evaluate
LANGUAGES=(
    "yor"
    "hau"
    "swa"
    "ibo"
    "zul"
    "fon"
    "bem"
    "kin"
    "twi"
)

# Track results
declare -a RESULTS

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Evaluation Pipeline                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Models: ${MODELS[*]}"
echo "Languages: ${LANGUAGES[*]}"
echo ""

# Loop through each model and language combination
for model in "${MODELS[@]}"; do
    for lang in "${LANGUAGES[@]}"; do
        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "Running: $model | Language: $lang"
        echo "═══════════════════════════════════════════════════════════"
        echo ""
        
        # Run evaluation
        python "$TASK_SCRIPT" \
            --model "$model" \
            --lang "$lang" \
            --output-dir "$OUTPUT_BASE_DIR"
        
        status=$?
        
        if [ $status -eq 0 ]; then
            echo "✅ Completed: $model | $lang"
            RESULTS+=("✅ $model | $lang")
        else
            echo "❌ Failed: $model | $lang"
            RESULTS+=("❌ $model | $lang")
        fi
        
        echo ""
        echo "───────────────────────────────────────────────────────────"
    done
done

# Print summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    EVALUATION SUMMARY                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

for result in "${RESULTS[@]}"; do
    echo "$result"
done

echo ""
echo "Results saved to: $OUTPUT_BASE_DIR"
echo ""