#!/usr/bin/env bash

# Exit immediately if a command fails, to prevent partial runs.
set -e

# --- 1. Environment Setup ---
echo "--- Setting up Python environment ---"
rm -rf .venv/
uv venv
uv pip install -e .
source .venv/bin/activate

# --- 3. Create output directories ---
RESULTS_DIR="data/search_results"

# --- 4. Loop through all experiment configurations ---
for config_file in configs/*.yml; do
    # Extract the base name of the config file (e.g., "baseline_experiment")
    config_name=$(basename "$config_file" .yml)
    
    echo "=================================================="
    echo "RUNNING EVALUATION: $config_name"
    echo "=================================================="

    results_path="$RESULTS_DIR/${config_name}_results.json"

    # --- Run Ingestion & Search ---
    echo "--- Running data ingestion and search queries for $config_name ---"
    python src/scripts/evaluate_results.py \
        --results-path "$results_path" \
        --text-column document

done

# --- 5. Teardown ---
echo "--- All experiments finished. Tearing down environment. ---"
rm -rf .venv/

echo "--- Full evaluation pipeline finished successfully! ---"

