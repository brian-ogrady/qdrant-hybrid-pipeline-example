#!/usr/bin/env bash

# Exit immediately if a command fails, to prevent partial runs.
set -e

# --- 1. Environment Setup ---
echo "--- Setting up Python environment ---"
rm -rf .venv/
uv venv
uv pip install -e .
source .venv/bin/activate

# --- 2. Start Services ---
echo "--- Starting Qdrant cluster in the background ---"
docker-compose up -d
echo "--- Waiting for cluster to stabilize (30s) ---"
sleep 30

# --- 3. Create output directories ---
RESULTS_DIR="data/search_results"
mkdir -p $RESULTS_DIR

# --- 4. Loop through all experiment configurations ---
for config_file in configs/*.yml; do
    # Extract the base name of the config file (e.g., "baseline_experiment")
    config_name=$(basename "$config_file" .yml)
    
    echo "=================================================="
    echo "RUNNING EXPERIMENT: $config_name"
    echo "=================================================="

    # Define dynamic paths for this experiment's output
    results_path="$RESULTS_DIR/${config_name}_results.json"
    # Use a unique collection name for each experiment to avoid conflicts
    collection_name="collection_${config_name}"

    # --- Run Ingestion & Search ---
    echo "--- Running data ingestion and search queries for $config_name ---"
    python src/scripts/search_ads.py \
        --config "$config_file" \
        --csv-path data/generated_keywords.csv \
        --text-column keyword \
        --queries-path data/queries.json \
        --results-path "$results_path" \
        --collection-name "$collection_name" \
        --batch-size 64 \
        --top-k 10

done

# --- 5. Teardown ---
echo "--- All experiments finished. Tearing down Qdrant cluster. ---"
docker-compose down -v
rm -rf .venv/

echo "--- Full evaluation pipeline finished successfully! ---"

