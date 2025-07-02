#!/usr/bin/env bash

rm -rf .venv/
uv venv
uv pip install -e .
source .venv/bin/activate

docker-compose up -d
sleep 30

python scripts/search_ads.py \
    --config configs/baseline_experiment.yml \
    --csv-path data/generated_keywords.csv \
    --text-column keyword \
    --queries-path data/queries.json \
    --batch-size 64 \
    --top-k 10

docker-compose down -v

rm -rf .venv/
