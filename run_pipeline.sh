#!/usr/bin/env bash

rm -rf .venv/
uv venv
uv pip install -e .
source .venv/bin/activate

docker-compose up -d
sleep 30

python search_ads.py --config configs/baseline_experiment.yml --csv-path generated_keywords.csv --text-column keyword --batch-size 64

docker-compose down -v

rm -rf .venv/
