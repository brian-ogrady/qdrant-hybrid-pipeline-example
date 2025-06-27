#!/usr/bin/env bash

rm -rf .venv/
uv venv
uv pip install -e .
source .venv/bin/activate

docker-compose up -d
sleep 30

python wiki_cohere.py --num-records 1000 --num-tenants 10 --batch-size 64

docker-compose down -v

rm -rf .venv/
