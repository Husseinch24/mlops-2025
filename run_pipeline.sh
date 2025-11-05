#!/bin/bash
set -e

uv run python scripts/preprocess.py \
  --train_path data/raw/train.csv \
  --test_path data/raw/test.csv \
  --output_train data/processed/train_processed.csv \
  --output_test data/processed/test_processed.csv

uv run python scripts/featurize.py \
  --input_train data/processed/train_processed.csv \
  --input_test data/processed/test_processed.csv \
  --output_train data/features/train_features.csv \
  --output_test data/features/test_features.csv

uv run python scripts/train.py \
  --input data/features/train_features.csv \
  --output_model models/log_reg.pkl

uv run python scripts/evaluate.py \
  --model models/log_reg.pkl \
  --data data/features/train_features.csv \
  --output_metrics metrics/train_eval.json

uv run python scripts/predict.py \
  --model models/log_reg.pkl \
  --input data/features/test_features.csv \
  --output predictions/test_predictions.csv
