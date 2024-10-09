#!/bin/bash

set -e
GROUP="verify/mp2"

# Run 1: train.precision="highest"
# Run 2: train.precision="high"
# Run 3: train.precision="medium"

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.precision "highest" \

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.precision "high" \

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.precision "medium" \