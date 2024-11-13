#!/bin/bash

set -e
GROUP="baseline/verify/ckpt"

python src/train/baseline.py @configs/debug.toml \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --sample.enable true \
    --train.max_steps 100 \
    --train.batch_size 1 \
    --train.micro_batch_size 1 \
    --logging.ckpt.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP