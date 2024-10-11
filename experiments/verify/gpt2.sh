#!/bin/bash

set -e

GROUP="verify/gpt2"

# Run 1: GPT-2 on 1% of Fineweb-Edu 10BT
# ---
# Model: GPT-2 (124M)
# Data: FineWeb-Edu (100MT)
# Configuration: config/train.toml

# Tokens/ Step = batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M
# Total Steps = 100M tokens / 0.5M tokens/step ~ 200 steps

python src/train/baseline.py @configs/train.toml \
    --model @configs/model/gpt2-124m.toml \
    --data @configs/data/fineweb-edu-100mt.toml \
    --train.scheduler.warmup_steps 20 \
    --train.micro_batch_size 2 \
    --logging.wandb.group $GROUP