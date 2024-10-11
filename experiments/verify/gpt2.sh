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

# RTX 4090 24GB: Max micro-batch size = 8 (33 kT/s)
# A100 80GB: ?
# A100 80GB: Max micro-batch size = 64 (44 kT/s)
# H100 80GB: ?

python src/train/baseline.py @configs/train.toml \
    --model @configs/model/gpt2-124m.toml \
    --data @configs/data/fineweb-edu-100mt.toml \
    --train.scheduler.warmup_steps 20 \
    --train.micro_batch_size 8 \
    --logging.wandb.group $GROUP