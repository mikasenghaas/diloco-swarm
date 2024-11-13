#!/bin/bash

set -e

GROUP="baseline/train"

# Train GPT-2 on Fineweb-Edu 1BT
# ---
# Model: GPT-2 (124M)
# Data: FineWeb-Edu (1BT)
# Configuration: config/train.toml

# Tokens/ Step = batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M
# Total Steps = 1B tokens / 0.5M tokens/step ~ 2000 steps

# Train on RTX 3090 (max. micro_batch_size = 16)
# Avg. Throughput: ~60K tokens/s

python src/train/baseline.py @configs/train.toml \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/fineweb-edu-1bt.toml \
    --logging.wandb.group $GROUP