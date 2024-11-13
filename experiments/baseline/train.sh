#!/bin/bash

set -e

GROUP="train/baseline/gpt2"

# Train GPT-2 on 10% of Fineweb-Edu (1BT)
# ---
# Model: GPT-2 (124M)
# Data: FineWeb-Edu (1BT)
# Configuration: config/train.toml

# Tokens/ Step = batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M
# Total Steps = 1B tokens / 0.5M tokens/step ~ 2000 steps

# Train on ???
# Avg. Throughput: 

python src/train/baseline.py @configs/train.toml \
    --model @configs/model/gpt2-124m.toml \
    --data @configs/data/fineweb-edu-1bt.toml \
    --train.scheduler.warmup_steps 200 \
    --train.micro_batch_size 8 \
    --logging.wandb.group $GROUP