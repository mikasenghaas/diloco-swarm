#!/bin/bash

set -e
GROUP="verify/perf"

# Debug Model: ~9M
# Tokens/ Step = batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M

# Run 1: micro_batch_size=1, batch_size=512, seq_length=1024 (T: 1 * 1024 ~ 1K)
# Run 2: micro_batch_size=2, batch_size=512, seq_length=1024 (T: 2 * 1024 ~ 2K)
# Run 3: micro_batch_size=4, batch_size=512, seq_length=1024 (T: 4 * 1024 ~ 4K)
# Run 4: micro_batch_size=8, batch_size=512, seq_length=1024 (T: 8 * 1024 ~ 8K)
# Run 5: micro_batch_size=16, batch_size=512, seq_length=1024 (T: 16 * 1024 ~ 16K)
# Run 6: micro_batch_size=32, batch_size=512, seq_length=1024 (T: 32 * 1024 ~ 32K)

for MICRO_BATCH_SIZE in 1 2 4 8 16 32
do
    python src/train/baseline.py @configs/baseline/debug.toml \
        --logging.file.enable true \
        --logging.wandb.enable true \
        --logging.wandb.group $GROUP \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --train.batch_size 512 \
        --data.seq_length 1024
done