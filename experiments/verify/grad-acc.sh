#!/bin/bash

set -e
GROUP="verify/grad-acc"

# Run 1: micro_batch_size=1, batch_size=64
# Run 2: micro_batch_size=2, batch_size=64
# Run 3: micro_batch_size=4, batch_size=64
# Run 4: micro_batch_size=8, batch_size=64
# Run 5: micro_batch_size=16, batch_size=64
# Run 6: micro_batch_size=32, batch_size=64
# Run 7: micro_batch_size=64, batch_size=64

for MICRO_BATCH_SIZE in 1 2 4 8 16 32 64
do
    python src/train/baseline.py @configs/debug.toml \
        --model @configs/model/llama2.toml \
        --data @configs/data/wikitext.toml \
        --logging.file.enable true \
        --logging.wandb.enable true \
        --logging.wandb.group $GROUP \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --train.batch_size 64
done