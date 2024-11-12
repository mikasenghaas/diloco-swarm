#!/bin/bash

set -e
GROUP="verify/baseline/ckpt"

# Single-GPU overfit experiments to verify checkpointing
# Run 1: LLama 2 (9M)
# Run 2: GPT-2 (124M)

MODELS="llama2-9m gpt2-124m"
for MODEL in $MODELS; do
    python src/train/baseline.py @configs/debug.toml \
        --model @configs/model/$MODEL.toml \
        --data @configs/data/memorize.toml \
        --sample.enable true \
        --train.max_steps 100 \
        --train.batch_size 1 \
        --train.micro_batch_size 1 \
        --logging.ckpt.enable true \
        --logging.wandb.enable true \
        --logging.wandb.group $GROUP
done
