#!/bin/bash

set -e
TAGS="baseline,grad-acc"

for MICRO_BATCH_SIZE in 1 32 64
do
    python src/train/baseline.py @configs/debug.toml \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/wikitext.toml \
        --logging.file.enable true \
        --logging.wandb.enable true \
        --logging.wandb.tags $TAGS \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --train.batch_size 64
done