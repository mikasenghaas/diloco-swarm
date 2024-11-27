#!/bin/bash

set -e
TAGS="pipeline,benchmark"

# Benchmarking GPU throughput of GPT-2 on 2x3090
# Model: GPT-2 (124M)
# Data: Debug (100%)

# batch_size = 512
# seq_length = 1024
# => batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M T/step

# train.micro_batch_size: [1, 2, 4, 8, 16]

MICRO_BATCH_SIZES=(1 2 4 8 16)
for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
do
    torchrun --nproc_per_node 2 src/train/pipeline.py @configs/benchmark.toml \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/wikitext.toml \
        --logging.console.enable false \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --logging.wandb.tags $TAGS
done