#!/bin/bash

GROUP="benchmark/baseline/gpt2"

# Benchmarking GPU throughput of GPT-2 on single GPU
# Model: GPT-2 (124M)
# Data: Debug (100%)

# batch_size = 512
# seq_length = 1024
# => batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M T/step

# train.amp.precision: [high, highest]
# train.amp.dtype: [float16, bfloat16]
# train.micro_batch_size: [1, 2, 4, 8, 16]

PRECISION="highest high"
DTYPE="float32 bfloat16"
MICRO_BATCH_SIZE="1 2 4 8 16"

for PRECISION in $PRECISION
do
    for DTYPE in $DTYPE
    do
        for MICRO_BATCH_SIZE in $MICRO_BATCH_SIZE
        do
            python src/train/baseline.py @configs/benchmark.toml \
                --model @configs/model/gpt2-124m.toml \
                --data @configs/data/wikitext.toml \
                --train.amp.precision $PRECISION \
                --train.amp.dtype $DTYPE \
                --train.micro_batch_size $MICRO_BATCH_SIZE \
                --logging.wandb.group $GROUP
        done
    done
done