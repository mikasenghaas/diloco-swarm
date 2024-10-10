#!/bin/bash

set -e

GROUP="verify/perf"

# Using debug model: ~9M with performance config (config/baseline/perf.toml)
# Tokens/ Step = batch_size * seq_length = 512 * 1024 = 524288 ~ 0.5M

# Run 1: micro_batch_size=1 (Tokens/Micro Step: 1 * 1024 ~ 1K)
# Run 2: micro_batch_size=2 (Tokens/Micro Step: 2 * 1024 ~ 2K)
# Run 3: micro_batch_size=4 (Tokens/Micro Step: 4 * 1024 ~ 4K)
# Run 4: micro_batch_size=8 (Tokens/Micro Step: 8 * 1024 ~ 8K)
# Run 5: micro_batch_size=16 (Tokens/Micro Step: 16 * 1024 ~ 16K)
# Run 6: micro_batch_size=32 (Tokens/Micro Step: 32 * 1024 ~ 32K)
# Run 7: micro_batch_size=64 (Tokens/Micro Step: 64 * 1024 ~ 64K)
# Run 8: micro_batch_size=128 (Tokens/Micro Step: 128 * 1024 ~ 128K)
# Run 9: micro_batch_size=256 (Tokens/Micro Step: 256 * 1024 ~ 256K)
# Run 10: micro_batch_size=512 (Tokens/Micro Step: 512 * 1024 ~ 512K)

for MICRO_BATCH_SIZE in 1 2 4 8 16 32 64 128 256 512
do
    python src/train/baseline.py @configs/benchmark.toml \
        --model @configs/model/llama2.toml \
        --data @configs/data/wikitext.toml \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --logging.wandb.group $GROUP
done