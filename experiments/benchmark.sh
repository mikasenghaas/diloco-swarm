#!/bin/bash

set -e
TAGS="Benchmark"

# Benchmarking GPU throughput of GPT-2 on RTX 3090
# Model: GPT-2 (124M)
# Data: Wikitext (5%)

# Batch Size = 512
# Sequence Length = 1024
# => 512 * 1024 = 524288 ~ 0.5M T/step

# Micro Batch Size: [1, 2, 4, 8, 16]
MICRO_BATCH_SIZES=(1 2 4 8 16)

# Single GPU
for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
do
    torchrun --nproc_per_node 1 src/train.py \
        @configs/benchmark.toml \
        --swarm.num_stages 1 \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/wikitext.toml \
        --train.micro_batch_size $MICRO_BATCH_SIZE \
        --logging.wandb.tags $TAGS
done

# DP
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 2 src/train.py \
#         @configs/benchmark.toml \
#         --swarm.num_stages 1 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --logging.wandb.tags $TAGS
# done
# 
# # PP
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 2 src/train.py \
#         @configs/benchmark.toml \
#         --swarm.num_stages 2 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --logging.wandb.tags $TAGS
# done
# 
# # SWARM
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 4 src/train.py \
#         @configs/benchmark.toml \
#         --world.num_stages 2 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --logging.wandb.tags $TAGS
# done
# 