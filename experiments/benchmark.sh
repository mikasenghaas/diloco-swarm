#!/bin/bash

TAGS="Benchmark"

# Benchmarking GPU throughput of GPT-2 on RTX 3090
# Model: GPT-2 (124M)
# Data: Wikitext (5%)

# Batch Size = 512
# Sequence Length = 1024
# => 512 * 1024 = 524288 ~ 0.5M T/step

# Constants
SEQ_LEN=1024
BATCH_SIZE=512
MAX_STEPS=5

# Vary
MICRO_BATCH_SIZES=(1 2 4 8 16)
MAX_MICRO_BATCHES=(1 2 4 8 16 32 64 128)

# Single GPU
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 1 src/train.py \
#         --swarm.num_stages 1 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.batch_size $BATCH_SIZE \
#         --data.seq_length $SEQ_LEN \
#         --train.max_steps $MAX_STEPS \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --eval.enable false \
#         --sample.enable false \
#         --logging.wandb.enable true \
#         --logging.wandb.tags "$TAGS,Single-GPU"
# done

# DP (2 Workers)
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 2 src/train.py \
#         --swarm.num_stages 1 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.batch_size $BATCH_SIZE \
#         --data.seq_length $SEQ_LEN \
#         --train.max_steps $MAX_STEPS \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --eval.enable false \
#         --sample.enable false \
#         --logging.wandb.enable true \
#         --logging.wandb.tags "$TAGS,DP"
# done

# DP (4 Workers)
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 4 src/train.py \
#         --swarm.num_stages 1 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.batch_size $BATCH_SIZE \
#         --data.seq_length $SEQ_LEN \
#         --train.max_steps $MAX_STEPS \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --eval.enable false \
#         --sample.enable false \
#         --logging.wandb.enable true \
#         --logging.wandb.tags "$TAGS,DP"
# done

# DP (8 Workers)
# for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
# do
#     torchrun --nproc_per_node 8 src/train.py \
#         --swarm.num_stages 1 \
#         --model @configs/model/gpt2-small.toml \
#         --data @configs/data/wikitext.toml \
#         --train.batch_size $BATCH_SIZE \
#         --data.seq_length $SEQ_LEN \
#         --train.max_steps $MAX_STEPS \
#         --train.micro_batch_size $MICRO_BATCH_SIZE \
#         --eval.enable false \
#         --sample.enable false \
#         --logging.wandb.enable true \
#         --logging.wandb.tags "$TAGS,DP"
# done

# PP (2 Workers)
for MICRO_BATCH_SIZE in "${MICRO_BATCH_SIZES[@]}"
do
    for MAX_MICRO_BATCHES in "${MAX_MICRO_BATCHES[@]}"
    do
        if [ $(($MICRO_BATCH_SIZE * $MAX_MICRO_BATCHES)) -le 128 ]; then
            torchrun --nproc_per_node 2 src/train.py \
                --swarm.num_stages 2 \
                --model @configs/model/gpt2-small.toml \
                --data @configs/data/wikitext.toml \
                --train.batch_size $BATCH_SIZE \
                --data.seq_length $SEQ_LEN \
                --train.max_steps $MAX_STEPS \
                --train.micro_batch_size $MICRO_BATCH_SIZE \
                --train.max_micro_batches $MAX_MICRO_BATCHES \
                --eval.enable false \
                --sample.enable false \
                --logging.wandb.enable true \
                --logging.wandb.tags "$TAGS,PP"
        fi
    done
done