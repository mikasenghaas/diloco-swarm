#!/bin/bash

set -e
TAGS="Memorize"

SEQ_LEN=128
SUBSET_SIZE=1
MAX_STEPS=100
MAX_EPOCHS=-1
BATCH_SIZE=1
MICRO_BATCH_SIZE=1
ENABLE_WANDB=true

# Single GPU
torchrun --nproc_per_node 1 src/train.py \
    --swarm.num_stages 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --train.batch_size $BATCH_SIZE \
    --data.seq_length $SEQ_LEN \
    --train.max_epochs $MAX_EPOCHS \
    --train.max_steps $MAX_STEPS \
    --train.micro_batch_size $MICRO_BATCH_SIZE \
    --eval.enable false \
    --logging.wandb.enable $ENABLE_WANDB \
    --logging.wandb.tags "$TAGS,Single-GPU"

# DP
torchrun --nproc_per_node 4 src/train.py \
    --swarm.num_stages 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --train.batch_size $BATCH_SIZE \
    --data.seq_length $SEQ_LEN \
    --train.max_epochs $MAX_EPOCHS \
    --train.max_steps $MAX_STEPS \
    --train.micro_batch_size $MICRO_BATCH_SIZE \
    --eval.enable false \
    --logging.wandb.enable $ENABLE_WANDB \
    --logging.wandb.tags "$TAGS,DP"

# PP
torchrun --nproc_per_node 4 src/train.py \
    --swarm.num_stages 4 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --train.batch_size $BATCH_SIZE \
    --data.seq_length $SEQ_LEN \
    --train.max_epochs $MAX_EPOCHS \
    --train.max_steps $MAX_STEPS \
    --train.micro_batch_size $MICRO_BATCH_SIZE \
    --eval.enable false \
    --logging.wandb.enable $ENABLE_WANDB \
    --logging.wandb.tags "$TAGS,PP"

# SWARM
torchrun --nproc_per_node 4 src/train.py \
    --swarm.num_stages 2 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --train.batch_size $BATCH_SIZE \
    --data.seq_length $SEQ_LEN \
    --train.max_epochs $MAX_EPOCHS \
    --train.max_steps $MAX_STEPS \
    --train.micro_batch_size $MICRO_BATCH_SIZE \
    --eval.enable false \
    --logging.wandb.enable $ENABLE_WANDB \
    --logging.wandb.tags "$TAGS,SWARM"
