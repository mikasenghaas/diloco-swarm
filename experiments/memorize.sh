#!/bin/bash

set -e
TAGS="Memorize"

# Single GPU
torchrun --nproc_per_node 1 src/train.py \
    @configs/memorize.toml \
    --world.num_stages 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --logging.wandb.tags "$TAGS,Single-GPU"

# DP
torchrun --nproc_per_node 2 src/train.py \
    @configs/memorize.toml \
    --world.num_stages 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --logging.wandb.tags "$TAGS,DP"

# PP
torchrun --nproc_per_node 2 src/train.py \
    @configs/memorize.toml \
    --world.num_stages 2 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --logging.wandb.tags "$TAGS,PP"

# SWARM
# torchrun --nproc_per_node 4 src/train/train.py \
#     @configs/memorize.toml \
#     --world.num_stages 2 \
#     --model @configs/model/gpt2-small.toml \
#     --data @configs/data/memorize.toml \
#     --logging.wandb.tags "$TAGS,PP"