#!/bin/bash

set -e
TAGS="Test,Sync"

# Baseline (Full DP)
torchrun --nproc_per_node 1 src/train.py \
    --swarm.num_stages 1 \
    --swarm.sync_every_n_steps 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/fineweb-edu-1bt.toml \
    --data.subset_size 0.1 \
    --train.max_epochs 1 \
    --train.micro_batch_size 16 \
    --eval.enable false \
    --eval.every_n_steps 10 \
    --eval.max_steps 50 \
    --sample.enable false \
    --logging.wandb.enable true \
    --logging.wandb.tags "$TAGS,DP"

# SYNC_EVERY_N_STEPS=(1 50 400)

# Disable sample because it blocks training after
# Disable eval because it blocks training at step 64