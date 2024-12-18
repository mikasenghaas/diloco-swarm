#!/bin/bash

set -e
TAGS="Test,Sync"

# Baseline (Single GPU)
# torchrun --nproc_per_node 1 src/train.py \
#     --swarm.num_stages 1 \
#     --swarm.sync_every_n_steps 1 \
#     --model @configs/model/gpt2-small.toml \
#     --data @configs/data/fineweb-edu-1bt.toml \
#     --data.subset_size 0.1 \
#     --train.max_epochs 1 \
#     --train.micro_batch_size 16 \
#     --eval.every_n_steps 20 \
#     --eval.max_steps 50 \
#     --sample.every_n_steps 10 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS,Single-GPU"

# Baseline (DP)
SYNC_EVERY_N_STEPS=(100)
for SYNC_EVERY_N_STEPS in ${SYNC_EVERY_N_STEPS[@]}; do
    torchrun --nproc_per_node 2 src/train.py \
        --swarm.num_stages 1 \
        --swarm.sync_every_n_steps $SYNC_EVERY_N_STEPS \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/fineweb-edu-1bt.toml \
        --data.subset_size 0.1 \
        --train.micro_batch_size 16 \
        --train.step_timeout 60 \
        --eval.every_n_steps 10 \
        --eval.max_steps 10 \
        --sample.enable true \
        --sample.every_n_steps 10 \
        --logging.wandb.enable true \
        --logging.wandb.tags "$TAGS,DP"
done