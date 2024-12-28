#!/bin/bash

set -e
TAGS="Sync"

# Baseline (Single GPU)
# torchrun --nproc_per_node 2 src/train.py \
#     --swarm.num_stages 1 \
#     --swarm.sync_every_n_steps 1 \
#     --model @configs/model/gpt2-small.toml \
#     --data @configs/data/fineweb-edu-1bt.toml \
#     --data.subset_size 0.1 \
#     --data.seq_length 1024 \
#     --train.inner_optimizer @configs/optimizer/adamw.toml \
#     --train.outer_optimizer @configs/optimizer/none.toml \
#     --train.inner_optimizer.lr 6e-4 \
#     --train.max_epochs 1 \
#     --train.batch_size 512 \
#     --train.micro_batch_size 16 \
#     --train.max_micro_batches 1 \
#     --eval.enable true \
#     --eval.max_steps 50 \
#     --eval.every_n_steps 10 \
#     --sample.enable true \
#     --sample.every_n_steps -1 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS,Single-GPU"

# Baseline (DP)
SYNC_EVERY_N_STEPS=(1)
for SYNC_EVERY_N_STEPS in ${SYNC_EVERY_N_STEPS[@]}; do
    torchrun --nproc_per_node 2 src/train.py \
        --swarm.num_stages 1 \
        --swarm.sync_every_n_steps $SYNC_EVERY_N_STEPS \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/fineweb-edu-1bt.toml \
        --data.subset_size 0.1 \
        --train.micro_batch_size 16 \
        --train.step_timeout 30 \
        --train.scheduler.enable true \
        --eval.every_n_steps 10 \
        --eval.max_steps 10 \
        --sample.enable true \
        --sample.every_n_steps 50 \
        --logging.wandb.enable true \
        --logging.wandb.tags "$TAGS,DP"
done