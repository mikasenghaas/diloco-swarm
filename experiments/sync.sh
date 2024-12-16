#!/bin/bash

set -e
TAGS="Sync"

SYNC_EVERY_N_STEPS=(1 400)

# DP
for SYNC_EVERY_N_STEPS in "${SYNC_EVERY_N_STEPS[@]}"
do
    torchrun --nproc_per_node 2 src/train.py \
        @configs/debug.toml \
        --swarm.num_stages 1 \
        --swarm.sync_every_n_steps $SYNC_EVERY_N_STEPS \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/wikitext.toml \
        --logging.wandb.enable false \
        --logging.wandb.tags "$TAGS,DP"
done