#!/bin/bash

set -e

TAGS="Test"
CMD="torchrun --nproc_per_node 1 src/train.py \
        --swarm.num_stages 1 \
        --swarm.sync_every_n_steps 1 \
        --model @configs/model/gpt2-tiny.toml \
        --data @configs/data/memorize.toml \
        --train.inner_optimizer @configs/optimizer/adamw.toml \
        --train.outer_optimizer @configs/optimizer/nesterov.toml \
        --train.inner_optimizer.lr 6e-3 \
        --train.outer_optimizer.lr 0.7 \
        --data.seq_length 128 \
        --train.max_epochs 40 \
        --train.batch_size 1 \
        --train.micro_batch_size 1 \
        --amp.enable false \
        --eval.enable false \
        --eval.every_n_steps 5 \
        --sample.enable true \
        --sample.every_n_steps -1 \
        --logging.wandb.enable false \
        --device cpu"

echo $CMD; eval $CMD;
