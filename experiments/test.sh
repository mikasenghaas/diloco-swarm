#!/bin/bash

set -e

TAGS="Test"
CMD="torchrun --nproc_per_node 9 src/train.py \
        --swarm.num_stages 3 \
        --model @configs/model/gpt2-tiny.toml \
        --data @configs/data/memorize.toml \
        --data.seq_length 128 \
        --train.optimizer.lr 0.006 \
        --train.max_epochs 35 \
        --train.batch_size 1 \
        --train.micro_batch_size 1 \
        --amp.enable false \
        --eval.enable false \
        --sample.enable true \
        --logging.wandb.enable false \
        --device cpu"

echo $CMD; eval $CMD;
