#!/bin/bash

set -e

TAGS="Test"

CMD="torchrun --nproc_per_node 1 src/train.py \
        --swarm.num_stages 1 \
        --model @configs/model/gpt2-tiny.toml \
        --data @configs/data/memorize.toml \
        --data.seq_length 128 \
        --train.optimizer.lr 0.004 \
        --train.max_epochs 50 \
        --train.batch_size 1 \
        --train.micro_batch_size 1 \
        --amp.dtype float32 \
        --amp.precision highest \
        --eval.enable false \
        --sample.enable true \
        --logging.wandb.enable false \
        --device cpu"

echo $CMD; eval $CMD;
