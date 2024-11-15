#!/bin/bash

set -e
TAGS="pipeline,ckpt"

torchrun --nproc_per_node 2 src/train/pipeline.py @configs/debug.toml \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --sample.enable true \
    --eval.every_n_steps -1 \
    --train.max_steps 100 \
    --train.batch_size 1 \
    --train.micro_batch_size 1 \
    --logging.console.enable false \
    --logging.ckpt.enable true
    # --logging.wandb.enable true \
    # --logging.wandb.tags $TAGS
