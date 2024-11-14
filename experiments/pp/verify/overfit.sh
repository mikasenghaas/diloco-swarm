#!/bin/bash

set -e
GROUP="pp/verify/overfit"

torchrun --nproc_per_node 2 src/train/pipeline.py @configs/debug.toml \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/memorize.toml \
    --logging.console.enable false \
    --logging.file.enable true \
    --eval.enable false \
    --sample.enable true \
    --train.max_steps 100 \
    --train.batch_size 1 \
    --train.micro_batch_size 1 
    # --logging.wandb.enable true \
    # --logging.wandb.group $GROUP
