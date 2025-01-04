#!/bin/bash

# This script is used for quick testing of experiment setup, e.g. tune (micro-) batch size, etc.

TAGS="Test"
CMD="torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 50 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/fineweb-edu-10bt.toml \
    --train.inner_optimizer @configs/optimizer/adamw.toml \
    --train.outer_optimizer @configs/optimizer/nesterov.toml \
    --train.inner_optimizer.lr 4e-4 \
    --train.outer_optimizer.lr 0.7 \
    --train.scheduler.enable true \
    --train.scheduler.num_warmup_steps 50 \
    --train.max_steps 2000 \
    --train.batch_size 2048 \
    --data.seq_length 1024 \
    --train.micro_batch_size 8 \
    --train.max_micro_batches 16 \
    --eval.enable true \
    --eval.every_n_steps 50 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 100 \
    --logging.wandb.enable false \
    --logging.wandb.tags $TAGS"

echo $CMD; eval $CMD;
