#!/bin/bash

set -e

TAGS="Test,Main"
CMD="torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 1 \
    --model @configs/model/gpt2-small.toml \
    --data @configs/data/fineweb-edu-10bt.toml \
    --train.inner_optimizer @configs/optimizer/adamw.toml \
    --train.outer_optimizer @configs/optimizer/none.toml \
    --train.inner_optimizer.lr 4e-4 \
    --train.scheduler.enable true \
    --train.scheduler.num_warmup_steps 50 \
    --train.max_steps 2000 \
    --train.batch_size 2048 \
    --data.seq_length 1024 \
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 50 \
    --eval.eval_size 0.001 \
    --sample.enable false \
    --sample.every_n_steps 100 \
    --logging.wandb.enable false \
    --logging.wandb.tags '$TAGS,SWARM'"

echo $CMD; eval $CMD;
