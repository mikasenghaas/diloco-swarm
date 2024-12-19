#!/bin/bash

set -e

TAGS="Test,Sync"

CMD="torchrun --nproc_per_node 2 src/train.py \
        --swarm.num_stages 1 \
        --model @configs/model/gpt2-small.toml \
        --data @configs/data/fineweb-edu-1bt.toml \
        --train.micro_batch_size 16 \
        --train.step_timeout 60 \
        --train.scheduler.enable true \
        --eval.enable true \
        --eval.every_n_steps 10 \
        --eval.max_steps 10 \
        --sample.enable true \
        --sample.every_n_steps 50 \
        --logging.wandb.enable true \
        --logging.wandb.project $TAGS"

echo $CMD; eval $CMD;