#!/bin/bash

TAGS="Experiment 4"

# Baseline (Single GPU), GPT-2 Tiny
# torchrun --nproc_per_node 8 src/train.py \
#     --swarm.num_stages 1 \
#     --swarm.sync_every_n_steps 1 \
#     --model @configs/model/gpt2-tiny.toml \
#     --data @configs/data/fineweb-edu-10bt.toml \
#     --train.inner_optimizer @configs/optimizer/adamw.toml \
#     --train.outer_optimizer @configs/optimizer/none.toml \
#     --train.inner_optimizer.lr 4e-4 \
#     --train.scheduler.enable true \
#     --train.scheduler.num_warmup_steps 50 \
#     --train.max_steps 2000 \
#     --train.batch_size 512 \
#     --data.seq_length 1024 \
#     --train.micro_batch_size 32 \
#     --eval.enable true \
#     --eval.every_n_steps 50 \
#     --eval.eval_size 0.001 \
#     --sample.enable true \
#     --sample.every_n_steps 100 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS"

# SWARM-DiLoCo, GPT-2 Tiny
# torchrun --nproc_per_node 8 src/train.py \
#     --swarm.num_stages 2 \
#     --swarm.sync_every_n_steps 50 \
#     --model @configs/model/gpt2-tiny.toml \
#     --data @configs/data/fineweb-edu-10bt.toml \
#     --train.inner_optimizer @configs/optimizer/adamw.toml \
#     --train.outer_optimizer @configs/optimizer/nesterov.toml \
#     --train.inner_optimizer.lr 4e-4 \
#     --train.outer_optimizer.lr 0.7 \
#     --train.scheduler.enable true \
#     --train.scheduler.num_warmup_steps 50 \
#     --train.max_steps 2000 \
#     --train.batch_size 2048 \
#     --data.seq_length 1024 \
#     --train.micro_batch_size 32 \
#     --train.max_micro_batches 12 \
#     --eval.enable true \
#     --eval.every_n_steps 50 \
#     --eval.eval_size 0.001 \
#     --sample.enable true \
#     --sample.every_n_steps 100 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS"

# SWARM-DiLoCo, GPT-2 Small
# torchrun --nproc_per_node 8 src/train.py \
#     --swarm.num_stages 2 \
#     --swarm.sync_every_n_steps 50 \
#     --model @configs/model/gpt2-small.toml \
#     --data @configs/data/fineweb-edu-10bt.toml \
#     --train.inner_optimizer @configs/optimizer/adamw.toml \
#     --train.outer_optimizer @configs/optimizer/nesterov.toml \
#     --train.inner_optimizer.lr 4e-4 \
#     --train.outer_optimizer.lr 0.7 \
#     --train.scheduler.enable true \
#     --train.scheduler.num_warmup_steps 50 \
#     --train.max_steps 2000 \
#     --train.batch_size 2048 \
#     --data.seq_length 1024 \
#     --train.micro_batch_size 32 \
#     --train.max_micro_batches 12 \
#     --eval.enable true \
#     --eval.every_n_steps 50 \
#     --eval.eval_size 0.001 \
#     --sample.enable true \
#     --sample.every_n_steps 100 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS,SWARM-DiLoCo"

# GPT-2 Medium Baseline
# torchrun --nproc_per_node 8 src/train.py \
#     --swarm.num_stages 1 \
#     --swarm.sync_every_n_steps 1 \
#     --model @configs/model/gpt2-medium.toml \
#     --data @configs/data/fineweb-edu-10bt.toml \
#     --train.inner_optimizer @configs/optimizer/adamw.toml \
#     --train.outer_optimizer @configs/optimizer/none.toml \
#     --train.inner_optimizer.lr 4e-4 \
#     --train.scheduler.enable true \
#     --train.scheduler.num_warmup_steps 50 \
#     --train.max_steps 2000 \
#     --train.batch_size 512 \
#     --data.seq_length 1024 \
#     --train.micro_batch_size 32 \
#     --eval.enable true \
#     --eval.every_n_steps 50 \
#     --eval.eval_size 0.001 \
#     --sample.enable true \
#     --sample.every_n_steps 100 \
#     --logging.wandb.enable true \
#     --logging.wandb.tags "$TAGS"

# GPT-2 Medium
torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 50 \
    --model @configs/model/gpt2-medium.toml \
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
    --logging.wandb.enable true \
    --logging.wandb.tags "$TAGS"