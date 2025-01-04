#!/bin/bash

# Experiment 2: Ablation on the communication frequency of DiLoCo-SWARM with synchronization every {200, 100, 50, 20, 10} steps.

# NB: Tune batch size and micro batch size to your hardware for best performance (Here, they are tuned for H100 80GB)
# NB: We can only evaluate at most every global step, so we set `eval.every_n_steps` to max(50, swarm.sync_every_n_steps) 

TAGS="Experiment 2"

# DiLoCo-SWARM, H=200
torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 200 \
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
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 200 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 200 \
    --logging.wandb.enable true \
    --logging.wandb.tags $TAGS

# DiLoCo-SWARM, H=100
torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 100 \
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
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 100 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 100 \
    --logging.wandb.enable true \
    --logging.wandb.tags $TAGS

# DiLoCo-SWARM, H=50
torchrun --nproc_per_node 8 src/train.py \
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
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 50 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 100 \
    --logging.wandb.enable true \
    --logging.wandb.tags $TAGS

# DiLoCo-SWARM, H=20
torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 20 \
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
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 50 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 100 \
    --logging.wandb.enable true \
    --logging.wandb.tags $TAGS

# DiLoCo-SWARM, H=10
torchrun --nproc_per_node 8 src/train.py \
    --swarm.num_stages 2 \
    --swarm.sync_every_n_steps 10 \
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
    --train.micro_batch_size 32 \
    --train.max_micro_batches 12 \
    --eval.enable true \
    --eval.every_n_steps 50 \
    --eval.eval_size 0.001 \
    --sample.enable true \
    --sample.every_n_steps 100 \
    --logging.wandb.enable true \
    --logging.wandb.tags $TAGS

# DiLoCo-SWARM, H=1 (Did not converge, probably do high learning rate)