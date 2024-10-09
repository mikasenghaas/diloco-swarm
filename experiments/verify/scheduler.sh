#!/bin/bash

set -e
GROUP="verify/scheduler"

# Run 1: scheduler.enable=false
# Run 3: scheduler.enable=true
# Run 2: scheduler.enable=true, scheduler.warmup_steps=50
# Run 4: scheduler.enable=true, scheduler.warmup_steps=50, scheduler.min_lr_factor=0.5

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.scheduler.enable true \

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.scheduler.enable true \
    --train.scheduler.warmup_steps 50

python src/train/baseline.py @configs/baseline/debug.toml \
    --logging.file.enable true \
    --logging.wandb.enable true \
    --logging.wandb.group $GROUP \
    --train.scheduler.enable true \
    --train.scheduler.warmup_steps 50 \
    --train.scheduler.min_lr_factor 0.5