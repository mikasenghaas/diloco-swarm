#!/bin/bash

# This script is used for quick testing of experiment setup, e.g. tune (micro-) batch size, etc.

TAGS="Test"
CMD="torchrun --nproc_per_node 2 src/train.py\
    --swarm.num_stages 1\
    --device cpu\
    --model @configs/model/gpt2-tiny.toml\
    --data @configs/data/memorize.toml\
    --data.tokenize true\
    --train.inner_optimizer @configs/optimizer/adamw.toml\
    --train.outer_optimizer @configs/optimizer/none.toml\
    --train.inner_optimizer.lr 0.006\
    --data.seq_length 128\
    --train.max_steps 40\
    --train.batch_size 1\
    --train.micro_batch_size 1\
    --logging.log_dir /tmp/swarm/test_memorize \
    --logging.run_id run_id\
    --amp.enable false\
    --eval.enable false\
    --logging.wandb.enable false"

echo $CMD; eval $CMD;
