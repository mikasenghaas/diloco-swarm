"""
Pipeline Parallel LLM Pre-Training.

Run with:
```
#VERBOSE=1 torchrun --nproc_per_node 2 src/train/pipeline.py @configs/debug.toml --model @configs/model/llama2-9m.toml --data @configs/data/wikitext.toml
```
"""
import autorootcwd

import os
import time
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import disable_progress_bar
from tqdm import tqdm

from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model
from src.world import World
from src.metrics import Outputs
from pydantic_config import BaseConfig, parse_argv

class PipelineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    sample: SampleConfig
    logging: LoggingConfig

def train() -> Outputs:
    pass

def eval() -> Outputs:
    pass

def sample() -> List[str]:
    pass

def main(config: PipelineConfig):
    # Set precision and seed
    seed_everything(config.train.seed)

    # Initialize world
    local_rank, world_size = get_world()
    world = World(local_rank, world_size)

    # Synchronize processes to ensure consistent timestamp
    dist.barrier()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get logger
    logger_name = f"{local_rank}" if world_size > 1 else "master"
    logger = get_logger(config.logging, logger_name, run_id)
    logger.log_config(config)
    logger.log_message(str(world))

    # Set device
    torch.cuda.set_device(local_rank)
    device = get_device(local_rank)
    logger.log_message(f"Using device: {device}")

    # Get model

    # Get data

    # Destroy process group
    if world.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    disable_progress_bar()
    main(PipelineConfig(**parse_argv()))