import os
import yaml
import logging
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime

import torch.nn as nn
from transformers import AutoTokenizer
from pydantic_config import BaseConfig
import wandb

from src.config import LoggingConfig
from src.ckpt import Checkpoint
from src.model import GPT2

class Level(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    ERROR = logging.ERROR

class CustomLogger:
    """A logger that logs to console, file, and wandb."""
    def __init__(self, config: LoggingConfig, name: Optional[str] = None, run_id: Optional[str] = None):
        self.config = config
        self.console_logger = None
        self.file_logger = None
        self.wandb_run = None
        self.log_dir = None
        self.checkpoint_dir = None

        self.name = name if name else "master"
        self.run_id = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup()

        self.checkpoint = Checkpoint(self.checkpoint_dir)

    def setup(self):
        # Create log directory
        self.log_dir = os.path.join(self.config.log_dir, self.run_id)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.samples_dir = os.path.join(self.log_dir, "samples")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Set up wandb logging
        if self.config.wandb.enable:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                group=self.config.wandb.group,
                name=self.config.wandb.run_name,
                dir=self.log_dir
            )

        # Set up console logging
        if self.config.console.enable:
            self.console_logger = logging.getLogger(f'console-{self.name}')
            self.console_logger.setLevel(self.config.console.log_level)
            formatter = logging.Formatter(f'[%(levelname)s][{self.name}] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.console_logger.addHandler(stream_handler)

        # Set up file logging
        if self.config.file.enable:
            self.file_logger = logging.getLogger(f'file-{self.name}')
            self.file_logger.setLevel(self.config.file.log_level)
            formatter = logging.Formatter(f'[%(levelname)s][{self.name}] %(asctime)s %(message)s')
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.name}.log"))
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)

    def log_message(self, message: str, level: Level = Level.INFO) -> None:
        if self.console_logger:
            self.console_logger.log(level=level.value, msg=message)
        if self.file_logger:
            self.file_logger.log(level=level.value, msg=message)

    def log_config(self, config: BaseConfig, level: Level = Level.INFO) -> None:
        config_dict = config.model_dump()
        yaml_str = yaml.dump(config_dict, sort_keys=False)
        config_str = "Setting configuration:\n\t" + yaml_str.replace("\n", "\n\t").strip()
        self.log_message(config_str, level=level)
        if self.wandb_run:
            wandb.config.update(config_dict)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None, level: Level = Level.INFO) -> None:
        self.log_message(metrics, level=level)
        if self.wandb_run:
            wandb.log(metrics, step=step)

    def log_checkpoint(self, step: int, model: GPT2, level: Level = Level.INFO) -> None:
        checkpoint_dir = self.checkpoint.save(step, model)
        self.log_message(f"Saved model checkpoint at step {step}", level=level)

    def log_samples(self, step: int, samples: List[str], level: Level = Level.INFO) -> None:
        for i, sample in enumerate(samples):
            self.log_message(f"Sample {i+1}: {sample}", level=level)

        sample_path = os.path.join(self.samples_dir, f"{step}.txt")
        with open(sample_path, "w") as f:
            for sample in samples:
                f.write(sample + "\n")

    def close(self) -> None:
        if self.wandb_run:
            wandb.finish()