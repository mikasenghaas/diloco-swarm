import os
import yaml
import logging
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime

from pydantic_config import BaseConfig
import wandb

from src.config import LoggingConfig
from src.world import World

class Level(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    ERROR = logging.ERROR

class Logger:
    """Automatically log to console, file, and wandb based on world configuration."""
    def __init__(self, world: World, config: LoggingConfig):
        self.world, self.config = world, config
        self.file_logger, self.console_logger, self.master_logger, self.wandb_run = None, None, None, None
        self.name = f"{world.local_rank}" if world.world_size > 1 else "*"
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.is_master = world.is_leader and world.is_last_stage # Access to sampled tokens and loss
        self.setup()

    def setup(self):
        # Create log directory
        self.log_dir = os.path.join(self.config.log_dir, self.run_id)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.samples_dir = os.path.join(self.log_dir, "samples")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Set up file logging
        self.file_logger = logging.getLogger(f'file-{self.name}')
        self.file_logger.setLevel(self.config.file.log_level)
        formatter = logging.Formatter(f'[%(levelname)s][{self.name}] %(asctime)s %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.name}.log"))
        file_handler.setFormatter(formatter)
        self.file_logger.addHandler(file_handler)

        # Set up console logging on last stage leader
        self.console_logger = logging.getLogger(f'console-{self.name}')
        self.console_logger.setLevel(self.config.console.log_level)
        formatter = logging.Formatter(f'[%(levelname)s][{self.name}] %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.console_logger.addHandler(stream_handler)

        if self.is_master:
            self.master_logger = logging.getLogger(f'master')
            self.master_logger.setLevel(self.config.console.log_level)
            formatter = logging.Formatter(f'[%(levelname)s][*] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.master_logger.addHandler(stream_handler)

        # Set up wandb logging
        if self.config.wandb.enable:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                tags=self.config.wandb.tags,
                name=self.config.wandb.run_name,
                group=self.run_id,
                dir=self.log_dir
            )

    def log_message(self, message: str, level: Level = Level.INFO, master_only: bool = True) -> None:
        if self.file_logger: self.file_logger.log(level=level.value, msg=message)
        if master_only:
            if self.master_logger: self.master_logger.log(level=level.value, msg=message); return
            return
        self.console_logger.log(level=level.value, msg=message)

    def log_config(self, config: BaseConfig, level: Level = Level.INFO) -> None:
        config_dict = config.model_dump()
        yaml_str = yaml.dump(config_dict, sort_keys=False)
        config_str = "Setting configuration:\n\t" + yaml_str.replace("\n", "\n\t").strip()
        self.log_message(config_str, level=level, master_only=True)
        if self.wandb_run and self.is_master: wandb.config.update(config_dict)

    def log_world(self, world: World, level: Level = Level.INFO) -> None:
        self.log_message(world, level=level, master_only=False)
        if self.wandb_run and self.is_master: wandb.config.update({"world": dict(world)})

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None, level: Level = Level.INFO) -> None:
        self.log_message(metrics, level=level)
        if self.wandb_run and self.is_master: wandb.log(metrics, step=step)

    def log_samples(self, samples: List[str], step: int, level: Level = Level.INFO) -> None:
        for i, sample in enumerate(samples):
            self.log_message(f"Sample {i+1}: {sample}", level=level)

        sample_path = os.path.join(self.samples_dir, f"{step}.txt")
        with open(sample_path, "w") as f:
            for sample in samples:
                f.write(sample + "\n")

        # TODO: Save samples to wandb

    def close(self) -> None:
        if self.wandb_run:
            wandb.finish()