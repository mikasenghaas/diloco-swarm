import os
import yaml
import logging
from typing import Dict, List
from enum import Enum
from datetime import datetime

import wandb
from pydantic_config import BaseConfig

from src.config import LoggingConfig
from src.world import World

class Level(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

class Logger:
    """Automatically log to console, file, and wandb based on world configuration."""
    def __init__(self, world: World, config: LoggingConfig):
        self.world, self.config = world, config
        self.file_logger, self.console_logger, self.wandb_run = None, None, None
        self.master_file_logger, self.master_console_logger = None, None
        self.name = f"rank{world.local_rank}"
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if config.run_id is None else config.run_id
        self.is_master = world.is_master
        self.setup()

    def setup(self):
        # Create log directory
        self.log_dir = os.path.join(self.config.log_dir, self.run_id)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.samples_dir = os.path.join(self.log_dir, "samples")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Set up local logging
        self.file_logger = self._setup_file_logger(f'file-{self.name}', self.world.local_rank, self.config.file.log_level)
        self.console_logger = self._setup_console_logger(f'console-{self.name}', self.world.local_rank, self.config.console.log_level)

        # Set up master logging
        if self.is_master:
            self.master_file_logger = self._setup_file_logger(f'master-file', "master", self.config.file.log_level)
            self.master_console_logger = self._setup_console_logger(f'master-console', "*", self.config.console.log_level)

        # Set up wandb logging
        if self.config.wandb.enable:
            self.wandb_run = wandb.init(
                name=self.name,
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                tags=self.config.wandb.tags,
                group=self.run_id,
                dir=self.log_dir
            )

    def log_message(self, message: str, master: bool, level: Level) -> None:
        if self.file_logger: self.file_logger.log(msg=message, level=level.value)
        if master:
            if self.master_console_logger: self.master_console_logger.log(msg=message, level=level.value)
            if self.master_file_logger: self.master_file_logger.log(msg=message, level=level.value)
        else:
            if self.console_logger: self.console_logger.log(msg=message, level=level.value)

    def log_config(self, config: BaseConfig) -> None:
        config_dict = config.model_dump()
        config_str = "Setting configuration:\n\t" + yaml.dump(config_dict, sort_keys=False).replace("\n", "\n\t").strip()
        self.log_message(config_str, master=True, level=Level.INFO)
        if self.wandb_run: self.wandb_run.config.update(config_dict)

    def log_world(self, world: World) -> None:
        self.log_message(world, master=False, level=Level.INFO)
        if self.wandb_run: wandb.config.update({"world": dict(world)})

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None, master: bool = True) -> None:
        metrics_str = f"{'Global' if master else 'Local'} metrics: " + str(metrics)
        self.log_message(metrics_str, master=master, level=Level.DEBUG)
        if self.wandb_run: self.wandb_run.log(metrics, step=step)

    def log_samples(self, samples: List[str], step: int) -> None:
        for i, sample in enumerate(samples):
            self.log_message(f"Sample {i+1}: {sample}", master=True, level=Level.INFO)

        sample_path = os.path.join(self.samples_dir, f"{step}.txt")
        with open(sample_path, "w") as f:
            for sample in samples:
                f.write(sample + "\n")

        # TODO: Save samples to wandb

    def _setup_file_logger(self, name: str, short_name: str, level: Level) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{short_name}.log"))
        file_handler.setFormatter(logging.Formatter(f'[%(levelname)s][{short_name}] %(message)s'))
        logger.addHandler(file_handler)
        return logger

    def _setup_console_logger(self, name: str, short_name: str, level: Level) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(f'[%(levelname)s][{short_name}] %(message)s'))
        logger.addHandler(stream_handler)
        return logger
