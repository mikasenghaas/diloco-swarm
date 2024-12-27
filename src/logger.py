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
    WARNING = logging.WARNING
    ERROR = logging.ERROR

class Logger:
    """Automatically log to console, file, and wandb based on world configuration."""
    def __init__(self, world: World, config: LoggingConfig):
        self.world, self.config = world, config
        self.file_logger, self.console_logger, self.wandb_run = None, None, None
        self.master_file_logger, self.master_console_logger, self.master_wandb_run = None, None, None
        self.name = world.local_rank
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

        # Set up file logging
        self.file_logger = logging.getLogger(f'file-{self.name}')
        self.file_logger.setLevel(self.config.file.log_level)
        file_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.name}.log"))
        file_handler.setFormatter(logging.Formatter(f'[%(levelname)s][{self.name}] %(asctime)s %(message)s'))
        self.file_logger.addHandler(file_handler)

        # Set up console logging
        self.console_logger = logging.getLogger(f'console-{self.name}')
        self.console_logger.setLevel(self.config.console.log_level)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(f'[%(levelname)s][{self.name}] %(message)s'))
        self.console_logger.addHandler(stream_handler)

        # Set up wandb logging
        if self.config.wandb.enable:
            self.wandb_run = wandb.init(
                name=self.name, # 0, 1, ...
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                tags=self.config.wandb.tags,
                group=self.run_id,
                dir=self.log_dir
            )

        # Setup file and remote logging for master
        if self.is_master:
            # File logging
            self.master_file_logger = logging.getLogger(f'master-file')
            self.master_file_logger.setLevel(self.config.file.log_level)
            formatter = logging.Formatter(f'[%(levelname)s][*] %(message)s')
            file_handler = logging.FileHandler(os.path.join(self.log_dir, f"master.log"))
            file_handler.setFormatter(formatter)
            self.master_file_logger.addHandler(file_handler)

            # Console logging
            self.master_console_logger = logging.getLogger(f'master-console')
            self.master_console_logger.setLevel(self.config.console.log_level)
            formatter = logging.Formatter(f'[%(levelname)s][*] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.master_console_logger.addHandler(stream_handler)

            # Wandb logging
            if self.config.wandb.enable:
                self.master_wandb_run = wandb.init(
                    name="master",
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
            return
        if self.console_logger: self.console_logger.log(msg=message, level=level.value)

    def log_config(self, config: BaseConfig) -> None:
        config_dict = config.model_dump()
        config_str = "Setting configuration:\n\t" + yaml.dump(config_dict, sort_keys=False).replace("\n", "\n\t").strip()
        self.log_message(config_str, master=True, level=Level.INFO)
        if self.wandb_run: self.wandb_run.config.update(config_dict)
        if self.master_wandb_run: self.master_wandb_run.config.update(config_dict)

    def log_world(self, world: World) -> None:
        self.log_message(world, master=False, level=Level.INFO)
        if self.wandb_run: wandb.config.update({"world": dict(world)})

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None, master: bool = True) -> None:
        metrics_str = f"{'Global' if master else 'Local'} metrics: " + str(metrics)
        self.log_message(metrics_str, master=master, level=Level.DEBUG)
        if master and self.master_wandb_run: self.master_wandb_run.log(metrics, step=step); return
        if self.wandb_run: self.wandb_run.log(metrics, step=step)

    def log_samples(self, samples: List[str], step: int) -> None:
        for i, sample in enumerate(samples):
            self.log_message(f"Sample {i+1}: {sample}", master=True, level=Level.INFO)

        sample_path = os.path.join(self.samples_dir, f"{step}.txt")
        with open(sample_path, "w") as f:
            for sample in samples:
                f.write(sample + "\n")

        # TODO: Save samples to wandb

    def close(self) -> None:
        if self.wandb_run: self.wandb_run.finish()
        if self.master_wandb_run: self.master_wandb_run.finish()
