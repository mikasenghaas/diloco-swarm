import os
import yaml
import logging
from typing import Dict
from enum import Enum
from datetime import datetime
from pydantic_config import BaseConfig

import wandb

from .config import LoggingConfig

class Level(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    ERROR = logging.ERROR

class CustomLogger:
    """A logger that logs to console, file, and wandb."""
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.console_logger = None
        self.file_logger = None
        self.wandb_run = None
        self.run_id = None
        self.setup()

    def setup(self):
        # Create log directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.config.log_dir, self.run_id)
        os.makedirs(log_dir, exist_ok=True)

        # Set up wandb logging
        if self.config.wandb.enable:
            self.wandb_run = wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                group=self.config.wandb.group,
                name=self.config.wandb.run_name,
                dir=log_dir
            )

        # Set up console logging
        if self.config.console.enable:
            self.console_logger = logging.getLogger('console')
            self.console_logger.setLevel(self.config.console.log_level)
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.console_logger.addHandler(stream_handler)

        # Set up file logging
        if self.config.file.enable:
            self.file_logger = logging.getLogger('file')
            self.file_logger.setLevel(self.config.file.log_level)
            formatter = logging.Formatter('[%(levelname)s]\t%(asctime)s\t%(message)s')
            file_handler = logging.FileHandler(os.path.join(log_dir, self.config.file.name))
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)

    def log_message(self, message: str, level: Level = Level.INFO):
        if self.console_logger:
            self.console_logger.log(level=level.value, msg=message)
        if self.file_logger:
            self.file_logger.log(level=level.value, msg=message)

    def log_config(self, config: BaseConfig, level: Level = Level.INFO):
        config_dict = config.model_dump()
        yaml_str = yaml.dump(config_dict, sort_keys=False)
        config_str = "Setting configuration:\n\t" + yaml_str.replace("\n", "\n\t").strip()
        self.log_message(config_str, level=level)
        if self.wandb_run:
            wandb.config.update(config_dict)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None, level: Level = Level.INFO):
        # metrics_str = "Logging metrics:\n\t" + yaml.dump(metrics, sort_keys=False).replace("\n", "\n\t").strip()
        self.log_message(metrics, level=level)
        if self.wandb_run:
            wandb.log(metrics, step=step)

    def close(self):
        if self.wandb_run:
            wandb.finish()