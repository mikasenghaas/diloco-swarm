import os
import json
import torch
from typing import Tuple, Optional

import wandb
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.config import BaseConfig
from src.model import GPT2, GPT2Config

class Checkpoint:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, step: int, model: GPT2) -> str:
        step_dir = self._get_step_dir(step)
        os.makedirs(step_dir, exist_ok=True)
        model.save_pretrained(step_dir)

        return step_dir

    def load(self, step: int) -> GPT2:
        # Get step directory
        step_dir = self._get_step_dir(step)

        if not os.path.exists(step_dir):
            raise ValueError(f"Checkpoint at step {step} does not exist")

        return self.load_from_dir(step_dir)

    def load_from_dir(self, ckpt_dir: str) -> GPT2:
        return GPT2.from_pretrained(ckpt_dir)

    def get_latest_step(self) -> Optional[int]:
        steps = [int(d.split("_")[1]) for d in os.listdir(self.base_dir) if d.startswith("step_")]
        return max(steps) if steps else None

    def _write_config(self, config: BaseConfig) -> str:
        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            f.write(config.model_dump_json())

    def _load_config(self) -> BaseConfig:
        with open(os.path.join(self.base_dir, "config.json"), "r") as f:
            return BaseConfig.model_validate_json(f.read())

    def _get_step_dir(self, step: int) -> str:
        return os.path.join(self.base_dir, f"step_{step}")
