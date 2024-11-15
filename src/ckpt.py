import os
from typing import Optional

import torch
import torch.nn as nn

from src.config import BaseConfig
from src.world import World

class Checkpoint:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.shard_id = None
        os.makedirs(base_dir, exist_ok=True)

    def setup(self, world: World):
        if world.world_size > 1:
            self.shard_id = world.local_rank

    def save(self, step: int, model: nn.Module) -> str:
        save_dir = self._get_save_dir(step)
        model_name = self._get_model_name()
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(save_dir, model_name))

        return os.path.join(save_dir, model_name)

    def load(self, step: int) -> nn.Module:
        save_dir = self._get_save_dir(step)

        if not os.path.exists(save_dir):
            raise ValueError(f"Checkpoint at step {step} does not exist")

        return self.load_from_dir(save_dir)

    def load_from_dir(self, save_dir: str) -> nn.Module:
        model_name = self._get_model_name()
        return torch.load(os.path.join(save_dir, model_name), weights_only=True)

    def get_latest_step(self) -> Optional[int]:
        steps = [int(d.split("_")[1]) for d in os.listdir(self.base_dir) if d.startswith("step_")]
        return max(steps) if steps else None

    def _get_save_dir(self, step: int) -> str:
        return os.path.join(self.base_dir, f"step_{step}")
    
    def _get_model_name(self) -> str:
        model_name = "model"
        model_name += f".{self.shard_id}" if self.shard_id is not None else ""
        model_name += ".pt"
        return model_name