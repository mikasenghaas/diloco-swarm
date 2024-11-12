import os
import json
import torch
from typing import Tuple, Optional

import wandb
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import BaseConfig
from .model import Model

CheckpointReturn = Tuple[nn.Module, AutoTokenizer, BaseConfig]

class Checkpoint:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _get_step_dir(self, step: int) -> str:
        return os.path.join(self.base_dir, f"step_{step}")

    def save(self, step: int, model: torch.nn.Module, config: BaseConfig) -> str:
        step_dir = self._get_step_dir(step)
        os.makedirs(step_dir, exist_ok=True)

        # Save model
        torch.save(model.state_dict(), os.path.join(step_dir, "model.pt"))

        # Save config
        with open(os.path.join(step_dir, "config.json"), "w") as f:
            f.write(config.model_dump_json())

        return step_dir

    def load(self, step: int) -> CheckpointReturn:
        # Get step directory
        step_dir = self._get_step_dir(step)

        if not os.path.exists(step_dir):
            raise ValueError(f"Checkpoint at step {step} does not exist")

        return self.load_from_dir(step_dir)

    def load_from_dir(self, ckpt_dir: str) -> CheckpointReturn:
        # Load config
        with open(os.path.join(ckpt_dir, "config.json"), "r") as f:
            config = json.loads(f.read())
            
        # Load model
        model_name = config["model"]["name"]
        model = Model(AutoModelForCausalLM.from_pretrained(model_name))
        model.load_state_dict(torch.load(
            os.path.join(ckpt_dir, "model.pt"),
            weights_only=True
        ))

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def load_from_artifact(self, artifact_name: str) -> CheckpointReturn:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download(root=self.base_dir)
        return self.load_from_dir(artifact_dir)

    def get_latest_step(self) -> Optional[int]:
        steps = [int(d.split("_")[1]) for d in os.listdir(self.base_dir) if d.startswith("step_")]
        return max(steps) if steps else None
