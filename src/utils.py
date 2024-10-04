import math
import time
from functools import wraps
import torch
import numpy as np
from itertools import cycle as cycle_iter
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

from .config import ModelConfig, TokenizerConfig, DataConfig, LoggingConfig, SchedulerConfig, OptimizerConfig, TrainConfig
from .logger import CustomLogger
from .metrics import Metrics

from typing import List, Dict, Any

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_logger(logging: LoggingConfig) -> CustomLogger:
    return CustomLogger(logging)

def get_model(model: ModelConfig) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model.name)

def get_tokenizer(tokenizer: TokenizerConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer.name, fast=tokenizer.fast)

def get_optimizer(train: TrainConfig, model: AutoModelForCausalLM) -> AdamW:
    return AdamW(model.parameters(), lr=train.optimizer.lr, weight_decay=train.optimizer.decay, betas=train.optimizer.betas)

def get_scheduler(train: TrainConfig, optimizer: AdamW, num_steps: int) -> LambdaLR:
    if train.scheduler.enable:
        def lr_lambda(step, warmup_steps, num_steps, num_cycles):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return LambdaLR(optimizer, lambda step: lr_lambda(step, train.scheduler.warmup_steps, num_steps, train.scheduler.num_cycles), last_epoch=train.scheduler.last_epoch)
    return LambdaLR(optimizer, lambda _: 1)

def get_dataset(data: DataConfig, split: str | None = None) -> Dataset:
    dataset = load_dataset(data.path, data.name, split=split)
    if split == "train" and data.subset_size < 1.0:
        return dataset.shuffle(seed=42).select(range(int(len(dataset) * data.subset_size)))
    return dataset

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, cycle: bool = True) -> DataLoader:
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.stack([torch.tensor(example['input_ids'][:-1]) for example in batch]),
            'attention_mask': torch.stack([torch.tensor(example['attention_mask'][:-1]) for example in batch]),
            'labels': torch.stack([torch.tensor(example['input_ids'][1:]) for example in batch]),
        }
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return cycle_iter(dataloader) if cycle else iter(dataloader)

def non_empty_text(examples: Dict[str, Any]) -> bool:
    return examples["text"] != ""

def non_headline(examples: Dict[str, Any]) -> bool:
    return not examples["text"].startswith(" = ")

def tokenize(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length+1)

def get_train_pbar_description(metrics: Metrics, prefix: str):
    curr_metrics = metrics.compute()
    step = curr_metrics.get(f"{metrics.name}/step/current")
    loss = curr_metrics.get(f"{metrics.name}/loss/average")
    perplexity = curr_metrics.get(f"{metrics.name}/perplexity/average")
    throughput = curr_metrics.get(f"{metrics.name}/throughput/current")
    return f"{prefix} Step: {step}, Avg. Loss: {loss:.4f}, Avg. Perplexity: {perplexity:.1f}, Throughput: {throughput:.1f}"

def get_eval_pbar_description(metrics: Metrics, prefix: str):
    curr_metrics = metrics.compute()
    loss = curr_metrics.get(f"{metrics.name}/loss/average")
    perplexity = curr_metrics.get(f"{metrics.name}/perplexity/average")
    return f"{prefix} Avg. Loss: {loss:.4f}, Avg. Perplexity: {perplexity:.1f}"

def get_num_steps(max_steps: int, max_epochs: int, num_examples: int, batch_size: int) -> int:
    """Get number of steps to train/val/test; whatever is reached first max_steps or max_epochs"""
    assert max_steps > 0 or max_epochs > 0, "Specify at least one of `max_steps` and `max_epochs`"
    max_steps_epoch = num_examples // batch_size * max_epochs
    if max_epochs == -1: return max_steps
    elif max_steps == -1: return max_steps_epoch
    else: return min(max_steps, max_steps_epoch)