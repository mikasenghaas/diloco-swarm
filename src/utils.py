import os
import math

import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from itertools import cycle as cycle_iter
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW

from src.config import LoggingConfig, ModelConfig, DataConfig, OptimizerConfig, SchedulerConfig
from src.logger import CustomLogger
from src.metrics import Metrics
from src.world import World
from src.model import GPT2, GPT2Config

from typing import Optional, List, Dict, Tuple, Any
    
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_persistent_dir() -> str | None:
    # Load env variable from ~/.env file
    load_dotenv(os.path.expanduser("~/.env"))
    persistent_dir = os.getenv("PERSISTENT_DIR")
    
    return persistent_dir

HF_CACHE_DIR = os.path.join(get_persistent_dir(), "huggingface")

def get_world() -> Tuple[int, int]:
    msg = "Try running with `torchrun --nproc_per_node <num_gpus> <script>.py`"
    assert "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ, f"LOCAL_RANK and WORLD_SIZE environment variable is not set. {msg}"
    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    return local_rank, world_size

def get_device(local_rank: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    else:
        raise RuntimeError("No CUDA device available.")

def get_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)

def get_logger(logging: LoggingConfig, name: Optional[str] = None, run_id: Optional[str] = None) -> CustomLogger:
    return CustomLogger(logging, name, run_id)

def get_model(model_config: ModelConfig) -> GPT2:
    return GPT2(GPT2Config(**model_config.dict()))

# def get_sharded_model(model: AutoModelForCausalLM, world: World, model_type: ModelType) -> ShardedModel:
#     match model_type:
#         case ModelType.LLAMA:
#             return ShardedLlamaModel(model, world)
#         case ModelType.GPT:
#             return ShardedGPTModel(model, world)
#         case _:
#             raise ValueError(f"Unknown model type: {model_type}")

def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("openai-community/gpt2", fast=True, cache_dir=HF_CACHE_DIR)

def get_optimizer(model: nn.Module | AutoModelForCausalLM, optimizer_config: OptimizerConfig) -> AdamW:
    return AdamW(model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay, betas=optimizer_config.betas)

def get_scheduler(optimizer: AdamW, num_steps: int, scheduler_config: SchedulerConfig) -> LambdaLR:
    def lr_lambda(step, warmup_steps, num_steps, num_cycles, min_lr_factor):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine_decay
    if scheduler_config.enable:
        return LambdaLR(optimizer, lambda step: lr_lambda(step, scheduler_config.num_warmup_steps, num_steps, scheduler_config.num_cycles, scheduler_config.min_lr_factor), last_epoch=scheduler_config.last_epoch)
    return LambdaLR(optimizer, lambda _: 1)

def get_dataset(data_config: DataConfig, split: str | None = None) -> Dataset:
    datadict = load_dataset(data_config.path, data_config.name, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
    if split is None:
        return datadict
    return datadict[split]

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, cycle: bool = True) -> DataLoader:
    def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        batch_attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        return {
            "input_ids": batch_input_ids[:, :-1].contiguous(),
            "target_ids": batch_input_ids[:, 1:].contiguous(),
            "attention_mask": batch_attention_mask[:, :-1].contiguous(),
            "hidden_states": None
        }
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
    return cycle_iter(dataloader) if cycle else iter(dataloader)

def get_micro_dataloader(batch: Dict[str, torch.Tensor], micro_batch_size: int) -> DataLoader:
    """Create a DataLoader for micro-batches from a single large batch."""
    class MicroBatchDataset(Dataset):
        def __init__(self, input_ids, target_ids, attention_mask):
            self.input_ids = input_ids
            self.target_ids = target_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'target_ids': self.target_ids[idx],
                'attention_mask': self.attention_mask[idx],
            }

    dataset = MicroBatchDataset(batch['input_ids'], batch["target_ids"], batch["attention_mask"])
    return DataLoader(dataset, batch_size=micro_batch_size, shuffle=False)

def tokenize(sample: str, tokenizer: AutoTokenizer, max_length: int | None = None, return_tensors: str | None = "pt") -> Dict[str, Any]:
    if max_length is None:
        return tokenizer(sample, return_tensors=return_tensors)
    return tokenizer(sample, truncation=True, padding="max_length", max_length=max_length+1, return_tensors=return_tensors)

def get_train_pbar_description(metrics: Metrics, prefix: str):
    curr_metrics = metrics.compute()
    step = curr_metrics.get(f"{metrics.name}/step/current")
    micro_time = curr_metrics.get(f"{metrics.name}/micro_time/current")
    loss = curr_metrics.get(f"{metrics.name}/loss/current")
    norm = curr_metrics.get(f"{metrics.name}/norm/current")
    norm = 0 if norm is None else norm
    perplexity = curr_metrics.get(f"{metrics.name}/perplexity/current")
    throughput = curr_metrics.get(f"{metrics.name}/throughput/current")
    return f"{prefix} Step: {step} - Time: {micro_time*1000:.1f}ms - Loss: {loss:.4f} - Norm: {norm:.4f} - Perplexity: {perplexity:.1f} - Throughput: {throughput:.1f}"

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

def get_train_setup(steps: int, batch_size: int, seq_length: int, micro_batch_size: int, num_examples: int):
    tokens_per_step = batch_size * seq_length
    total_tokens = steps * tokens_per_step
    avg_token_repetitions = total_tokens / (num_examples * seq_length)
    grad_accumulation_steps = batch_size // micro_batch_size if micro_batch_size > 0 else 1

    return {
        "steps": steps,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "tokens_per_step": tokens_per_step,
        "total_tokens": total_tokens,
        "avg_token_repetitions": avg_token_repetitions,
        "grad_accumulation_steps": grad_accumulation_steps,
    }

def format_int(num: int, prec: int = 2) -> str:
    if num < 1e3:
        return str(num)
    if num < 1e6:
        return f"{num/1e3:.{prec}f}K"
    if num < 1e9:
        return f"{num/1e6:.{prec}f}M"
    else:
        return f"{num/1e9:.{prec}f}B"

def format_float(num: float, prec: int = 2) -> str:
    return f"{num:.{prec}f}"