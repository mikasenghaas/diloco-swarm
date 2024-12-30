import os
import copy
import contextlib
from typing import Optional, List, Dict, Tuple, Any, Generator

import torch
import torch.nn as nn
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer, AdamW, SGD

from src.config import LoggingConfig, ModelConfig, DataConfig, OptimizerConfig, SchedulerConfig
from src.logger import Logger
from src.metrics import Metrics
from src.world import World
from src.model import GPT2, ShardedGPT2, GPT2Config
from src.sampler import BatchData, BatchSampler

    
def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_hf_cache_dir() -> str:
    load_dotenv(os.path.expanduser("~/.env"))
    persistent_dir = os.getenv("PERSISTENT_DIR")
    return os.path.join(persistent_dir, "huggingface")

def get_device(device: Optional[str] = None, local_rank: Optional[int] = None) -> torch.device:
    if device: return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")

def get_dtype(dtype: str) -> torch.dtype:
    return getattr(torch, dtype)

def get_logger(world: World, logging: LoggingConfig) -> Logger:
    return Logger(world, logging)

def get_model(model_config: ModelConfig) -> GPT2:
    return GPT2(GPT2Config(**model_config.dict()))

def get_sharded_model(model: GPT2, world: World) -> ShardedGPT2:
    return ShardedGPT2(model, world)

def get_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", fast=True, trust_remote_code=True, clean_up_tokenization_spaces=True, cache_dir=get_hf_cache_dir())
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_optimizer(model: nn.Module | AutoModelForCausalLM, optimizer_config: OptimizerConfig) -> Optimizer:
    if optimizer_config.type == "AdamW":
        return AdamW(model.parameters(), lr=optimizer_config.lr, weight_decay=optimizer_config.weight_decay, betas=optimizer_config.betas)
    elif optimizer_config.type == "SGD":
        return SGD(model.parameters(), lr=optimizer_config.lr, momentum=optimizer_config.momentum, nesterov=optimizer_config.nesterov)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_config.type}")

# def get_scheduler(optimizer: Optimizer, num_steps: int, scheduler_config: SchedulerConfig) -> LambdaLR:
#     def lr_lambda(step, warmup_steps, num_steps, num_cycles, min_lr_factor):
#         if step < warmup_steps:
#             return step / max(1, warmup_steps)
#         progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
#         cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
#         return min_lr_factor + (1 - min_lr_factor) * cosine_decay
#     if scheduler_config.enable:
#         return LambdaLR(optimizer, lambda step: lr_lambda(step, scheduler_config.num_warmup_steps, num_steps, scheduler_config.num_cycles, scheduler_config.min_lr_factor), last_epoch=scheduler_config.last_epoch)
#     return LambdaLR(optimizer, lambda _: 1)

def get_scheduler(optimizer: Optimizer, scheduler_config: SchedulerConfig) -> LambdaLR:
    def lr_lambda(step, warmup_steps):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    if scheduler_config.enable:
        return LambdaLR(optimizer, lambda step: lr_lambda(step, scheduler_config.num_warmup_steps), last_epoch=scheduler_config.last_epoch)
    return LambdaLR(optimizer, lambda _: 1)


def get_dataset(data_config: DataConfig, split: str) -> Dataset:
    # Define a path for the processed dataset
    processed_path = os.path.join(get_hf_cache_dir(), data_config.path)
    try: # Try loading from disk
        datadict = load_from_disk(processed_path)
    except (FileNotFoundError, ValueError): # If not found, load from HF and save to disk
        datadict = load_dataset(data_config.path, trust_remote_code=True)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        datadict.save_to_disk(processed_path)
    dataset = datadict[split]
    if data_config.subset_size < 1.0:
        dataset = dataset.select(range(int(len(dataset) * data_config.subset_size)))
    return dataset

def tokenize(sample: str, tokenizer: AutoTokenizer, max_length: int | None = None, return_tensors: str | None = "pt") -> Dict[str, Any]:
    if max_length is None:
        return tokenizer(sample, return_tensors=return_tensors)
    return tokenizer(sample, truncation=True, padding="max_length", max_length=max_length, return_tensors=return_tensors)

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch], dim=0)
        batch_attention_mask = torch.stack([torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch], dim=0)

        return {
            "input_ids": batch_input_ids[:, :-1].contiguous(),
            "target_ids": batch_input_ids[:, 1:].contiguous(),
            "attention_mask": batch_attention_mask[:, :-1].contiguous(),
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, pin_memory=True, num_workers=4)

def get_micro_batches(batch: Dict[str, torch.Tensor], micro_batch_size: int, world: World) -> Generator:
    batch_data = BatchData(batch)
    for rank in world.first_stage_ranks:
        micro_sampler = BatchSampler(batch_data, rank=rank, ranks=world.first_stage_ranks, micro_batch_size=micro_batch_size)
        micro_dataloader = DataLoader(batch_data, batch_size=micro_batch_size, shuffle=False, sampler=micro_sampler)
        for local_micro_step, micro_batch in enumerate(micro_dataloader, start=1):
            yield rank, local_micro_step, micro_batch

def get_train_pbar_description(metrics: Metrics, prefix: str):
    curr_metrics = metrics.compute()
    step = curr_metrics.get(f"{metrics.name}/step/current")
    time = curr_metrics.get(f"{metrics.name}/time/current")
    loss = curr_metrics.get(f"{metrics.name}/loss/current")
    norm = curr_metrics.get(f"{metrics.name}/norm/current")
    norm = 0 if norm is None else norm
    perplexity = curr_metrics.get(f"{metrics.name}/perplexity/current")
    throughput = curr_metrics.get(f"{metrics.name}/throughput/current")
    return f"{prefix} Step: {step} - Time: {time:.1f}s - Loss: {loss:.4f} - Norm: {norm:.4f} - Perplexity: {perplexity:.1f} - Throughput: {throughput:.1f}"

def get_eval_pbar_description(metrics: Metrics, prefix: str):
    curr_metrics = metrics.compute()
    loss = curr_metrics.get(f"{metrics.name}/loss/average")
    perplexity = curr_metrics.get(f"{metrics.name}/perplexity/average")
    return f"{prefix} Avg. Loss: {loss:.4f}, Avg. Perplexity: {perplexity:.1f}"

def get_num_steps(max_steps: int, max_epochs: int, num_examples: int, batch_size: int) -> int:
    """Get number of steps to train/val/test; whatever is reached first max_steps or max_epochs"""
    assert num_examples >= batch_size, "Number of examples must be at least batch size"
    assert max_steps > 0 or max_epochs > 0, "Specify at least one of `max_steps` and `max_epochs`"
    max_steps_epoch = num_examples // batch_size * max_epochs
    if max_epochs == -1: return max_steps
    if max_steps == -1: return max_steps_epoch
    return min(max_steps, max_steps_epoch)

def get_train_setup(steps: int, batch_size: int, seq_length: int, micro_batch_size: int, num_examples: int):
    tokens_per_step = batch_size * seq_length
    total_tokens = steps * tokens_per_step
    avg_token_repetitions = total_tokens / (num_examples * seq_length)
    grad_accumulation_steps = batch_size // micro_batch_size if micro_batch_size > 0 else 1

    return {
        "steps": steps,
        "batch_size": batch_size,
        "seq_length": seq_length,
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

def filter_logits_targets(logits: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_flat = logits.view(-1, logits.size(-1))  # (B*L, V)
    targets_flat = target_ids.detach().view(-1)  # (B*L)
    
    # Apply mask by filtering out padded positions
    mask_flat = attention_mask.view(-1)  # (B*L)
    logits_filtered = logits_flat[mask_flat.bool()]  # ((B*L)', V)
    targets_filtered = targets_flat[mask_flat.bool()]  # ((B*L)')

    return logits_filtered, targets_filtered

def initialize_gradients(model):
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            param.grad = torch.zeros_like(param)

def nullcontext():
    return contextlib.nullcontext()

"""
Helpers need for inner-outer optimizer scheme for Diloco. The main idea is to use a CPU
outer optimizer which has a copy of the model between outer steps. For H steps, we use
a regular inner optimizer to update a local model. After H steps, we
1. Compute the pseudo gradient as the difference between the outer and inner model
2. Sync the pseudo gradient to the stage process group
3. Update the outer model with the pseudo gradient
4. Copy the outer model to the inner model
"""

def get_outer_model(inner_model: nn.Module) -> nn.Module:
    """Initializes the outer model from the inner model"""
    outer_model = copy.deepcopy(inner_model)
    return outer_model.to("cpu")

def compute_pseudo_gradient(inner_model: nn.Module, outer_model: nn.Module):
    """Computes the pseudo gradient as the difference between the outer and inner model"""
    for param_outer, param_inner in zip(outer_model.parameters(), inner_model.parameters()):
        param_outer.grad = (param_outer.data - param_inner.to(param_outer.device).data).clone()

def sync_inner_model(outer_model: nn.Module, inner_model: nn.Module):
    """Syncs the inner model from the CPU outer model to GPU"""
    for param_outer, param_inner in zip(outer_model.parameters(), inner_model.parameters()):
        param_inner.data.copy_(param_outer.detach().to(param_inner.device))