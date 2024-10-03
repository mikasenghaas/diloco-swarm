import torch
import numpy as np
from itertools import cycle as cycle_iter
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig, TokenizerConfig, DataConfig, LoggingConfig
from .logger import Logger

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

def get_model(model: ModelConfig) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model.name)

def get_tokenizer(tokenizer: TokenizerConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer.name, fast=tokenizer.fast)

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

def get_logger(logging: LoggingConfig) -> Logger:
    return Logger(logging)

def non_empty_text(examples: Dict[str, Any]) -> bool:
    return examples["text"] != ""

def non_headline(examples: Dict[str, Any]) -> bool:
    return not examples["text"].startswith(" = ")

def tokenize(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length+1)
