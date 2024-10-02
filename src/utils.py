import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import List, Dict, Any

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_model(model_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model_name)

def get_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)

def get_dataset(data_path: str, data_name: str) -> Dataset:
    return load_dataset(data_path, data_name)

def get_subset(dataset: Dataset, subset_size: float) -> Dataset:
    return dataset.shuffle(seed=42).select(range(int(len(dataset) * subset_size)))

def non_empty_text(examples: Dict[str, Any]) -> bool:
    return examples["text"] != ""

def non_headline(examples: Dict[str, Any]) -> bool:
    return not examples["text"].startswith(" = ")

def tokenize(examples: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length+1)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        'input_ids': torch.stack([torch.tensor(example['input_ids'][:-1]) for example in batch]),
        'attention_mask': torch.stack([torch.tensor(example['attention_mask'][:-1]) for example in batch]),
        'labels': torch.stack([torch.tensor(example['input_ids'][1:]) for example in batch]),
    }