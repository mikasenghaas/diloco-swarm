"""
Minimal implementation of pre-training a LLM for text generation.
"""
import autorootcwd
import time
from typing import Dict

import torch
from torch.optim import AdamW, Optimizer
from transformers import AutoModelForCausalLM
from datasets import disable_progress_bar
from tqdm import tqdm

from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, non_empty_text, non_headline, tokenize, track_time, get_train_pbar_description, get_eval_pbar_description
from pydantic import validate_call
from pydantic_config import parse_argv
from src.config import Config
from src.logger import Level
from src.metrics import Step, ExamplesSeen, TokensSeen, Loss, Perplexity, Throughput, Metrics

@track_time
def train(model: AutoModelForCausalLM, optimizer: Optimizer, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    model.train()
    model.to(device)
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()

    return outputs

@track_time
def eval(model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        return model(input_ids, attention_mask=attention_mask, labels=labels)

@validate_call
def main(config: Config):
    # Set seed
    seed_everything(config.train.seed)

    # Get logger
    logger = get_logger(config.logging)
    logger.log_config(config)

    # Get device
    device = get_device()
    logger.log_message(f"Using device: {device}")

    # Load model
    model = get_model(config.model)
    logger.log_message(f"Loaded model '{config.model.name}' ({model.num_parameters() / 1e6:.1f}M parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded tokenizer '{config.tokenizer.name}' ({len(tokenizer) / 1e3:.1f}K tokens)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")

    # Prepare dataset
    train_data_tok = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    val_data_tok = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    test_data_tok = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    logger.log_message(f"Tokenized dataset with {len(train_data_tok)} train, {len(val_data_tok)} validation, {len(test_data_tok)} test samples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data_tok, batch_size=config.train.batch_size, shuffle=True, cycle=config.data.cycle)
    val_dataloader = get_dataloader(val_data_tok, batch_size=config.eval.batch_size, shuffle=True, cycle=config.data.cycle)
    test_dataloader = get_dataloader(test_data_tok, batch_size=config.eval.batch_size, shuffle=False, cycle=False)
    logger.log_message(f"Prepared dataloaders")

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay, betas=config.train.adam_betas)

    # Start Training
    logger.log_message("Starting training")
    train_metrics = Metrics([Step(), ExamplesSeen(), TokensSeen(), Loss(), Perplexity(), Throughput()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    num_train_batches = len(train_data) // config.train.batch_size
    train_bar = tqdm(range(1, min(num_train_batches, config.train.max_steps)+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        outputs = train(model, optimizer, batch, device)
        
        # Compute and log metrics
        train_metrics.update(outputs)
        curr_metrics = train_metrics.compute()
        logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
        train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))

        # Evaluate
        if config.eval.enable and train_step % config.eval.every_n_steps == 0:
            num_eval_batches = len(val_data) // config.eval.batch_size
            eval_bar = tqdm(range(1, min(num_eval_batches, config.eval.max_steps)+1), position=1, leave=False)
            eval_metrics.reset()
            for _ in eval_bar:
                # Eval step
                batch = next(val_dataloader)
                outputs = eval(model, batch, device)

                # Compute and log metrics
                eval_metrics.update(outputs)
                eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL ]"))

            curr_metrics = eval_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    logger.log_message("Finished training!")
    logger.close()

if __name__ == "__main__":
    disable_progress_bar()
    main(Config(**parse_argv()))
