"""
Minimal implementation of pre-training a LLM for text generation.
"""
import autorootcwd
import time
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM
from datasets import disable_progress_bar
from tqdm import tqdm

from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_optimizer, get_scheduler, non_empty_text, non_headline, tokenize, get_train_pbar_description, get_eval_pbar_description
from pydantic import validate_call
from pydantic_config import parse_argv
from src.config import Config
from src.logger import Level
from src.metrics import Outputs, Step, ExamplesSeen, TokensSeen, Loss, Perplexity, Throughput, LearningRate, Metrics

def train(step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], optimizer: AdamW, scheduler: LambdaLR, device: torch.device) -> Outputs:
    start = time.time()
    model.train()
    model.to(device)
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()
    scheduler.step()

    return Outputs(step=step, lr=scheduler.get_last_lr()[0], loss=outputs.loss, logits=outputs.logits, time=time.time() - start)

def eval(step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Outputs:
    start = time.time()
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        return Outputs(step=step, loss=outputs.loss, logits=outputs.logits, time=time.time() - start)

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
    val_dataloader = get_dataloader(val_data_tok, batch_size=config.val.batch_size, shuffle=True, cycle=config.data.cycle)
    test_dataloader = get_dataloader(test_data_tok, batch_size=config.test.batch_size, shuffle=False, cycle=False)
    logger.log_message(f"Prepared dataloaders")
    
    # Compute number of training batches
    epoch_train_batches = len(train_data) // config.train.batch_size
    num_batches = min(epoch_train_batches, config.train.max_steps)

    # Set up optimizer
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer, num_batches)

    # Start Training
    logger.log_message("Starting training")
    train_metrics = Metrics([Step(), ExamplesSeen(), TokensSeen(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_bar = tqdm(range(1, num_batches+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        outputs = train(train_step, model, batch, optimizer, scheduler, device)
        
        # Compute and log metrics
        train_metrics.update(outputs)
        curr_metrics = train_metrics.compute()
        logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
        train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))

        # Validate
        if config.val.enable and train_step % config.val.every_n_steps == 0:
            num_eval_batches = len(val_data) // config.val.batch_size
            eval_bar = tqdm(range(1, min(num_eval_batches, config.val.max_steps)+1), position=1, leave=False)
            eval_metrics.reset()
            for val_step in eval_bar:
                # Eval step
                batch = next(val_dataloader)
                outputs = eval(val_step, model, batch, device)

                # Compute and log metrics
                eval_metrics.update(outputs)
                eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[VAL]"))

            curr_metrics = eval_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    logger.log_message("Finished training!")
    logger.close()

if __name__ == "__main__":
    disable_progress_bar()
    main(Config(**parse_argv()))
