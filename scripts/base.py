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

from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_optimizer, get_scheduler, non_empty_text, non_headline, tokenize, get_train_pbar_description, get_eval_pbar_description, get_num_steps
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
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test examples")

    # Prepare dataset
    train_data = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    val_data = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    test_data = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    logger.log_message(f"Tokenized dataset with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test examples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=True, cycle=True)
    val_dataloader = get_dataloader(val_data, batch_size=config.eval.batch_size, shuffle=True, cycle=True)
    test_dataloader = get_dataloader(test_data, batch_size=config.eval.batch_size, shuffle=False, cycle=True)
    logger.log_message(f"Prepared dataloaders")
    
    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.eval.batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.eval.batch_size)
    logger.log_message(f"Train setup:\tSteps: {num_train_steps}\tBatch Size: {config.train.batch_size}\tExamples: {num_train_steps * config.train.batch_size}\t Tokens: {num_train_steps * config.train.batch_size * config.data.seq_length}\t Fraction of Data: {num_train_steps * config.train.batch_size / len(train_data):.2f}")
    logger.log_message(f"Eval setup:\tSteps: {num_eval_steps}\tBatch Size: {config.eval.batch_size}\tExamples: {num_eval_steps * config.eval.batch_size}\t Tokens: {num_eval_steps * config.eval.batch_size * config.data.seq_length}\t Fraction of Data: {num_eval_steps * config.eval.batch_size / len(val_data):.2f}")
    logger.log_message(f"Test setup:\tSteps: {num_test_steps}\tBatch Size: {config.eval.batch_size}\tExamples: {num_test_steps * config.eval.batch_size}\t Tokens: {num_test_steps * config.eval.batch_size * config.data.seq_length}\t Fraction of Data: {num_test_steps * config.eval.batch_size / len(test_data):.2f}")

    # Set up optimizer
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)

    # Start Training
    logger.log_message("Starting training")
    train_metrics = Metrics([Step(), ExamplesSeen(), TokensSeen(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
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
        if config.eval.enable and config.eval.every_n_steps > 0 and train_step % config.eval.every_n_steps == 0:
            eval_bar = tqdm(range(1, num_eval_steps+1), position=1, leave=False)
            eval_metrics.reset()
            for eval_step in eval_bar:
                # Eval step
                batch = next(val_dataloader)
                outputs = eval(eval_step, model, batch, device)

                # Compute and log metrics
                eval_metrics.update(outputs)
                eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))

            curr_metrics = eval_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

        # Checkpoint
        if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
            logger.log_checkpoint(model, tokenizer, train_step)

    # Evaluate
    if config.eval.enable:
        test_metrics = Metrics([Loss(), Perplexity()], name="test")
        num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.eval.batch_size)
        test_bar = tqdm(range(1, num_test_steps+1), position=0, leave=True)
        for test_step in test_bar:
            batch = next(test_dataloader)
            outputs = eval(test_step, model, batch, device)

            # Compute and log metrics
            test_metrics.update(outputs)
            test_bar.set_description(get_eval_pbar_description(test_metrics, prefix="[TEST]"))

        curr_metrics = test_metrics.compute()
        logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    # Checkpoint
    if config.logging.ckpt.enable:
        logger.log_checkpoint(model, tokenizer, train_step)

    logger.log_message("Finished training!")
    logger.close()

if __name__ == "__main__":
    disable_progress_bar()
    main(Config(**parse_argv()))
