"""
Minimal implementation of pre-training a LLM for text generation.
"""
import autorootcwd
import time
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from datasets import disable_progress_bar
from tqdm import tqdm

from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, non_empty_text, non_headline, tokenize, get_train_pbar_description, get_eval_pbar_description, get_num_steps, format_int
from pydantic import validate_call
from pydantic_config import parse_argv
from src.config import Config
from src.logger import Level
from src.metrics import Outputs, Step, Examples, Tokens, Norm, Loss, Perplexity, Throughput, LearningRate, Metrics

def train(grad_accumulation_steps: int, micro_batch_size: int,step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], optimizer: AdamW, scheduler: LambdaLR, device: torch.device) -> Outputs:
    start = time.time()
    model.train()
    model.to(device)
    optimizer.zero_grad()
    batch_loss = torch.Tensor([0.0]).to(device)
    batch_tokens, batch_examples = 0, 0
    for micro_step in range(grad_accumulation_steps):
        # micro_batch = {k: v.to(device) for k, v in micro_batch.items()}
        start_idx = micro_step * micro_batch_size
        end_idx = start_idx + micro_batch_size
        micro_batch = {k: v[start_idx:end_idx].to(device) for k, v in batch.items()}
        outputs = model(micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"], labels=micro_batch["labels"])
        outputs.loss /= grad_accumulation_steps
        outputs.loss.backward()

        batch_loss += outputs.loss.detach()
        batch_examples += micro_batch["input_ids"].shape[0]
        batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    return Outputs(step=step, lr=scheduler.get_last_lr()[0], loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=time.time() - start)

def eval(step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Outputs:
    start = time.time()
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

        return Outputs(step=step, loss=outputs.loss, tokens_processed=batch["input_ids"].shape[0] * batch["input_ids"].shape[1], time=time.time() - start)

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
    logger.log_message(f"Loaded model '{config.model.name}' ({format_int(model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded tokenizer '{config.tokenizer.name}' ({format_int(len(tokenizer), 0)} tokens)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Prepare dataset
    train_data = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    val_data = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    test_data = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    logger.log_message(f"Tokenized dataset with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=True, cycle=True)
    val_dataloader = get_dataloader(val_data, batch_size=config.eval.batch_size, shuffle=True, cycle=True)
    test_dataloader = get_dataloader(test_data, batch_size=config.eval.batch_size, shuffle=False, cycle=True)
    logger.log_message(f"Prepared dataloaders")
    
    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.eval.batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.eval.batch_size)
    # TODO: Show total steps/examples/tokens, then examples/tokens per step (function of B*L)
    logger.log_message(f"Train setup:\tSteps: {format_int(num_train_steps, 0)}\tBatch Size: {config.train.batch_size}\tExamples: {format_int(num_train_steps * config.train.batch_size, 0)}\t Tokens: {format_int(num_train_steps * config.train.batch_size * config.data.seq_length, 0)}\t Fraction of Data: {num_train_steps * config.train.batch_size / len(train_data):.2f}")
    logger.log_message(f"Eval setup:\tSteps: {format_int(num_eval_steps, 0)}\tBatch Size: {config.eval.batch_size}\tExamples: {format_int(num_eval_steps * config.eval.batch_size, 0)}\t Tokens: {format_int(num_eval_steps * config.eval.batch_size * config.data.seq_length, 0)}\t Fraction of Data: {num_eval_steps * config.eval.batch_size / len(val_data):.2f}")
    logger.log_message(f"Test setup:\tSteps: {format_int(num_test_steps, 0)}\tBatch Size: {config.eval.batch_size}\tExamples: {format_int(num_test_steps * config.eval.batch_size, 0)}\t Tokens: {format_int(num_test_steps * config.eval.batch_size * config.data.seq_length, 0)}\t Fraction of Data: {num_test_steps * config.eval.batch_size / len(test_data):.2f}")

    # Compute grad accumulation steps
    grad_accumulation_steps = config.train.batch_size // config.train.micro_batch_size
    assert config.train.batch_size % grad_accumulation_steps == 0, "Batch size must be divisible by grad accumulation steps"
    logger.log_message(f"Gradient Accumulation Steps: {grad_accumulation_steps}")

    # Set up optimizer
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)

    # Start Training
    logger.log_message("Starting training")
    train_metrics = Metrics([Step(), Examples(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        micro_dataloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(grad_accumulation_steps, config.train.micro_batch_size, train_step, model, batch, optimizer, scheduler, device)
        
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

    # Test
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

    # Final Checkpoint
    if config.logging.ckpt.enable:
        logger.log_checkpoint(model, tokenizer, train_step)

    logger.log_message("Finished training!")
    logger.close()

if __name__ == "__main__":
    disable_progress_bar()
    main(Config(**parse_argv()))
