"""
Single-GPU LLM pre-training.
"""
import autorootcwd

import os
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import disable_progress_bar
from tqdm import tqdm

from src.logger import Level
from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, tokenize, get_train_pbar_description, get_eval_pbar_description, get_num_steps, get_train_setup, format_int, format_float, get_dtype
from src.metrics import Outputs, Step, Time, MicroTime, Examples, Tokens, Norm, Loss, Perplexity, Throughput, LearningRate, Metrics
from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from pydantic_config import BaseConfig, parse_argv

class BaselineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    sample: SampleConfig
    logging: LoggingConfig

def train(step: int, model: AutoModelForCausalLM, batch_loader: DataLoader, loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, device: torch.device, config: BaselineConfig) -> Outputs:
    start = time.time()
    model.train()
    model.to(device)
    optimizer.zero_grad()
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0

    grad_accumulation_steps = len(batch_loader)
    for micro_batch in batch_loader:
        micro_batch = {k: v.to(device) for k, v in micro_batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=get_dtype(config.train.amp.dtype)):
            outputs = model.forward(micro_batch["input_ids"])
            loss = loss_fn(outputs.logits.transpose(1, 2), micro_batch["target_ids"])

        # Scale loss
        loss = loss / grad_accumulation_steps
        
        # Backward
        loss.backward()

        batch_loss += loss.detach().item()
        batch_examples += micro_batch["input_ids"].shape[0]
        batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / grad_accumulation_steps

    return Outputs(step=step, lr=lr, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=step_time, micro_step_time=micro_step_time)

def eval(step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Outputs:
    start = time.time()
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        num_examples = batch["input_ids"].shape[0]
        num_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

    end = time.time()
    step_time = end - start

    return Outputs(step=step, loss=outputs.loss.item(), num_tokens=num_tokens, num_examples=num_examples, time=step_time)

def sample(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device, config: BaselineConfig) -> List[str]:
    model.eval()
    model.to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(config.sample.prompt, return_tensors="pt").to(device)
    with torch.amp.autocast(device_type=device.type, dtype=get_dtype(config.train.amp.dtype)):
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=config.sample.max_new_tokens, top_k=config.sample.top_k, top_p=config.sample.top_p, num_return_sequences=config.sample.num_return_sequences, temperature=config.sample.temperature)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main(config: BaselineConfig):
    # Set precision and seed
    seed_everything(config.train.seed)

    # Get logger
    logger = get_logger(config.logging)
    logger.log_config(config)

    # Set precision
    torch.set_float32_matmul_precision(config.train.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}")

    # Get device
    device = get_device()
    logger.log_message(f"Using device: {device}")

    # Load model
    model = get_model(config.model)
    logger.log_message(f"Loaded model '{config.model.name}' ({format_int(model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(config.model)
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded tokenizer '{config.model.name}' ({format_int(len(tokenizer), 0)} vocab size)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Prepare dataset
    seq_length = config.data.seq_length
    train_data = train_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count())
    val_data = val_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count())
    test_data = test_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count())
    logger.log_message(f"Tokenized dataset with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=True, cycle=True)
    val_dataloader = get_dataloader(val_data, batch_size=config.train.micro_batch_size, shuffle=True, cycle=True)
    test_dataloader = get_dataloader(test_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=True)
    
    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.train.micro_batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.train.micro_batch_size)

    # Get training, evaluation and testing setup
    train_setup = get_train_setup(num_train_steps, config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(train_data))
    eval_setup = get_train_setup(num_eval_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(val_data))
    test_setup = get_train_setup(num_test_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(test_data))

    logger.log_message("Train setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in train_setup.items()]))
    logger.log_message("Eval setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in eval_setup.items()]))
    logger.log_message("Test setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in test_setup.items()]))

    # Compute grad accumulation steps
    grad_accumulation_steps = config.train.batch_size // config.train.micro_batch_size
    assert config.train.batch_size % grad_accumulation_steps == 0, "Batch size must be divisible by grad accumulation steps"

    # Set up optimizer
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)
    loss_fn = nn.CrossEntropyLoss()

    # Start Training
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(train_step, model, micro_batchloader, loss_fn, optimizer, scheduler, device, config)
        
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

        # Sample
        if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
            samples = sample(model, tokenizer, device, config)
            logger.log_samples(samples, train_step)

        # Checkpoint
        if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
            logger.log_checkpoint(model, tokenizer, train_step)

    # Test
    if config.eval.enable:
        test_metrics = Metrics([Loss(), Perplexity()], name="test")
        num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.train.micro_batch_size)
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
    main(BaselineConfig(**parse_argv()))
