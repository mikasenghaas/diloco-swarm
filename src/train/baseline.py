"""
Single-GPU LLM Pre-Training.

python src/train/baseline.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml
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
from transformers import GPT2Tokenizer
from datasets import disable_progress_bar
from tqdm import tqdm

from src.ckpt import Checkpoint
from src.logger import Level
from src.world import World
from src.utils import seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, tokenize, get_train_pbar_description, get_eval_pbar_description, get_num_steps, get_train_setup, format_int, format_float, get_dtype
from src.metrics import Outputs, Step, Time, MicroTime, Examples, Tokens, Norm, Loss, Perplexity, Throughput, LearningRate, Metrics
from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.model import GPT2
from src.config import TrainConfig, SampleConfig
from pydantic_config import BaseConfig, parse_argv

class BaselineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

def train(step: int, model: GPT2, batch_loader: DataLoader, loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, config: TrainConfig, device: torch.device) -> Outputs:
    # Prepare model
    start = time.time()
    model.to(device)
    model.train()

    # Prepare statistics
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0

    # Zero-out gradient
    optimizer.zero_grad()
    grad_accumulation_steps = len(batch_loader)
    for micro_batch in batch_loader:
        micro_batch = {k: v.to(device) for k, v in micro_batch.items()}
        with torch.amp.autocast(device_type=device.type, dtype=get_dtype(config.amp.dtype)):
            # Forward
            logits = model.forward(input_ids=micro_batch["input_ids"])

            # Reshape logits and targets for loss calculation
            mask = micro_batch["attention_mask"] # (B, L)
            logits_flat = logits.view(-1, logits.size(-1))  # (B*L, V)
            targets_flat = micro_batch["target_ids"].view(-1)  # (B*L)
            
            # Apply mask by filtering out padded positions
            mask_flat = mask.view(-1)  # (B*L)
            logits_filtered = logits_flat[mask_flat.bool()]  # ((B*L)', V)
            targets_filtered = targets_flat[mask_flat.bool()]  # ((B*L)')
            
            # Calculate loss only on non-padded positions
            loss = loss_fn(logits_filtered, targets_filtered)
        
        # Backward with scaled loss
        loss = loss / grad_accumulation_steps
        loss.backward()

        # Compute statistics
        batch_loss += loss.detach().item()
        batch_examples += micro_batch["input_ids"].shape[0]
        batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

    # Step optimizer and scheduler
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
    optimizer.step()
    lr = scheduler.get_last_lr()[0]
    scheduler.step()

    # Synchronize CUDA
    torch.cuda.synchronize()

    # Compute step time
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / grad_accumulation_steps

    return Outputs(step=step, lr=lr, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=step_time, micro_step_time=micro_step_time)

def eval(step: int, model: GPT2, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device) -> Outputs:
    # Prepare model
    start = time.time()
    model.to(device)
    model.eval()

    # Prepare batch
    batch = {k: v.to(device) for k, v in batch.items() if v is not None}
    with torch.no_grad():
        # Forward
        logits = model.forward(input_ids=batch["input_ids"])

        # Reshape logits and targets for loss calculation
        mask = batch["attention_mask"]  # [B, L]
        logits_flat = logits.view(-1, logits.size(-1))  # [B*L, V]
        targets_flat = batch["target_ids"].view(-1)  # [B*L]
        
        # Apply mask by filtering out padded positions
        mask_flat = mask.view(-1)  # [B*L]
        logits_filtered = logits_flat[mask_flat.bool()]  # [(B*L)', V]
        targets_filtered = targets_flat[mask_flat.bool()]  # [(B*L)']
        
        # Calculate loss only on non-padded positions
        loss = loss_fn(logits_filtered, targets_filtered)

        # Compute statistics
        num_examples = batch["input_ids"].shape[0]
        num_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]

    end = time.time()
    step_time = end - start

    return Outputs(step=step, loss=loss.item(), num_tokens=num_tokens, num_examples=num_examples, time=step_time)

def sample(model: GPT2, tokenizer: GPT2Tokenizer, config: SampleConfig, device: torch.device) -> List[str]:
    # Prepare model
    model.to(device)
    model.eval()

    # Prepare input and generate output
    input_ids = tokenize(config.prompt, tokenizer)["input_ids"].to(device).repeat(config.num_samples, 1)
    all_generated_ids = model.generate(input_ids, **config.model_dump(), eos_token_id=tokenizer.eos_token_id)

    # Decode the generated text
    return [tokenizer.decode(generated_ids, skip_special_tokens=True) for generated_ids in all_generated_ids]
    
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

    # Get world
    world = World(local_rank=0, world_size=1, device=device)
    logger.log_world(world)

    if config.logging.ckpt.enable:
        ckpt = Checkpoint(logger.checkpoint_dir)
        ckpt.setup(world)
        logger.log_message(f"Checkpoint directory: {ckpt.base_dir}")

    # Load model
    model = get_model(config.model)
    logger.log_message(f"Loaded GPT-2 ({format_int(model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded GPT-2 tokenizer ({format_int(len(tokenizer), 0)} vocab size)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Prepare dataset
    seq_length = config.data.seq_length
    train_data = train_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=os.cpu_count())
    val_data = val_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=os.cpu_count())
    test_data = test_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=os.cpu_count())
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
    optimizer = get_optimizer(model, config.train.optimizer)
    scheduler = get_scheduler(optimizer, num_train_steps, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()

    # Sample before training
    if config.sample.enable:
        samples = sample(model, tokenizer, config.sample, device)
        logger.log_samples(0, samples)

    # Start Training
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")

    # Validate before training
    if config.eval.enable:
        eval_bar = tqdm(range(1, num_eval_steps+1), position=0, leave=True)
        eval_metrics.reset()
        for eval_step in eval_bar:
            batch = next(val_dataloader)
            outputs = eval(eval_step, model, batch, loss_fn, device)
            eval_metrics.update(outputs)
            eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))

        curr_metrics = eval_metrics.compute()
        logger.log_metrics(curr_metrics, level=Level.DEBUG, step=0)

    train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(train_step, model, micro_batchloader, loss_fn, optimizer, scheduler, config.train, device)
        
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
                batch = next(val_dataloader)
                outputs = eval(eval_step, model, batch, loss_fn, device)
                eval_metrics.update(outputs)
                eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))

            curr_metrics = eval_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

        # Sample
        if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
            samples = sample(model, tokenizer, config.sample, device)
            logger.log_samples(train_step, samples)

        # Checkpoint
        if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
            ckpt_dir = ckpt.save(train_step, model)
            logger.log_message(f"Saved model checkpoint at {ckpt_dir}")

    # Sample after training
    if config.sample.enable:
        samples = sample(model, tokenizer, config.sample, device)
        logger.log_samples(train_step, samples)

    # Test
    if config.eval.enable:
        test_metrics = Metrics([Loss(), Perplexity()], name="test")
        num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.train.micro_batch_size)
        test_bar = tqdm(range(1, num_test_steps+1), position=0, leave=True)
        for test_step in test_bar:
            batch = next(test_dataloader)
            outputs = eval(test_step, model, batch, loss_fn, device)

            # Compute and log metrics
            test_metrics.update(outputs)
            test_bar.set_description(get_eval_pbar_description(test_metrics, prefix="[TEST]"))

        curr_metrics = test_metrics.compute()
        logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    # Final Checkpoint
    if config.logging.ckpt.enable:
        ckpt_dir = ckpt.save(train_step, model)
        logger.log_message(f"Saved model checkpoint at {ckpt_dir}")

    logger.log_message("Finished training!")
    logger.close()

if __name__ == "__main__":
    disable_progress_bar()
    main(BaselineConfig(**parse_argv()))
