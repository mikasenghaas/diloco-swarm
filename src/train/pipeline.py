"""
Pipeline Parallel LLM Pre-Training.

torchrun --nproc_per_node 2 src/train/pipeline.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --logging.console.enable false --logging.file.enable true
"""
import autorootcwd

import os
import time
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import disable_progress_bar
from transformers import AutoTokenizer

from src.comm import Comm
from src.world import World
from src.logger import Level
from src.ckpt import Checkpoint
from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model, get_sharded_model, get_tokenizer, get_dataset, tokenize, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Examples, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate
from pydantic_config import BaseConfig, parse_argv

# Global logger
logger = None

class PipelineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

def train(step: int, sharded_model: nn.Module, batch_loader: DataLoader, loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, world: World, comm: Comm, config: TrainConfig, device: torch.device) -> Outputs:
    # Prepare model
    start = time.time()
    sharded_model.to(device)
    sharded_model.train()
    
    # Prepare statistics
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0
    input_tensors, output_tensors = [], []

    # Zero-out gradient
    optimizer.zero_grad()
    grad_accumulation_steps = len(batch_loader)
    for micro_batch in batch_loader:
        # Forward
        input_ids, target_ids = micro_batch["input_ids"].to(device), micro_batch["target_ids"].to(device)
        input_tensor = comm.recv_forward(device=device)
        output_tensor = sharded_model.forward(input_ids=input_ids, hidden_states=input_tensor)
        comm.send_forward(output_tensor)

        if world.is_last_stage:
            # Reshape logits and targets for loss calculation
            mask = micro_batch["attention_mask"] # (B, L)
            logits_flat = output_tensor.view(-1, output_tensor.size(-1))  # (B*L, V)
            targets_flat = target_ids.view(-1)  # (B*L)
            
            # Apply mask by filtering out padded positions
            mask_flat = mask.view(-1)  # (B*L)
            logits_filtered = logits_flat[mask_flat.bool()]  # ((B*L)', V)
            targets_filtered = targets_flat[mask_flat.bool()]  # ((B*L)')
            
            # Calculate loss only on non-padded positions
            output_tensor = loss_fn(logits_filtered, targets_filtered)
            output_tensor /= grad_accumulation_steps # Scale loss

            # Update statistics
            batch_loss += output_tensor.detach().item()
            batch_examples += input_ids.shape[0]
            batch_tokens += input_ids.shape[0] * input_ids.shape[1]

        output_tensor_grad = comm.recv_backward(device=device)
        # input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = sharded_model.backward(input_tensor, output_tensor, output_tensor_grad)
        comm.send_backward(input_tensor_grad)
    
    # Optimizer and scheduler step
    norm = torch.nn.utils.clip_grad_norm_(sharded_model.parameters(), max_norm=config.max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    # Synchronization and timing
    torch.cuda.synchronize()
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / grad_accumulation_steps

    return Outputs(step=step, lr=lr, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=step_time, micro_step_time=micro_step_time)

def eval(step: int, sharded_model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, world: World, comm: Comm, device: torch.device) -> Outputs:
    start = time.time()
    sharded_model.to(device)
    sharded_model.eval()
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0
    with torch.no_grad():
        input_ids, target_ids = batch["input_ids"].to(device), batch["target_ids"].to(device)
        hidden_states = comm.recv_forward(device=device)
        output_tensor = sharded_model.forward(input_ids=input_ids, hidden_states=hidden_states)
        comm.send_forward(output_tensor)

        if world.is_last_stage:
            output_tensor = loss_fn(output_tensor.transpose(1, 2), target_ids)
            batch_loss += output_tensor.detach().item()
            batch_examples += input_ids.shape[0]
            batch_tokens += input_ids.shape[0] * input_ids.shape[1]
    
    end = time.time()
    step_time = end - start

    # Synchronize at end of iteration
    torch.cuda.synchronize()
    dist.barrier()

    return Outputs(step=step, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, time=step_time)

def sample(sharded_model: nn.Module, tokenizer: AutoTokenizer, world: World, activation_comm: Comm, input_ids_comm: Comm, prompt_length: int, config: SampleConfig, device: torch.device) -> List[str]:
    # Prepare model
    sharded_model.to(device)
    sharded_model.eval()

    # Prepare input and generate output
    tensor_length = prompt_length + config.max_new_tokens
    input_ids = tokenize(config.prompt, tokenizer, max_length=tensor_length)["input_ids"].to(device).repeat(config.num_samples, 1)
    stop_flag = -torch.ones((config.num_samples, tensor_length), device=device, dtype=torch.long)
    for generated_id in range(prompt_length, tensor_length):
        if world.is_first_stage and generated_id > prompt_length:
            input_ids = input_ids_comm.recv_from(world.last_stage, requires_grad=False, device=device)
            if (input_ids == stop_flag).all():
                break
        hidden_states = activation_comm.recv_forward(device=device)
        output_tensor = sharded_model.forward(input_ids=input_ids, hidden_states=hidden_states)
        activation_comm.send_forward(output_tensor)

        if world.is_last_stage:
            logits = output_tensor[:, generated_id-1, :] / config.temperature # (B, V)
            if config.top_k is not None:
                v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # (B, V)
            probs = F.softmax(logits, dim=-1) # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            input_ids[:, generated_id] = idx_next.squeeze()
            if tokenizer.eos_token_id is not None and (idx_next == tokenizer.eos_token_id).all():
                input_ids_comm.send_to(stop_flag, world.first_stage)
                break
            if generated_id < tensor_length - 1:
                input_ids_comm.send_to(input_ids, world.first_stage)

    # Synchronize at end of iteration
    torch.cuda.synchronize()

    return [tokenizer.decode(generated_ids, skip_special_tokens=True) for generated_ids in input_ids] if world.is_last_stage else []

def main(config: PipelineConfig):
    # Seed everything
    seed_everything(config.train.seed)

    # Get world parameters
    local_rank, world_size = get_world()

    # Set device
    device = get_device(local_rank)
    torch.cuda.set_device(local_rank)

    # Get world
    world = World(local_rank, world_size, device)

    # Get logger
    logger_name = f"{local_rank}" if world_size > 1 else "master"
    logger = get_logger(config.logging, logger_name, world.run_id)
    
    # Log values
    logger.log_config(config)
    logger.log_world(world)

    # Get checkpoint
    if config.logging.ckpt.enable:
        ckpt = Checkpoint(logger.checkpoint_dir)
        ckpt.setup(world)
        logger.log_message(f"Checkpoint directory: {ckpt.base_dir}")

    # Set precision
    torch.set_float32_matmul_precision(config.train.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}")

    # Load model
    model = get_model(config.model)
    logger.log_message(f"Loaded GPT-2 ({format_int(model.num_parameters(), 2)} parameters)")

    # Get sharded model
    sharded_model = get_sharded_model(model, world)
    logger.log_message(f"Sharded model ({format_int(sharded_model.num_parameters(), 2)} parameters)")

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
    train_data = train_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=min(len(train_data), os.cpu_count()))
    val_data = val_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=min(len(val_data), os.cpu_count()))
    test_data = test_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length+1, return_tensors=None), batched=True, num_proc=min(len(test_data), os.cpu_count()))
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
    optimizer = get_optimizer(sharded_model, config.train.optimizer)
    scheduler = get_scheduler(optimizer, num_train_steps, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()

    # Get communication
    B, L, H = config.train.micro_batch_size, config.data.seq_length, model.config.n_embd
    comm = Comm(world, (B, L, H), torch.float32)
    logger.log_message(f"Initialized communication: {comm}")

    if config.sample.enable:
        prompt_length = tokenize(config.sample.prompt, tokenizer)["input_ids"].shape[1]
        activation_comm = Comm(world, (config.sample.num_samples, prompt_length+config.sample.max_new_tokens, H), torch.float32)
        input_ids_comm = Comm(world, (config.sample.num_samples, prompt_length+config.sample.max_new_tokens), torch.long)
        logger.log_message(f"Initialized communication: {input_ids_comm}")
        samples = sample(sharded_model, tokenizer, world, activation_comm, input_ids_comm, prompt_length, config.sample, device)
        if world.is_last_stage:
            logger.log_samples(0, samples)

    # Start training
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Loss(), Perplexity(), Throughput(), Norm(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")

    # Validate before training
    if config.eval.enable:
        eval_range = range(1, num_eval_steps+1)
        eval_bar = tqdm(eval_range, position=0, leave=True)
        eval_metrics.reset()
        for eval_step in eval_range:
            # Eval step
            batch = next(val_dataloader)
            outputs = eval(eval_step, sharded_model, batch, loss_fn, world, comm, device)

            # Compute log metrics
            if world.is_last_stage:
                eval_metrics.update(outputs)
                eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))
                eval_bar.update()

        if world.is_last_stage:
            curr_metrics = eval_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=0)

    train_range = range(1, num_train_steps+1)
    train_bar = tqdm(train_range, position=0, leave=True) if world.is_last_stage else None
    for train_step in train_range:
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(train_step, sharded_model, micro_batchloader, loss_fn, optimizer, scheduler, world, comm, config.train, device)

        # Compute and log metrics (only last stage)
        if world.is_last_stage:
            train_metrics.update(outputs)
            curr_metrics = train_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
            train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))
            train_bar.update()

        # Validate
        if config.eval.enable and config.eval.every_n_steps > 0 and train_step % config.eval.every_n_steps == 0:
            eval_range = range(1, num_eval_steps+1)
            eval_bar = tqdm(eval_range, position=1, leave=False) if world.is_last_stage else None
            eval_metrics.reset()
            for eval_step in eval_range:
                # Eval step
                batch = next(val_dataloader)
                outputs = eval(eval_step, sharded_model, batch, loss_fn, world, comm, device)

                # Compute log metrics
                if world.is_last_stage:
                    eval_metrics.update(outputs)
                    eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))
                    eval_bar.update()

            if world.is_last_stage:
                curr_metrics = eval_metrics.compute()
                logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

        # Sample
        if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
            samples = sample(sharded_model, tokenizer, world, activation_comm, input_ids_comm, prompt_length, config.sample, device)
            if world.is_last_stage:
                logger.log_samples(train_step, samples)

        # Checkpoint
        if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
            ckpt_dir = ckpt.save(train_step, sharded_model)
            logger.log_message(f"Saved checkpoint at {ckpt_dir}")

    # Sample after training
    if config.sample.enable:
        samples = sample(sharded_model, tokenizer, world, activation_comm, input_ids_comm, prompt_length, config.sample, device)
        if world.is_last_stage:
            logger.log_samples(train_step, samples)

    # Checkpoint
    if config.logging.ckpt.enable:
        ckpt_dir = ckpt.save(train_step, sharded_model)
        logger.log_message(f"Saved checkpoint at {ckpt_dir}")

    # Test
    if config.eval.enable:
        test_metrics = Metrics([Loss(), Perplexity()], name="test")
        test_range = range(1, num_test_steps+1)
        test_bar = tqdm(test_range, position=0, leave=True)
        for test_step in test_range:
            batch = next(test_dataloader)
            outputs = eval(test_step, sharded_model, batch, loss_fn, world, comm, device)

            # Compute and log metrics
            if world.is_last_stage:
                test_metrics.update(outputs)
                test_bar.set_description(get_eval_pbar_description(test_metrics, prefix="[TEST]"))
                test_bar.update()

        if world.is_last_stage:
            curr_metrics = test_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    # Destroy process group
    if world.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    disable_progress_bar()
    main(PipelineConfig(**parse_argv()))