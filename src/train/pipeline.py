"""
Pipeline Parallel LLM Pre-Training.

Run with:
```
#VERBOSE=1 torchrun --nproc_per_node 2 src/train/pipeline.py @configs/debug.toml --model @configs/model/llama2-9m.toml --data @configs/data/wikitext.toml
```
"""
import autorootcwd

import os
import time
from tqdm import tqdm
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import disable_progress_bar

from src.logger import Level
from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model, get_model_type, get_sharded_model, get_tokenizer, get_dataset, tokenize, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description
from src.world import World
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Examples, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate
from src.comm import Comm
from pydantic_config import BaseConfig, parse_argv

# Global logger
logger = None

class PipelineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    sample: SampleConfig
    logging: LoggingConfig

def train(step: int, sharded_model: nn.Module, batch_loader: DataLoader, loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, device: torch.device, world: World, comm: Comm, max_norm: float) -> Outputs:
    start = time.time()
    sharded_model.train()
    sharded_model.to(device)
    optimizer.zero_grad()
    
    # Initialization
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0
    input_tensors, output_tensors = [], []

    # Forward
    grad_accumulation_steps = len(batch_loader)
    for micro_batch_id, micro_batch in enumerate(batch_loader):
        input_tensor = comm.recv_forward(micro_batch_id)
        micro_batch["hidden_states"] = input_tensor
        micro_batch = {k: v.to(device) if v is not None else v for k, v in micro_batch.items()}
        output_tensor = sharded_model.forward(**micro_batch)
        comm.send_forward(output_tensor, micro_batch_id)

        if world.is_last_stage:
            output_tensor = loss_fn(output_tensor.transpose(1, 2), micro_batch["target_ids"].to(device))
            output_tensor = output_tensor / grad_accumulation_steps
            batch_loss += output_tensor.detach().item()
            batch_examples += micro_batch["input_ids"].shape[0]
            batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    # Backward
    for micro_batch_id, micro_batch in enumerate(batch_loader): 
        output_tensor_grad = comm.recv_backward(micro_batch_id)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = sharded_model.backward(input_tensor, output_tensor, output_tensor_grad)
        comm.send_backward(input_tensor_grad, micro_batch_id)
    
    # Optimizer and scheduler step
    norm = torch.nn.utils.clip_grad_norm_(sharded_model.parameters(), max_norm=max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    # Synchronization and timing
    torch.cuda.synchronize()
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / grad_accumulation_steps

    return Outputs(step=step, lr=lr, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=step_time, micro_step_time=micro_step_time)

def eval(step: int, sharded_model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device, world: World, comm: Comm) -> Outputs:
    start = time.time()
    sharded_model.eval()
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0
    with torch.no_grad():
        input_tensor = comm.recv_forward(0)
        batch["hidden_states"] = input_tensor
        batch = {k: v.to(device) if v is not None else v for k, v in batch.items()}
        output_tensor = sharded_model.forward(**batch)
        comm.send_forward(output_tensor, 0)

        if world.is_last_stage:
            output_tensor = loss_fn(output_tensor.transpose(1, 2), batch["target_ids"].to(device))
            batch_loss += output_tensor.detach().item()
            batch_examples += batch["input_ids"].shape[0]
            batch_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
    
    end = time.time()
    step_time = end - start

    return Outputs(step=step, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, time=step_time)

def sample() -> List[str]:
    pass

def main(config: PipelineConfig):
    global logger
    # Set precision and seed
    seed_everything(config.train.seed)

    # Initialize world
    local_rank, world_size = get_world()
    world = World(local_rank, world_size)

    # Synchronize processes to ensure consistent timestamp
    dist.barrier()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get logger
    logger_name = f"{local_rank}" if world_size > 1 else "master"
    logger = get_logger(config.logging, logger_name, run_id)
    logger.log_config(config)
    logger.log_message(str(world))

    # Set device
    torch.cuda.set_device(local_rank)
    device = get_device(local_rank)
    logger.log_message(f"Using device: {device}")

    # Set precision
    torch.set_float32_matmul_precision(config.train.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}")

    # Get model
    model = get_model(config.model, device)
    logger.log_message(f"Loaded model '{config.model.name}' ({format_int(model.num_parameters(), 2)} parameters)")

    # Get sharded model
    sharded_model = get_sharded_model(model, world, get_model_type(config.model))
    logger.log_message(f"Sharded model '{config.model.name}' ({format_int(sharded_model.num_parameters(), 2)} parameters)")
    # logger.log_message(str(sharded_model))

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
    train_data = train_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count() // world_size, remove_columns=train_data.column_names)
    val_data = val_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count() // world_size, remove_columns=val_data.column_names)
    test_data = test_data.map(lambda examples: tokenize(examples, tokenizer, seq_length), batched=True, num_proc=os.cpu_count() // world_size, remove_columns=test_data.column_names)
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
    optimizer = get_optimizer(config.train, sharded_model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)
    loss_fn = nn.CrossEntropyLoss()

    # Get communication
    B, L, H = config.train.micro_batch_size, config.data.seq_length, model.config.hidden_size
    comm = Comm(world, (B, L, H), torch.float32)
    logger.log_message(f"Initialized communication: {comm}")

    # Training loop
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Loss(), Perplexity(), Throughput(), Norm(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_range = range(1, num_train_steps+1)
    train_bar = tqdm(train_range, position=0, leave=True) if world.is_last_stage else None
    for train_step in train_range:
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(train_step, sharded_model, micro_batchloader, loss_fn, optimizer, scheduler, device, world, comm, max_norm=config.train.max_norm)

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
                outputs = eval(eval_step, sharded_model, batch, loss_fn, device, world, comm)

                # Compute log metrics
                if world.is_last_stage:
                    eval_metrics.update(outputs)
                    eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]"))
                    eval_bar.update()

            # Log eval metrics
            if world.is_last_stage:
                curr_metrics = eval_metrics.compute()
                logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)

    # Destroy process group
    if world.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    disable_progress_bar()
    main(PipelineConfig(**parse_argv()))