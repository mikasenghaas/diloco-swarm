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
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import disable_progress_bar
from tqdm import tqdm

from src.logger import Level
from src.config import ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model, get_model_type, get_sharded_model, get_tokenizer, get_dataset, tokenize, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, format_int, get_train_pbar_description
from src.world import World
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Examples, Tokens, Loss, Perplexity, Throughput
from pydantic_config import BaseConfig, parse_argv

class PipelineConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    sample: SampleConfig
    logging: LoggingConfig

def communicate(world, step, operation, tensor=None, shapes=None, dtype=None):
    if operation == 'recv_forward':
        if world.is_first_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = world.prev_rank()
    elif operation == 'send_forward':
        if world.is_last_stage: return
        dest = world.next_rank()
    elif operation == 'recv_backward':
        if world.is_last_stage: return None
        tensor = torch.empty(shapes, requires_grad=True, device='cuda', dtype=dtype)
        src = world.next_rank()
    elif operation == 'send_backward':
        if world.is_first_stage: return
        dest = world.prev_rank()

    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src

    # Send or receive tensor
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    # print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} {world.local_rank} {'→' if is_send else '←'} {peer_rank} | STEP:{step} | RANK:{world.local_rank}", flush=True)

    # Wait for operation to complete
    [req.wait() for req in dist.batch_isend_irecv([op])]

    # Synchronize
    torch.cuda.synchronize()

    return tensor if not is_send else None

def train(step: int, sharded_model: nn.Module, batch_loader: DataLoader, loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, device: torch.device, world: World, config: PipelineConfig) -> Outputs:
    start = time.time()
    sharded_model.train()
    sharded_model.to(device)
    optimizer.zero_grad()
    
    # Initialize batch loss
    batch_loss = torch.Tensor([0.0]).to(device)
    batch_tokens, batch_examples = 0, 0

    # Initialize input and output tensors (for communication)
    input_tensors, output_tensors = [], []
    tensor_shapes = (1, 128, 768)
    num_micro_batches = len(batch_loader)
    grad_accumulation_steps = len(batch_loader)
    for _ in range(num_micro_batches): # All forward passes
        micro_batch = next(iter(batch_loader))
        input_tensor = communicate(world, step, operation="recv_forward", shapes=tensor_shapes, dtype=torch.float32)
        micro_batch["hidden_states"] = input_tensor
        micro_batch = {k: v.to(device) if v is not None else v for k, v in micro_batch.items()}
        output_tensor = sharded_model.forward(micro_batch)
        communicate(world, step, operation='send_forward', tensor=output_tensor)

        if world.is_last_stage:
            loss = loss_fn(output_tensor.transpose(1, 2), micro_batch["target_ids"].to(device))
            loss /= grad_accumulation_steps
            batch_loss += loss
            batch_examples += micro_batch["input_ids"].shape[0]
            batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    for _ in range(num_micro_batches): # All backward passes
        output_tensor_grad = communicate(world, step, operation='recv_backward', shapes=tensor_shapes, dtype=torch.float32)
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = sharded_model.backward(input_tensor, output_tensor, output_tensor_grad)
        communicate(world, step, operation="send_backward", tensor=input_tensor_grad)
    
    # Update model parameters
    optimizer.step()

    # Synchronize timing
    torch.cuda.synchronize()
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / grad_accumulation_steps

    return Outputs(step=step, lr=None, loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=None, time=step_time, micro_step_time=micro_step_time)


def eval() -> Outputs:
    pass

def sample() -> List[str]:
    pass

def main(config: PipelineConfig):
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
    # val_dataloader = get_dataloader(val_data, batch_size=config.train.micro_batch_size, shuffle=True, cycle=True)
    # test_dataloader = get_dataloader(test_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=True)

    # TODO: Compute based on config
    num_train_steps = config.train.max_steps
    
    # Setup tensor shapes for communication
    B, L, H = config.train.micro_batch_size, config.data.seq_length, model.config.hidden_size
    tensor_shape = (B, L, H)
    logger.log_message(f"Tensor shapes: {tensor_shape}")

    # Set up optimizer
    optimizer = get_optimizer(config.train, sharded_model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Loss(), Perplexity(), Throughput()], name="train")
    train_bar = None
    if world.is_last_stage:
        train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
    
    for train_step in range(1, num_train_steps+1):
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(train_step, sharded_model, micro_batchloader, loss_fn, optimizer, scheduler, device, world, config)

        # Compute and log metrics
        if world.is_last_stage:
            train_metrics.update(outputs)
            curr_metrics = train_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
            train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))
            train_bar.update()

    # Destroy process group
    if world.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    disable_progress_bar()
    main(PipelineConfig(**parse_argv()))