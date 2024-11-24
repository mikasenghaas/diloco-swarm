"""
SWARM Parallel LLM Pre-Training.

torchrun --nproc_per_node 4 src/train/swarm.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --logging.console.enable false --logging.file.enable true
"""
import autorootcwd

import os
import time
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from datasets import disable_progress_bar

from src.logger import Level
# from src.ckpt import Checkpoint
from src.world import SwarmWorld
from src.serializer import SwarmSerializer
from src.comm import SwarmComm
from src.sampler import BatchData, DistributedSampler
from src.config import WorldConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import (
    seed_everything, get_world, get_logger, get_device, get_model, get_sharded_model,
    get_tokenizer, get_dataset, tokenize, get_dataloader, get_optimizer, 
    get_scheduler, get_num_steps, get_train_setup, format_int, format_float, 
    get_train_pbar_description, 
)
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Examples, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate
from pydantic_config import BaseConfig, parse_argv

TIMEOUT = 0.001  # Critical to tune

# Reuse SwarmWorld, SwarmComm and Serializer from min_swarm.py
class SwarmConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    world: WorldConfig = WorldConfig()
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

def train(
    step: int,
    sharded_model: nn.Module, 
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    world: SwarmWorld,
    comm: SwarmComm,
    config: TrainConfig,
    device: torch.device
) -> Outputs:
    # Prepare model
    start = time.time()
    sharded_model.to(device)
    sharded_model.train()
    optimizer.zero_grad()

    # Setup step
    num_micro_steps = config.batch_size // config.micro_batch_size
    world.setup_step(step, num_micro_steps)

    # Prepare batch batch
    batch_data = BatchData(batch)
    micro_batches = {} # (rank, local_micro_step) -> micro_batch
    for rank in world.get_stage_ranks(0):
        micro_sampler = DistributedSampler(batch_data, rank=rank, ranks=world.get_stage_ranks(0), micro_batch_size=config.micro_batch_size)
        micro_dataloader = iter(DataLoader(batch_data, batch_size=config.micro_batch_size, sampler=micro_sampler, shuffle=False))
        for local_micro_step, micro_batch in enumerate(micro_dataloader, start=1):
            if world.is_last_stage: micro_batches[(rank, local_micro_step)] = micro_batch
            elif world.is_first_stage and rank == world.rank: comm.load_forward_queue(local_micro_step, micro_batch["input_ids"])
    
    # Prepare statistics
    batch_loss = 0.0
    batch_tokens, batch_examples = 0, 0
    input_output_tensors = {}

    # Zero gradients
    optimizer.zero_grad()

    while not world.step_done(step):
        if comm.can_receive_forward():
            src, input_tensor, (root, local_micro_step) = comm.recv_forward()
            output_tensor = sharded_model(input_tensor.to(device))
            input_output_tensors[(root, local_micro_step)] = (src, input_tensor, output_tensor)
            comm.send_forward(output_tensor, (root, local_micro_step))

            if world.is_last_stage:
                # Reshape logits and targets for loss calculation
                micro_batch = micro_batches[(root, local_micro_step)]
                mask, target_ids = micro_batch["attention_mask"].to(device), micro_batch["target_ids"].to(device)
                logits_flat = output_tensor.view(-1, output_tensor.size(-1))  # (B*L, V)
                targets_flat = target_ids.detach().view(-1)  # (B*L)
                
                # Apply mask by filtering out padded positions
                mask_flat = mask.view(-1)  # (B*L)
                logits_filtered = logits_flat[mask_flat.bool()]  # ((B*L)', V)
                targets_filtered = targets_flat[mask_flat.bool()]  # ((B*L)')

                loss = loss_fn(logits_filtered, targets_filtered)
                loss = loss / num_micro_steps
                
                # Update statistics
                batch_loss += loss.detach().item()
                batch_examples += input_tensor.shape[0]
                batch_tokens += input_tensor.shape[0] * input_tensor.shape[1]
                
                # Backward pass
                input_tensor_grad = sharded_model.backward(input_tensor, loss, None)
                comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

        if comm.can_receive_backward():
            _, output_tensor_grad, (root, local_micro_step) = comm.recv_backward()
            src, input_tensor, output_tensor = input_output_tensors[(root, local_micro_step)]
            input_tensor_grad = sharded_model.backward(input_tensor if not world.is_first_stage else None, output_tensor, output_tensor_grad)
            comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

            if world.is_first_stage:
                world.micro_step_done()

    # Sync gradients across ranks in same stage
    comm.sync_gradients()
    
    # Optimizer steps
    norm = torch.nn.utils.clip_grad_norm_(sharded_model.parameters(), config.max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    # Timing
    torch.cuda.synchronize()
    end = time.time()
    step_time = end - start
    micro_step_time = step_time / num_micro_steps

    local_outputs = Outputs(
        step=step,
        lr=lr,
        loss=batch_loss,
        num_tokens=batch_tokens,
        num_examples=batch_examples,
        norm=norm,
        time=step_time,
        micro_step_time=micro_step_time
    )

    stage_outputs = comm.sync_outputs(local_outputs)
    return local_outputs, stage_outputs

def main(config: SwarmConfig):
    # Seed everything
    seed_everything(config.train.seed)

    # Get world parameters
    local_rank, world_size = get_world()

    # Set device
    device = get_device(local_rank)
    torch.cuda.set_device(local_rank)

    # Get world
    world = SwarmWorld(config.world)

    # Get logger
    logger_name = f"{local_rank}" if world_size > 1 else "master"
    logger = get_logger(config.logging, logger_name, world.run_id)
    
    # Log values
    logger.log_config(config)
    logger.log_world(world)

    # Get checkpoint
    # if config.logging.ckpt.enable:
    #     ckpt = Checkpoint(logger.checkpoint_dir)
    #     ckpt.setup(world)
    #     logger.log_message(f"Checkpoint directory: {ckpt.base_dir}")

    # Set precision
    torch.set_float32_matmul_precision(config.train.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}")

    # Load model and create sharded version
    model = get_model(config.model)
    logger.log_message(f"Loaded model ({format_int(model.num_parameters(), 2)} parameters)")
    
    sharded_model = get_sharded_model(model, world)
    logger.log_message(f"Sharded model ({format_int(sharded_model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded tokenizer ({format_int(len(tokenizer), 0)} vocab size)")

    # Load dataset
    train_data = get_dataset(config.data, split="train")
    logger.log_message(f"Loaded dataset with {format_int(len(train_data))} examples")

    # Prepare dataset
    seq_length = config.data.seq_length + 1
    train_data = train_data.map(lambda examples: tokenize(examples["text"], tokenizer, seq_length), batched=True, num_proc=min(len(train_data), os.cpu_count()))

    # Setup training
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=True)
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(sharded_model, config.train.optimizer)
    scheduler = get_scheduler(optimizer, num_train_steps, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize communication
    B, L, H = config.train.micro_batch_size, config.data.seq_length, model.config.n_embd
    serializer = SwarmSerializer(shape=(B, L, H))
    comm = SwarmComm(sharded_model, world, serializer, serializer.shape, device, logger)

    # Training metrics
    train_metrics = Metrics([Step(), Time(), MicroTime(), Examples(), Tokens(), Loss(), Perplexity(), Throughput(), Norm(), LearningRate()], name="train")

    # Training loop
    train_range = range(1, num_train_steps + 1)
    train_bar = tqdm(train_range, position=0, leave=True) if world.is_last_stage else None
    for train_step in train_range:
        # Train step
        batch = next(train_dataloader)
        _, stage_outputs = train(train_step, sharded_model, batch, loss_fn, optimizer, scheduler, world, comm, config.train, device)

        # Update metrics
        if world.is_last_stage and world.is_leader:
            train_metrics.update(stage_outputs)
            curr_metrics = train_metrics.compute()
            logger.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
            train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))
            train_bar.update()

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    disable_progress_bar()
    main(SwarmConfig(**parse_argv())) 