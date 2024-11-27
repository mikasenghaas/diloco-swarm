"""
Simple SWARM Parallel LLM Pre-Training.

Single GPU: ```torchrun --nproc_per_node 1 src/train/swarm.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --world.num_stages 1```
DP:         ```torchrun --nproc_per_node 2 src/train/swarm.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --world.num_stages 1```
PP:         ```torchrun --nproc_per_node 2 src/train/swarm.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --world.num_stages 2```
SWARM:      ```torchrun --nproc_per_node 4 src/train/swarm.py @configs/debug.toml --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml --world.num_stages 2```
"""
import autorootcwd

import time
from tqdm import tqdm
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pydantic_config import BaseConfig, parse_argv

from src.logger import Level
from src.world import World
from src.serializer import Serializer
from src.comm import Comm
from src.sampler import BatchData, BatchSampler
from src.config import WorldConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model, get_sharded_model, get_tokenizer, get_dataset, tokenize, get_dataloader, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description, filter_logits_targets
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate

TIMEOUT = 0.01
LOGGER = None # Global logger

class SwarmConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    world: WorldConfig = WorldConfig()
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

def train(train_step: int, model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, world: World, comm: Comm, device: torch.device, config: TrainConfig) -> Outputs:
    # Prepare model
    start = time.time()
    model.to(device)
    model.train()
    optimizer.zero_grad()

    # Setup step
    num_micro_steps = config.batch_size // config.micro_batch_size
    world.setup_step(train_step, num_micro_steps=num_micro_steps)

    # Prepare batch batch
    batch_data = BatchData(batch)
    micro_batches = {} # (rank, local_micro_step) -> micro_batch
    first_stage_ranks = world.get_stage_ranks(0)
    for rank in first_stage_ranks:
        micro_sampler = BatchSampler(batch_data, rank=rank, ranks=first_stage_ranks, micro_batch_size=config.micro_batch_size)
        micro_dataloader = DataLoader(batch_data, batch_size=config.micro_batch_size, shuffle=False, sampler=micro_sampler)
        for local_micro_step, micro_batch in enumerate(micro_dataloader, start=1):
            if world.is_last_stage: micro_batches[(rank, local_micro_step)] = micro_batch
            if world.is_first_stage and rank == world.rank: comm.load_forward_queue(local_micro_step, micro_batch["input_ids"])
    
    # Prepare statistics
    batch_loss, batch_tokens = 0.0, 0
    input_output_tensors = {}

    # Zero gradients
    optimizer.zero_grad()
    LOGGER.log_message(f"Starting train step {train_step} (batch_size: {config.batch_size}, micro_batch_size: {config.micro_batch_size}, num_micro_steps: {num_micro_steps})", Level.DEBUG)
    while True:
        if world.step_done(train_step): break
        if comm.can_receive_forward():
            src, input_tensor, (root, local_micro_step) = comm.recv_forward()
            output_tensor = model(input_tensor.to(device))
            input_output_tensors[(root, local_micro_step)] = (src, input_tensor, output_tensor)
            comm.send_forward(output_tensor, (root, local_micro_step))

            if world.is_last_stage:
                # Reshape logits and targets for loss calculation
                micro_batch = micro_batches[(root, local_micro_step)]
                mask, target_ids = micro_batch["attention_mask"].to(device), micro_batch["target_ids"].to(device)
                logits_filtered, targets_filtered = filter_logits_targets(output_tensor, target_ids, mask)
                
                loss = loss_fn(logits_filtered, targets_filtered)
                loss = loss / num_micro_steps
                
                # Update statistics
                batch_loss += loss.detach().item()
                batch_tokens += input_tensor.shape[0] * input_tensor.shape[1]
                
                # Backward pass
                input_tensor_grad = model.backward(input_tensor if not world.is_first_stage else None, loss, None)
                comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

                if world.is_first_stage:
                    if rank == 0: time.sleep(TIMEOUT)
                    world.micro_step_done()

        if comm.can_receive_backward():
            _, output_tensor_grad, (root, local_micro_step) = comm.recv_backward()
            src, input_tensor, output_tensor = input_output_tensors[(root, local_micro_step)]
            input_tensor_grad = model.backward(input_tensor if not world.is_first_stage else None, output_tensor, output_tensor_grad)
            comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

            if world.is_first_stage: world.micro_step_done()
        time.sleep(TIMEOUT)

    # Sync gradients across ranks in same stage
    LOGGER.log_message(f"Syncing gradients", Level.DEBUG)
    s = time.time(); comm.sync_gradients()
    
    # Optimizer steps
    LOGGER.log_message(f"Step optimizer and scheduler", Level.DEBUG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    # Timing
    torch.cuda.synchronize()
    step_time = time.time() - start
    micro_step_time = step_time / num_micro_steps

    # Compute and sync outputs
    local_outputs = Outputs( step=train_step, lr=lr, loss=batch_loss, tokens=batch_tokens, norm=norm, time=step_time, micro_step_time=micro_step_time)
    stage_outputs = comm.sync_outputs(local_outputs)

    return local_outputs, stage_outputs

def eval(eval_step: int, eval_type: Literal["eval", "test"], model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device, world: World, comm: Comm) -> Outputs:
    # Prepare model
    start = time.time()
    model.to(device)
    model.eval()

    # Prepare batch batch
    batch_data = BatchData(batch)

    # Setup step
    micro_batch_size = len(batch_data)
    num_micro_steps = 1
    world.setup_step(eval_step, num_micro_steps=num_micro_steps, type=eval_type) # Assume batch size = micro batch size

    micro_batches = {} # (rank, local_micro_step) -> micro_batch
    first_stage_ranks = world.get_stage_ranks(0)
    for rank in first_stage_ranks:
        micro_sampler = BatchSampler(batch_data, rank, first_stage_ranks, micro_batch_size)
        micro_dataloader = DataLoader(batch_data, batch_size=micro_batch_size, shuffle=False, sampler=micro_sampler)
        for local_micro_step, micro_batch in enumerate(micro_dataloader):
            if world.is_last_stage: micro_batches[(rank, local_micro_step)] = micro_batch
            if world.is_first_stage and rank == world.rank: comm.load_forward_queue(local_micro_step, micro_batch["input_ids"])
    
    # Prepare statistics
    batch_loss, batch_tokens = 0.0, 0
    input_output_tensors = {}

    LOGGER.log_message(f"{eval_type.capitalize()} step {eval_step}", Level.DEBUG)
    while not world.step_done(eval_step, type=eval_type):
        if comm.can_receive_forward():
            src, input_tensor, (root, local_micro_step) = comm.recv_forward()
            output_tensor = model(input_tensor.to(device))
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
                batch_tokens += input_tensor.shape[0] * input_tensor.shape[1]

                LOGGER.log_message(f"{eval_type.capitalize()} micro step done", Level.DEBUG)
                world.micro_step_done(eval_type)

    # Timing
    torch.cuda.synchronize()
    step_time = time.time() - start

    outputs = Outputs(step=eval_step, time=step_time, loss=batch_loss, tokens=batch_tokens)
    stage_outputs = comm.sync_outputs(outputs)

    return outputs, stage_outputs

def eval_loop(eval_type: Literal["eval", "test"], model: nn.Module, loss_fn: nn.Module, eval_dataloader: DataLoader, eval_metrics: Metrics, world: World, comm: Comm, device: torch.device, eval_config: EvalConfig) -> Outputs:
    eval_range = range(1, eval_config.max_steps + 1)
    eval_bar = tqdm(eval_range, position=1, leave=False if eval_type == "eval" else True) if world.is_leader and world.is_last_stage else None
    eval_metrics.reset()
    for eval_step in eval_range:
        batch = next(eval_dataloader)
        _, stage_outputs = eval(eval_step, eval_type, model, batch, loss_fn, device, world, comm, TIMEOUT)
        eval_metrics.update(stage_outputs)
        
        if not (world.is_leader and world.is_last_stage): continue
        eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]" if eval_type == "eval" else "[TEST]"))
    
    final_metrics = eval_metrics.compute()
    return final_metrics

def sample_loop():
    pass

def main(config: SwarmConfig):
    global LOGGER
    # Seed everything
    seed_everything(config.train.seed)

    # Get world parameters
    local_rank, world_size = get_world()

    # Set device
    device = get_device(local_rank)
    torch.cuda.set_device(local_rank)

    # Get world
    world = World(config.world)

    # Get logger
    logger_name = f"{local_rank}" if world_size > 1 else "master"
    LOGGER = get_logger(config.logging, logger_name, world.run_id)
    
    # Log values
    LOGGER.log_config(config)
    LOGGER.log_world(world)

    # Get checkpoint
    # if config.logging.ckpt.enable:
    #     ckpt = Checkpoint(logger.checkpoint_dir)
    #     ckpt.setup(world)
    #     logger.log_message(f"Checkpoint directory: {ckpt.base_dir}")

    # Set precision
    torch.set_float32_matmul_precision(config.train.amp.precision)
    LOGGER.log_message(f"Using precision: {torch.get_float32_matmul_precision()}")

    # Load model and create sharded version
    model = get_model(config.model)
    LOGGER.log_message(f"Loaded model ({format_int(model.num_parameters(), 2)} parameters)")
    
    model = get_sharded_model(model, world)
    LOGGER.log_message(f"Sharded model ({format_int(model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    LOGGER.log_message(f"Loaded tokenizer ({format_int(len(tokenizer), 0)} vocab size)")

    # Load
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    LOGGER.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Prepare dataset
    seq_length = config.data.seq_length + 1
    train_data = train_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    val_data = val_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    test_data = test_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    LOGGER.log_message(f"Tokenized dataset with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Setup training
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=False, cycle=True)
    eval_dataloader = get_dataloader(val_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=False)
    test_dataloader = get_dataloader(test_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=False)

    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.train.micro_batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.train.micro_batch_size)

    # Get training, evaluation and testing setup
    train_setup = get_train_setup(num_train_steps, config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(train_data))
    eval_setup = get_train_setup(num_eval_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(val_data))
    test_setup = get_train_setup(num_test_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(test_data))
    LOGGER.log_message("Train setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in train_setup.items()]))
    LOGGER.log_message("Eval setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in eval_setup.items()]))
    LOGGER.log_message("Test setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in test_setup.items()]))

    # Get optimizer and scheduler
    optimizer = get_optimizer(model, config.train.optimizer)
    scheduler = get_scheduler(optimizer, num_train_steps, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize metrics
    train_metrics = Metrics([Step(), Time(), MicroTime(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Throughput(), Loss(), Perplexity()], name="eval")
    test_metrics = Metrics([Throughput(), Loss(), Perplexity()], name="test")

    # Initialize communication
    B, L, H = config.train.micro_batch_size, config.data.seq_length, model.config.n_embd
    serializer = Serializer(shape=(B, L, H))
    comm = Comm(model, world, serializer, serializer.shape, device, LOGGER, TIMEOUT)

    # Initialize training progress bar
    train_range = range(1, num_train_steps + 1)
    train_bar = tqdm(train_range, position=0, leave=True) if world.is_leader and world.is_last_stage else None
    if world.is_leader and world.is_last_stage:
        train_metrics.update(Outputs(step=0, time=1e-4, micro_step_time=1e-4, loss=0, tokens=0))
        train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))

    # Validate before training
    if config.eval.enable:
        outputs = eval_loop("eval", model, loss_fn, eval_dataloader, eval_metrics, world, comm, device, config.eval)
        if world.is_leader and world.is_last_stage:
            LOGGER.log_metrics(outputs, level=Level.DEBUG, step=0)

    # Sample before training
    # if config.sample.enable:
    #     samples = sample(model, tokenizer, config.sample, device)
    #     logger.log_samples(0, samples)

    # Training loop
    train_metrics.reset()
    for train_step in train_range:
        # Train step
        batch = next(train_dataloader)
        _, stage_outputs = train(train_step, model, batch, loss_fn, optimizer, scheduler, world, comm, device, config.train)

        # Update metrics
        if world.is_leader and world.is_last_stage:
            train_metrics.update(stage_outputs)
            curr_metrics = train_metrics.compute()
            LOGGER.log_metrics(curr_metrics, level=Level.DEBUG, step=train_step)
            train_bar.set_description(get_train_pbar_description(train_metrics, prefix="[TRAIN]"))
            train_bar.update()

        # Validate
        if config.eval.enable and config.eval.every_n_steps > 0 and train_step % config.eval.every_n_steps == 0:
            outputs = eval_loop("eval", model, loss_fn, eval_dataloader, eval_metrics, world, comm, device, config.eval)
            if world.is_leader and world.is_last_stage:
                LOGGER.log_metrics(outputs, level=Level.DEBUG, step=train_step)

        # Sample
        # if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
        #     samples = sample(model, tokenizer, config.sample, device)
        #     logger.log_samples(train_step, samples)

        # Checkpoint
        # if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
        #     ckpt_dir = ckpt.save(train_step, model)
        #     logger.log_message(f"Saved model checkpoint at {ckpt_dir}")

    if config.eval.enable:
        outputs = eval_loop("test", model, loss_fn, test_dataloader, test_metrics, world, comm, device, config.eval)
        if world.is_leader and world.is_last_stage:
            LOGGER.log_metrics(outputs, level=Level.DEBUG, step=train_step)

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    from datasets import disable_progress_bar; disable_progress_bar()
    main(SwarmConfig(**parse_argv())) 