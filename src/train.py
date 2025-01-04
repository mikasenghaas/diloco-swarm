"""
SWARM Parallel LLM Pre-Training.

Single GPU: torchrun --nproc_per_node 1 src/train.py --swarm.num_stages 1 --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml
DP:         torchrun --nproc_per_node 2 src/train.py --swarm.num_stages 2 --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml
PP:         torchrun --nproc_per_node 2 src/train.py --swarm.num_stages 2 --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml
SWARM:      torchrun --nproc_per_node 4 src/train.py --swarm.num_stages 2 --model @configs/model/gpt2-small.toml --data @configs/data/wikitext.toml
"""
import autorootcwd

import time
from tqdm import tqdm
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pydantic_config import BaseConfig, parse_argv
from transformers import AutoTokenizer

from src.logger import Level
from src.world import World
from src.comm import TrainingComm, InferenceComm
from src.config import SwarmConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig, AmpConfig
from src.utils import seed_everything, get_logger, get_device, get_dtype, get_model, get_sharded_model, initialize_gradients, get_tokenizer, get_dataset, tokenize, get_dataloader, get_micro_batches, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description, filter_logits_targets, nullcontext, get_outer_model, compute_pseudo_gradient, sync_inner_model
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Tokens, NumMicroBatches, Loss, Perplexity, Throughput, Norm, LearningRate

logger = None
timeout = 1e-4

class SwarmConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    device: str | None = None
    amp: AmpConfig = AmpConfig()
    swarm: SwarmConfig = SwarmConfig()
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

@torch.no_grad()
def sample_loop(step: int, model: nn.Module, tokenizer: AutoTokenizer, world: World, inference_comm: InferenceComm, prompt_length: int, config: SampleConfig, device: torch.device) -> List[str]:
    """Sample loop for generating tokens"""
    model.to(device); model.eval()

    # Prepare input
    tensor_length = prompt_length + config.max_new_tokens
    input_ids = tokenize(config.prompt, tokenizer, max_length=tensor_length)["input_ids"].to(device).repeat(config.num_samples, 1)
    micro_batch = {"input_ids": input_ids, "attention_mask": None, "hidden_states": None}
    if world.is_leader and world.is_first_stage: inference_comm.load(input_ids)

    generated_id = prompt_length; world.setup_step(step, num_micro_steps=config.max_new_tokens, type="sample")
    while world.is_leader and not world.is_step_done(step, type="sample"):
        if not inference_comm.recv_thread.can_receive: time.sleep(timeout); continue
        tensor_type, input_tensor = inference_comm.receive()
        micro_batch[tensor_type] = input_tensor
        output_tensor = model.forward(micro_batch, device)
        if world.is_last_stage:
            probs = F.softmax(output_tensor[:, generated_id-1, :], dim=-1) # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            input_ids[:, generated_id] = idx_next.squeeze()
            output_tensor = input_ids
            world.micro_step_done(type="sample")
            if (idx_next == tokenizer.eos_token_id).all(): world.step_done(type="sample"); break
            if generated_id >= tensor_length-1: break
            generated_id += 1
        inference_comm.send(output_tensor)

    return [tokenizer.decode(generated_ids, skip_special_tokens=True) for generated_ids in input_ids]

def eval_step(step: int, eval_type: Literal["eval", "test"], model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device, world: World, training_comm: TrainingComm, config: SwarmConfig) -> Outputs:
    """Evaluation step on eval batch"""
    # Prepare model
    start = time.time()
    model.to(device); model.eval()

    num_micro_steps = config.train.batch_size // config.train.micro_batch_size
    num_micro_steps_per_device = max(1, num_micro_steps // (len(world.stage2ranks[world.num_stages-1]))) # TODO: How to scale w/ heterogenity?
    tokens_per_micro_batch = config.train.micro_batch_size * config.data.seq_length
    logger.log_message(f"Setup eval step {step} in world", master=False, level=Level.DEBUG)
    world.setup_step(step, num_micro_steps=num_micro_steps, type=eval_type)

    micro_batches = {}
    logger.log_message(f"Preparing micro batch distribution", master=False, level=Level.DEBUG)
    for rank, local_micro_step, micro_batch in get_micro_batches(batch, config.train.micro_batch_size, world):
        micro_batches[(rank, local_micro_step)] = micro_batch
        if world.is_first_stage and rank == world.rank:
            training_comm.load_forward(metadata=(rank, local_micro_step))
    
    # Start evaluation step
    local_batch_loss, local_batch_tokens, local_num_micro_batches = 0.0, 0, 0
    logger.log_message(f"{eval_type.capitalize()} step {step}", master=False, level=Level.DEBUG)
    while not world.is_step_done(step, type=eval_type):
        if not (training_comm.forward_recv_thread.can_receive): time.sleep(timeout); continue
        # Receive input tensor
        _, input_tensor, (root, local_micro_step) = training_comm.recv_forward()
        micro_batch = micro_batches[(root, local_micro_step)]
        micro_batch["hidden_states"] = input_tensor

        # Update statistics
        local_num_micro_batches += 1
        local_batch_tokens += tokens_per_micro_batch

        # Forward pass
        with torch.amp.autocast(device_type=device.type, dtype=get_dtype(config.amp.dtype)) if config.amp.enable else nullcontext():
            output_tensor = model.forward(micro_batch, device)

        # Send output tensor
        training_comm.send_forward(tensor=output_tensor, metadata=(root, local_micro_step))

        if world.is_last_stage:
            # Filter logits and targets
            attention_mask, target_ids = micro_batch["attention_mask"].to(device), micro_batch["target_ids"].to(device)
            logits_filtered, targets_filtered = filter_logits_targets(output_tensor, target_ids, attention_mask)

            # Compute loss
            loss = loss_fn(logits_filtered, targets_filtered)
            loss = loss / num_micro_steps_per_device
            logger.log_message(f"Computed local loss: {loss.item()}", master=False, level=Level.DEBUG)

            # Update statistics
            local_batch_loss += loss.detach().item()

            world.micro_step_done(eval_type)

    # Timing
    logger.log_message(f"{eval_type.capitalize()} step {step} done", master=False, level=Level.DEBUG)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    step_time = time.time() - start

    # Sync outputs
    local_outputs = Outputs(step=step, time=step_time, loss=local_batch_loss, tokens=local_batch_tokens, num_micro_batches=local_num_micro_batches)
    stage_outputs = training_comm.sync_outputs(local_outputs)
    logger.log_message(f"Aggregated outputs={stage_outputs}", master=False, level=Level.DEBUG)

    return local_outputs, stage_outputs

@torch.no_grad()
def eval_loop(eval_type: Literal["eval", "test"], model: nn.Module, loss_fn: nn.Module, eval_dataloader: DataLoader, eval_metrics: Metrics, world: World, training_comm: TrainingComm, device: torch.device, config: SwarmConfig) -> Outputs:
    """Evaluation loop on eval data loader"""
    eval_range = range(1, len(eval_dataloader) + 1)
    eval_bar = tqdm(eval_range, position=1, leave=False) if world.is_master else None
    eval_metrics.reset()
    for step, batch in enumerate(eval_dataloader, start=1):
        _, stage_outputs = eval_step(step, eval_type, model, batch, loss_fn, device, world, training_comm, config)
        
        if world.is_master:
            eval_metrics.update(stage_outputs)
            eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]" if eval_type == "eval" else "[TEST]"))
            eval_bar.update()

    return eval_metrics.compute()

def train_step(step: int, num_train_steps: int, inner_model: nn.Module, outer_model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, inner_optimizer: Optimizer, outer_optimizer: Optional[Optimizer], scheduler: LambdaLR, world: World, training_comm: TrainingComm, device: torch.device, config: SwarmConfig) -> Outputs:
    """Training step on train batch"""
    # Prepare model
    start = time.time()
    inner_model.to(device); inner_model.train()
    inner_optimizer.zero_grad()

    # Setup step
    num_micro_steps = config.train.batch_size // config.train.micro_batch_size
    num_micro_steps_per_device = max(1, num_micro_steps // (len(world.stage2ranks[world.num_stages-1]))) # TODO: How to scale w/ heterogenity?
    tokens_per_micro_batch = config.train.micro_batch_size * config.data.seq_length
    logger.log_message(f"Setup train step {step} in world", master=False, level=Level.DEBUG)
    world.setup_step(step, num_micro_steps=num_micro_steps)

    micro_batches = {}
    logger.log_message(f"Preparing micro batch distribution", master=False, level=Level.DEBUG)
    for rank, local_micro_step, micro_batch in get_micro_batches(batch, config.train.micro_batch_size, world):
        micro_batches[(rank, local_micro_step)] = micro_batch
        if world.is_first_stage and rank == world.rank:
            training_comm.load_forward(metadata=(rank, local_micro_step))

    # Start training step
    local_batch_loss, local_batch_tokens, local_num_micro_batches = 0.0, 0, 0
    input_output_tensors = {}
    logger.log_message(f"Starting train step {step}", master=False, level=Level.DEBUG)
    while True:
        if world.is_step_done(step): break
        if time.time() - start > config.train.step_timeout: logger.log_message(f"Train step {step} timed out", master=False, level=Level.WARNING); break
        if not (training_comm.forward_recv_thread.can_receive or training_comm.backward_recv_thread.can_receive): time.sleep(timeout); continue
        if training_comm.forward_recv_thread.can_receive and len(input_output_tensors) < config.train.max_micro_batches:
            # Receive input tensor
            src, input_tensor, (root, local_micro_step) = training_comm.recv_forward()
            micro_batch = micro_batches[(root, local_micro_step)]
            micro_batch["hidden_states"] = input_tensor

            # Update statistics
            local_num_micro_batches += 1
            local_batch_tokens += tokens_per_micro_batch
            
            # Forward pass
            with torch.amp.autocast(device_type=device.type, dtype=get_dtype(config.amp.dtype)) if config.amp.enable else nullcontext():
                output_tensor = inner_model.forward(micro_batch, device)

            # Remember tensors for backward in all but last stage
            if world.num_stages > 1 and not world.is_last_stage: # Maybe don't optimize non-pipeline for comparability
                logger.log_message(f"Offloading input and output tensors to CPU", master=False, level=Level.DEBUG)
                if input_tensor is None: input_tensor = micro_batch["input_ids"]
                input_output_tensors[(root, local_micro_step)] = (src, input_tensor, output_tensor)

            # Send output tensor
            training_comm.send_forward(tensor=output_tensor, metadata=(root, local_micro_step))

            if world.is_last_stage:
                # Filter logits and targets for loss calculation
                mask, target_ids = micro_batch["attention_mask"].to(device), micro_batch["target_ids"].to(device)
                logits_filtered, targets_filtered = filter_logits_targets(output_tensor, target_ids, mask)
                
                # Compute loss
                loss = loss_fn(logits_filtered, targets_filtered)
                loss = loss / num_micro_steps_per_device
                logger.log_message(f"Computed local loss: {loss.item()}", master=False, level=Level.DEBUG)
                
                # Update statistics
                local_batch_loss += loss.detach().item()
                
                # Backward pass
                input_tensor_grad = inner_model.backward(input_tensor if not world.is_first_stage else None, loss, None, device)
                training_comm.send_backward(dst=src, tensor=input_tensor_grad, metadata=(root, local_micro_step))

                if world.is_first_stage: world.micro_step_done()

        elif training_comm.backward_recv_thread.can_receive:
            # Receive output tensor gradient
            _, output_tensor_grad, (root, local_micro_step) = training_comm.recv_backward()

            # Get input and output tensors
            src, input_tensor, output_tensor = input_output_tensors.pop((root, local_micro_step))

            # Backward pass
            input_tensor_grad = inner_model.backward(input_tensor if not world.is_first_stage else None, output_tensor, output_tensor_grad, device)
            training_comm.send_backward(dst=src, tensor=input_tensor_grad, metadata=(root, local_micro_step))
        
            if world.is_first_stage: world.micro_step_done()
        
    # Synchronize
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dist.barrier()

    # Sync gradients 
    do_sync = (config.swarm.sync_every_n_steps > 0 and step % config.swarm.sync_every_n_steps == 0) or step == num_train_steps
    if do_sync and outer_optimizer is None:
        logger.log_message(f"No outer optimizer, syncing gradients directly", master=False, level=Level.DEBUG)
        training_comm.sync_gradients(inner_model)

    # Optimizer steps
    logger.log_message(f"Step inner optimizer and scheduler", master=False, level=Level.DEBUG)
    norm = torch.nn.utils.clip_grad_norm_(inner_model.parameters(), config.train.max_norm)
    lr = scheduler.get_last_lr()[0]
    inner_optimizer.step()
    scheduler.step()

    # Sync gradients across ranks in same stage
    if do_sync and outer_optimizer is not None:
        logger.log_message(f"Computing pseudo gradient", master=False, level=Level.DEBUG)
        compute_pseudo_gradient(inner_model, outer_model)
        logger.log_message(f"Syncing pseudo gradient", master=False, level=Level.DEBUG)
        training_comm.sync_gradients(outer_model)
        logger.log_message(f"Step outer optimizer", master=False, level=Level.DEBUG)
        outer_optimizer.step()
        logger.log_message(f"Syncing inner model", master=False, level=Level.DEBUG)
        sync_inner_model(outer_model, inner_model)

    # Timing
    step_time = time.time() - start
    micro_step_time = step_time / num_micro_steps_per_device

    # Compute and sync outputs
    local_outputs, stage_outputs = Outputs(step=step, lr=lr, loss=local_batch_loss, tokens=local_batch_tokens, norm=norm, time=step_time, micro_step_time=micro_step_time, num_micro_batches=local_num_micro_batches), None
    if do_sync:
        logger.log_message(f"Syncing outputs", master=False, level=Level.DEBUG)
        stage_outputs = training_comm.sync_outputs(local_outputs)

    return local_outputs, stage_outputs

def train_loop(num_train_steps: int, inner_model: nn.Module, outer_model: nn.Module, tokenizer: AutoTokenizer, train_dataloader: DataLoader, eval_dataloader: DataLoader, loss_fn: nn.Module, inner_optimizer: Optimizer, outer_optimizer: Optimizer, scheduler: LambdaLR, world: World, device: torch.device, config: SwarmConfig) -> Outputs:
    # Initialize metrics
    local_train_metrics = Metrics([Step(), Time(), MicroTime(), Tokens(), NumMicroBatches(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train/local")
    global_train_metrics = Metrics([Step(), Time(), MicroTime(), Tokens(), NumMicroBatches(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train/global")
    eval_metrics = Metrics([Throughput(), Loss(), Perplexity()], name="eval")

    # Initialize training communication
    training_shape = (config.train.micro_batch_size, config.data.seq_length, inner_model.config.n_embd)
    training_comm = TrainingComm(world, training_shape, logger)

    # Initialize inference communication
    if config.sample.enable:
        prompt_length = tokenize(config.sample.prompt, tokenizer)["input_ids"].shape[1]
        inference_shape = (config.sample.num_samples, prompt_length+config.sample.max_new_tokens, inner_model.config.n_embd)
        inference_comm = InferenceComm(world, inference_shape, logger)

    # Sample before training
    if config.sample.enable:
        samples = sample_loop(0, inner_model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
        logger.log_samples(samples, step=0)

    # Validate before training
    if config.eval.enable:
        outputs = eval_loop("eval", inner_model, loss_fn, eval_dataloader, eval_metrics, world, training_comm, device, config)
        logger.log_metrics(outputs, step=0, master=True)

    logger.log_message(f"Starting training", master=False, level=Level.DEBUG)
    train_range = range(1, num_train_steps + 1)
    train_bar = tqdm(train_range, position=0, leave=False) if world.is_master else None
    train_iter = iter(train_dataloader)
    for step in train_range:
        logger.log_message(f"Preparing batch for train step {step}", master=False, level=Level.DEBUG)
        batch = next(train_iter)
        local_outputs, stage_outputs = train_step(step, num_train_steps, inner_model, outer_model, batch, loss_fn, inner_optimizer, outer_optimizer, scheduler, world, training_comm, device, config)

        # Update local metrics
        local_train_metrics.update(local_outputs)
        curr_metrics = local_train_metrics.compute()
        logger.log_metrics(curr_metrics, step=step, master=False)

        # Update global metrics
        if world.is_master and stage_outputs is not None:
            global_train_metrics.update(stage_outputs)
            curr_metrics = global_train_metrics.compute()
            logger.log_metrics(curr_metrics, step=step, master=True)
            train_bar.set_description(get_train_pbar_description(global_train_metrics, prefix="[TRAIN]"))
            train_bar.update()

        # Sample
        if config.sample.enable and config.sample.every_n_steps > 0 and step % config.sample.every_n_steps == 0:
            samples = sample_loop(step, inner_model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
            logger.log_samples(samples, step=step)

        # Validate
        if config.eval.enable and config.eval.every_n_steps > 0 and step % config.eval.every_n_steps == 0:
            outputs = eval_loop("eval", inner_model, loss_fn, eval_dataloader, eval_metrics, world, training_comm, device, config)
            logger.log_metrics(outputs, step=step, master=True)

    # Final evaluation
    if config.eval.enable:
        outputs = eval_loop("test", inner_model, loss_fn, eval_dataloader, eval_metrics, world, training_comm, device, config)
        logger.log_metrics(outputs, step=step, master=True)

    # Sample
    if config.sample.enable:
        samples = sample_loop(step, inner_model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
        logger.log_samples(samples, step=step)

def main(config: SwarmConfig):
    global logger

    # Get world
    world = World(config.swarm)

    # Get logger
    logger = get_logger(world, config.logging)
    logger.log_message(f"Starting process {world.local_rank}", master=False, level=Level.INFO)
    logger.log_config(config)
    logger.log_world(world)

    # Seed everything
    seed_everything(config.train.seed)
    logger.log_message(f"Seeded everything with {config.train.seed}", master=True, level=Level.INFO)

    # Set device
    device = get_device(config.device, world.local_rank)
    logger.log_message(f"Using device {device}", master=False, level=Level.INFO)

    # Set precision
    torch.set_float32_matmul_precision(config.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}", master=True, level=Level.INFO)

    # Load inner model
    base_model = get_model(config.model)
    inner_model = get_sharded_model(base_model, world)
    initialize_gradients(inner_model)
    logger.log_message(f"Initialized inner model ({format_int(base_model.num_parameters(), 2)} parameters)", master=True, level=Level.INFO)

    # Get outer model
    outer_model = get_outer_model(inner_model)
    logger.log_message(f"Initialized outer model {format_int(outer_model.num_parameters(), 2)} parameters)", master=False, level=Level.INFO)

    # Load tokenizer
    tokenizer = get_tokenizer()
    logger.log_message(f"Loaded tokenizer ({format_int(len(tokenizer), 0)} vocab size)", master=True, level=Level.INFO)

    # Load dataset (NB: Must be pre-tokenized)
    data = get_dataset(config.data, split="train")
    logger.log_message(f"Loaded dataset {config.data.path} with {format_int(len(data))} examples", master=True, level=Level.INFO)

    # Tokenize (optional)
    if config.data.tokenize:
        seq_length = config.data.seq_length + 1
        data = data.map(lambda x: tokenize(x["text"], tokenizer, seq_length, return_tensors=None), remove_columns=["text"])
        logger.log_message(f"Tokenized dataset {config.data.path} with {format_int(len(data))} examples", master=True, level=Level.INFO)
    
    # Split dataset
    train_val_dict = data.train_test_split(test_size=config.eval.eval_size, shuffle=True, seed=config.train.seed)
    train_data, val_data = train_val_dict["train"], train_val_dict["test"]
    logger.log_message(f"Split dataset {config.data.path} into {format_int(len(train_data))} train, {format_int(len(val_data))} validation examples", master=True, level=Level.INFO)

    # Setup data loaders
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=False, pin_memory=config.data.pin_memory, num_workers=config.data.num_workers)
    eval_dataloader = get_dataloader(val_data, batch_size=config.train.batch_size, shuffle=False, pin_memory=config.data.pin_memory, num_workers=config.data.num_workers)

    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)

    # Get training, evaluation and testing setup
    train_setup = get_train_setup(num_train_steps, config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(train_data))
    eval_setup = get_train_setup(len(eval_dataloader), config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(val_data))
    logger.log_message("Train setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in train_setup.items()]), master=True, level=Level.INFO)
    logger.log_message("Eval setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in eval_setup.items()]), master=True, level=Level.INFO)

    # Get optimizer and scheduler
    inner_optimizer = get_optimizer(inner_model, config.train.inner_optimizer)
    outer_optimizer = None
    if config.train.outer_optimizer.type != "None":
        outer_optimizer = get_optimizer(outer_model, config.train.outer_optimizer)
    scheduler = get_scheduler(inner_optimizer, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()
    logger.log_message("Initialized optimizer, scheduler and loss function", master=True, level=Level.INFO)

    # Training loop
    train_loop(num_train_steps, inner_model, outer_model, tokenizer, train_dataloader, eval_dataloader, loss_fn, inner_optimizer, outer_optimizer, scheduler, world, device, config)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(SwarmConfig(**parse_argv()))
