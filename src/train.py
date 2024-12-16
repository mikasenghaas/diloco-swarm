"""
Simple SWARM Parallel LLM Pre-Training.

Single GPU:  torchrun --nproc_per_node 1 src/train/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --world.num_stages 1
DP: torchrun --nproc_per_node 2 src/train/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --world.num_stages 1
PP: torchrun --nproc_per_node 2 src/train/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --world.num_stages 2
"""
import autorootcwd

import time
from tqdm import tqdm
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from pydantic_config import BaseConfig, parse_argv
from transformers import AutoTokenizer

from src.logger import Level
from src.world import World
from src.serializer import Serializer
from src.comm import Comm
from src.sampler import BatchData, BatchSampler
from src.config import SwarmConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig
from src.utils import seed_everything, get_world, get_logger, get_device, get_model, get_sharded_model, get_tokenizer, get_dataset, tokenize, get_dataloader, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description, filter_logits_targets
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate

TIMEOUT = 0.001
LOGGER = None

class SwarmConfig(BaseConfig):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    swarm: SwarmConfig = SwarmConfig()
    eval: EvalConfig = EvalConfig()
    sample: SampleConfig = SampleConfig()
    logging: LoggingConfig = LoggingConfig()

def train(train_step: int, model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, world: World, comm: Comm, device: torch.device, swarm_config: SwarmConfig, train_config: TrainConfig) -> Outputs:
    """Training step on train batch"""
    # Prepare model
    start = time.time()
    model.to(device)
    model.train()
    optimizer.zero_grad()

    # Setup step
    num_micro_steps = train_config.batch_size // train_config.micro_batch_size
    world.setup_step(train_step, num_micro_steps=num_micro_steps)

    # Prepare batch batch
    batch_data = BatchData(batch)
    micro_batches = {} # (rank, local_micro_step) -> micro_batch
    for rank in world.first_stage_ranks:
        micro_sampler = BatchSampler(batch_data, rank=rank, ranks=world.first_stage_ranks, micro_batch_size=train_config.micro_batch_size)
        micro_dataloader = DataLoader(batch_data, batch_size=train_config.micro_batch_size, shuffle=False, sampler=micro_sampler)
        for local_micro_step, micro_batch in enumerate(micro_dataloader, start=1):
            if world.is_last_stage: micro_batches[(rank, local_micro_step)] = micro_batch
            if world.is_first_stage and rank == world.rank: comm.load_forward_queue(local_micro_step, micro_batch["input_ids"])
    
    # Prepare statistics
    batch_loss, batch_tokens = 0.0, 0
    input_output_tensors = {}

    # Zero gradients
    optimizer.zero_grad()
    LOGGER.log_message(f"Starting train step {train_step} (batch_size: {train_config.batch_size}, micro_batch_size: {train_config.micro_batch_size}, num_micro_steps: {num_micro_steps})", Level.DEBUG)
    while True:
        if world.step_done(train_step): break
        if comm.can_receive_forward():
            src, input_tensor, (root, local_micro_step) = comm.recv_forward()
            LOGGER.log_message(f"Forward tensor of shape={input_tensor.shape} and dtype={input_tensor.dtype}", Level.DEBUG)
            output_tensor = model(input_tensor.to(device))
            LOGGER.log_message(f"Computed output tensor of shape={output_tensor.shape} and dtype={output_tensor.dtype}", Level.DEBUG)
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
    if swarm_config.sync_every_n_steps > 0 and train_step % swarm_config.sync_every_n_steps == 0:
        comm.sync_gradients()
    
    # Optimizer steps
    LOGGER.log_message(f"Step optimizer and scheduler", Level.DEBUG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_norm)
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
    """Evaluation step on eval batch"""
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
    for rank in world.first_stage_ranks:
        micro_sampler = BatchSampler(batch_data, rank, world.first_stage_ranks, micro_batch_size)
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
    """Evaluation loop on eval data loader"""
    eval_range = range(1, eval_config.max_steps + 1)
    eval_bar = tqdm(eval_range, position=1, leave=False if eval_type == "eval" else True) if world.is_leader and world.is_last_stage else None
    eval_metrics.reset()
    for eval_step in eval_range:
        batch = next(eval_dataloader)
        _, stage_outputs = eval(eval_step, eval_type, model, batch, loss_fn, device, world, comm)
        eval_metrics.update(stage_outputs)
        
        if not (world.is_leader and world.is_last_stage): continue
        eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]" if eval_type == "eval" else "[TEST]"))
    
    final_metrics = eval_metrics.compute()
    return final_metrics

def sample_loop(model: nn.Module, tokenizer: AutoTokenizer, world: World, comm: Comm, prompt_length: int, config: SampleConfig, device: torch.device) -> List[str]:
    """Sample loop for generating tokens"""
    model.to(device); model.eval()

    # Prepare input and generate output
    tensor_length = prompt_length + config.max_new_tokens
    stop_input_ids = -torch.ones((config.num_samples, tensor_length), device=device, dtype=torch.long)
    stop_activations = -torch.ones((config.num_samples, tensor_length, model.config.n_embd), device=device, dtype=torch.float32)
    input_ids = tokenize(config.prompt, tokenizer, max_length=tensor_length)["input_ids"].to(device).repeat(config.num_samples, 1)
    comm.load_input_ids_queue(input_ids)
    for sample_step, generated_id in enumerate(range(prompt_length, tensor_length)):
        LOGGER.log_message(f"Step {sample_step}: Sampling token {generated_id}", Level.DEBUG)
        if world.is_first_stage:
            if not world.is_last_stage:
                input_ids = comm.recv_input_ids()
                if (input_ids == stop_input_ids).all(): comm.send_activations(stop_activations); break
            output_tensor = model(input_ids)
        else:
            hidden_states = comm.recv_activations()
            if (hidden_states == stop_activations).all(): break
            output_tensor = model(hidden_states)
        comm.send_activations(output_tensor)

        if world.is_last_stage:
            logits = output_tensor[:, generated_id-1, :] / config.temperature # (B, V)
            if config.top_k is not None:
                v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # (B, V)
            probs = F.softmax(logits, dim=-1) # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1),
            input_ids[:, generated_id] = idx_next.squeeze()
            if (idx_next == tokenizer.eos_token_id).all():
                comm.send_input_ids(stop_input_ids)
                if world.is_first_stage: break
            if generated_id < tensor_length - 1: comm.send_input_ids(input_ids)

    return [tokenizer.decode(generated_ids, skip_special_tokens=True) for generated_ids in input_ids] if world.is_last_stage and world.is_leader else []

def main(config: SwarmConfig):
    global LOGGER

    # Get world
    world = World(config.swarm)

    # Get logger
    logger_name = f"{world.local_rank}" if world.world_size > 1 else "master"
    LOGGER = get_logger(config.logging, logger_name, world.run_id)
    LOGGER.log_message(f"Starting process {world.local_rank}", Level.INFO)
    LOGGER.log_config(config)
    LOGGER.log_world(world)

    # Seed everything
    seed_everything(config.train.seed)
    LOGGER.log_message(f"Seeded everything with {config.train.seed}", Level.INFO)

    # Set device
    device = get_device(world.local_rank)
    torch.cuda.set_device(world.local_rank)
    LOGGER.log_message(f"Using device {device}", Level.INFO)

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

    # Initialize training communication
    activations_shape, input_ids_shape = None, None
    if config.sample.enable:
        prompt_length = tokenize(config.sample.prompt, tokenizer)["input_ids"].shape[1]
        activations_shape = (config.sample.num_samples, prompt_length+config.sample.max_new_tokens, model.config.n_embd)
        input_ids_shape = (config.sample.num_samples, prompt_length+config.sample.max_new_tokens)

    serializer = Serializer((config.train.micro_batch_size, config.data.seq_length, model.config.n_embd))
    comm = Comm(world, serializer.shape, activations_shape, input_ids_shape, torch.float32, model, serializer, device, LOGGER, TIMEOUT)

    # Sample before training
    if config.sample.enable:
        LOGGER.log_message(f"Sampling before training")
        samples = sample_loop(model, tokenizer, world, comm, prompt_length, config.sample, device)
        LOGGER.log_samples(samples, step=0)

    # Validate before training
    if config.eval.enable:
        outputs = eval_loop("eval", model, loss_fn, eval_dataloader, eval_metrics, world, comm, device, config.eval)
        if world.is_leader and world.is_last_stage:
            LOGGER.log_metrics(outputs, level=Level.DEBUG, step=0)

    # Training loop
    train_range = range(1, num_train_steps + 1)
    train_bar = tqdm(train_range, position=0, leave=True) if world.is_leader and world.is_last_stage else None
    for train_step in train_range:
        batch = next(train_dataloader)
        _, stage_outputs = train(train_step, model, batch, loss_fn, optimizer, scheduler, world, comm, device, config.swarm, config.train)

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
        if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
            samples = sample_loop(model, tokenizer, world, comm, prompt_length, config.sample, device)
            LOGGER.log_samples(samples, step=train_step)

        # TODO: Implement checkpointing
        # if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
        #     ckpt_dir = ckpt.save(train_step, model)
        #     logger.log_message(f"Saved model checkpoint at {ckpt_dir}")

    # Sample
    if config.sample.enable:
        samples = sample_loop(model, tokenizer, world, comm, prompt_length, config.sample, device)
        LOGGER.log_samples(samples, step=train_step)

    # Test
    if config.eval.enable:
        outputs = eval_loop("test", model, loss_fn, test_dataloader, test_metrics, world, comm, device, config.eval)
        if world.is_leader and world.is_last_stage:
            LOGGER.log_metrics(outputs, level=Level.DEBUG, step=train_step)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    from datasets import disable_progress_bar; disable_progress_bar()
    main(SwarmConfig(**parse_argv()))
