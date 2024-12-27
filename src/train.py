"""
Simple SWARM Parallel LLM Pre-Training.

Single GPU:  torchrun --nproc_per_node 1 src/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --swarm.num_stages 1
DP: torchrun --nproc_per_node 2 src/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --swarm.num_stages 1
PP: torchrun --nproc_per_node 2 src/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --swarm.num_stages 2
SWARM: torchrun --nproc_per_node 4 src/train.py @configs/debug.toml --model @configs/model/gpt2-tiny.toml --data @configs/data/wikitext.toml --swarm.num_stages 2
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
from src.comm import TrainingComm, InferenceComm
from src.sampler import BatchData, BatchSampler
from src.config import SwarmConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig, SampleConfig, LoggingConfig, AmpConfig
from src.utils import seed_everything, get_logger, get_device, get_dtype, get_model, get_sharded_model, initialize_gradients, get_tokenizer, get_dataset, tokenize, get_dataloader, get_optimizer, get_scheduler, get_num_steps, get_train_setup, format_int, format_float, get_train_pbar_description, get_eval_pbar_description, filter_logits_targets, nullcontext
from src.metrics import Outputs, Metrics, Step, Time, MicroTime, Tokens, Loss, Perplexity, Throughput, Norm, LearningRate

logger = None

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

def train(train_step: int, num_train_steps: int, model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, optimizer: AdamW, scheduler: LambdaLR, world: World, training_comm: TrainingComm, device: torch.device, swarm_config: SwarmConfig, train_config: TrainConfig, amp_config: AmpConfig) -> Outputs:
    """Training step on train batch"""
    # Prepare model
    start = time.time()
    model.to(device); model.train()
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
            micro_batches[(rank, local_micro_step)] = micro_batch
            if world.is_first_stage and rank == world.rank: training_comm.load_forward(tensor=micro_batch["input_ids"], metadata=(rank, local_micro_step))

    # Prepare statistics
    batch_loss, batch_tokens = 0.0, 0
    input_output_tensors = {}

    # Zero gradients
    optimizer.zero_grad()
    logger.log_message(f"Train step {train_step}", master=False, level=Level.DEBUG)
    while True:
        if world.is_step_done(train_step): break
        if time.time() - start > train_config.step_timeout: logger.log_message(f"Train step {train_step} timed out", master=False, level=Level.WARNING); break
        if not (training_comm.forward_recv_thread.can_receive or training_comm.backward_recv_thread.can_receive): time.sleep(1e-4); continue
        if training_comm.forward_recv_thread.can_receive and len(input_output_tensors) < train_config.max_micro_batches:
            # Receive input tensor
            src, input_tensor, (root, local_micro_step) = training_comm.recv_forward()
            micro_batch = micro_batches[(root, local_micro_step)]
            micro_batch["hidden_states"] = input_tensor
            
            # Forward pass
            with torch.amp.autocast(device_type=device.type, dtype=get_dtype(amp_config.dtype)) if amp_config.enable else nullcontext():
                output_tensor = model.forward(micro_batch, device)

            # Remember tensors for backward on all but last stages
            if world.num_stages > 1 and not world.is_last_stage: # Maybe don't optimize non-pipeline for comparability
                logger.log_message(f"Offloading input and output tensors to CPU", master=False, level=Level.DEBUG)
                input_output_tensors[(root, local_micro_step)] = (src, input_tensor, output_tensor)

            # Send output tensor
            training_comm.send_forward(tensor=output_tensor, metadata=(root, local_micro_step))

            if world.is_last_stage:
                # Get targets and attention mask for current micro batch
                micro_batch = micro_batches[(root, local_micro_step)]

                # Filter logits and targets for loss calculation
                mask, target_ids = micro_batch["attention_mask"].to(device), micro_batch["target_ids"].to(device)
                logits_filtered, targets_filtered = filter_logits_targets(output_tensor, target_ids, mask)
                
                # Compute loss
                loss = loss_fn(logits_filtered, targets_filtered)
                loss = loss / num_micro_steps
                
                # Update statistics
                batch_loss += loss.detach().item()
                batch_tokens += input_tensor.shape[0] * input_tensor.shape[1]
                
                # Backward pass
                input_tensor_grad = model.backward(input_tensor if not world.is_first_stage else None, loss, None)
                training_comm.send_backward(dst=src, tensor=input_tensor_grad, metadata=(root, local_micro_step))

                if world.is_first_stage: world.micro_step_done()

        elif training_comm.backward_recv_thread.can_receive:
            _, output_tensor_grad, (root, local_micro_step) = training_comm.recv_backward()
            src, input_tensor, output_tensor = input_output_tensors.pop((root, local_micro_step))
            input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
            input_tensor_grad = model.backward(input_tensor if not world.is_first_stage else None, output_tensor, output_tensor_grad)
            training_comm.send_backward(dst=src, tensor=input_tensor_grad, metadata=(root, local_micro_step))
        
            if world.is_first_stage: world.micro_step_done()
        
    # Synchronize
    dist.barrier()
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # Sync gradients across ranks in same stage
    if (swarm_config.sync_every_n_steps > 0 and train_step % swarm_config.sync_every_n_steps == 0) or train_step == num_train_steps:
        logger.log_message(f"Syncing gradients", master=False, level=Level.DEBUG)
        training_comm.sync_gradients(model)
    
    # Optimizer steps
    logger.log_message(f"Step optimizer and scheduler", master=False, level=Level.DEBUG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_norm)
    lr = scheduler.get_last_lr()[0]
    optimizer.step()
    scheduler.step()

    # Empty cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Timing
    step_time = time.time() - start
    micro_step_time = step_time / num_micro_steps

    # Compute and sync outputs
    local_outputs = Outputs(step=train_step, lr=lr, loss=batch_loss, tokens=batch_tokens, norm=norm, time=step_time, micro_step_time=micro_step_time)
    stage_outputs = training_comm.sync_outputs(local_outputs)

    return local_outputs, stage_outputs

def eval(eval_step: int, eval_type: Literal["eval", "test"], model: nn.Module, batch: Dict[str, torch.Tensor], loss_fn: nn.Module, device: torch.device, world: World, training_comm: TrainingComm, amp_config: AmpConfig) -> Outputs:
    """Evaluation step on eval batch"""
    # Prepare model
    start = time.time()
    model.to(device); model.eval()

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
            micro_batches[(rank, local_micro_step)] = micro_batch
            if world.is_first_stage and rank == world.rank: training_comm.load_forward(tensor=micro_batch["input_ids"], metadata=(rank, local_micro_step))
    
    # Prepare statistics
    batch_loss, batch_tokens = 0.0, 0
    logger.log_message(f"{eval_type.capitalize()} step {eval_step}", master=False, level=Level.DEBUG)
    while not world.is_step_done(eval_step, type=eval_type):
        if training_comm.forward_recv_thread.can_receive:
            _, input_tensor, (root, local_micro_step) = training_comm.recv_forward()
            micro_batch = micro_batches[(root, local_micro_step)]
            micro_batch["hidden_states"] = input_tensor
            with torch.amp.autocast(device_type=device.type, dtype=get_dtype(amp_config.dtype)):
                output_tensor = model.forward(micro_batch, device)
            training_comm.send_forward(tensor=output_tensor, metadata=(eval_step, root, local_micro_step))

            if world.is_last_stage:
                # Filter logits and targets for loss calculation
                mask, target_ids = micro_batch["attention_mask"], micro_batch["target_ids"]
                logits_filtered, targets_filtered = filter_logits_targets(output_tensor, target_ids, mask)

                # Compute loss
                loss = loss_fn(logits_filtered, targets_filtered)
                loss = loss / num_micro_steps
                
                # Update statistics
                batch_loss += loss.detach().item()
                batch_tokens += input_tensor.shape[0] * input_tensor.shape[1]

                world.micro_step_done(eval_type)

    # Timing
    dist.barrier()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    step_time = time.time() - start

    outputs = Outputs(step=eval_step, time=step_time, loss=batch_loss, tokens=batch_tokens)
    stage_outputs = training_comm.sync_outputs(outputs)

    return outputs, stage_outputs

@torch.no_grad()
def eval_loop(eval_type: Literal["eval", "test"], num_eval_steps: int, model: nn.Module, loss_fn: nn.Module, eval_dataloader: DataLoader, eval_metrics: Metrics, world: World, training_comm: TrainingComm, device: torch.device, train_config: TrainConfig) -> Outputs:
    """Evaluation loop on eval data loader"""
    eval_range = range(1, num_eval_steps + 1)
    eval_bar = tqdm(eval_range, position=1, leave=False) if world.is_leader and world.is_last_stage else None
    eval_metrics.reset()
    for eval_step in eval_range:
        batch = next(eval_dataloader)
        _, stage_outputs = eval(eval_step, eval_type, model, batch, loss_fn, device, world, training_comm, train_config)
        
        if not (world.is_leader and world.is_last_stage): continue
        eval_metrics.update(stage_outputs)
        eval_bar.set_description(get_eval_pbar_description(eval_metrics, prefix="[EVAL]" if eval_type == "eval" else "[TEST]"))
        eval_bar.update()
    
    dist.barrier()

    if world.is_leader and world.is_last_stage: eval_metrics = eval_metrics.compute()
    return eval_metrics

@torch.no_grad()
def sample_loop(train_step: int, model: nn.Module, tokenizer: AutoTokenizer, world: World, inference_comm: InferenceComm, prompt_length: int, config: SampleConfig, device: torch.device) -> List[str]:
    """Sample loop for generating tokens"""
    model.to(device); model.eval()

    # Prepare input
    tensor_length = prompt_length + config.max_new_tokens
    input_ids = tokenize(config.prompt, tokenizer, max_length=tensor_length)["input_ids"].to(device).repeat(config.num_samples, 1)
    micro_batch = {"input_ids": input_ids, "attention_mask": None, "hidden_states": None}
    if world.is_leader and world.is_first_stage: inference_comm.load(input_ids)

    generated_id = prompt_length; world.setup_step(train_step, num_micro_steps=config.max_new_tokens, type="sample")
    while world.is_leader and not world.is_step_done(train_step, type="sample"):
        if not inference_comm.recv_thread.can_receive: time.sleep(1e-4); continue
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
    logger.log_message(f"Seeded everything with {config.train.seed}", master=False, level=Level.INFO)

    # Set device
    device = get_device(config.device, world.local_rank)
    logger.log_message(f"Using device {device}", master=False, level=Level.INFO)

    # Set precision
    torch.set_float32_matmul_precision(config.amp.precision)
    logger.log_message(f"Using precision: {torch.get_float32_matmul_precision()}", master=True, level=Level.INFO)

    # Load model and create sharded version
    model = get_model(config.model)
    original_num_params = model.num_parameters()
    logger.log_message(f"Loaded model ({format_int(model.num_parameters(), 2)} parameters)", master=True, level=Level.INFO)
    
    model = get_sharded_model(model, world)
    if model.num_parameters() < original_num_params:
        logger.log_message(f"Sharded model ({format_int(model.num_parameters(), 2)} parameters)", master=True, level=Level.INFO)

    # Initialize gradients (prevents sync issues when first evaluating)
    initialize_gradients(model)

    # Load tokenizer
    tokenizer = get_tokenizer()
    logger.log_message(f"Loaded tokenizer ({format_int(len(tokenizer), 0)} vocab size)", master=True, level=Level.INFO)

    # Load dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples", master=True, level=Level.INFO)

    # Prepare dataset
    seq_length = config.data.seq_length + 1
    train_data = train_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    val_data = val_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    test_data = test_data.map(lambda examples: tokenize(examples["text"], tokenizer, max_length=seq_length, return_tensors=None), batched=True)
    logger.log_message(f"Tokenized dataset with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples", master=True, level=Level.INFO)

    # Setup training
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=False, cycle=True)
    eval_dataloader = get_dataloader(val_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=True)
    test_dataloader = get_dataloader(test_data, batch_size=config.train.micro_batch_size, shuffle=False, cycle=True)

    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.train.micro_batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.train.micro_batch_size)

    # Get training, evaluation and testing setup
    train_setup = get_train_setup(num_train_steps, config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(train_data))
    eval_setup = get_train_setup(num_eval_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(val_data))
    test_setup = get_train_setup(num_test_steps, config.train.micro_batch_size, config.data.seq_length, -1, len(test_data))
    logger.log_message("Train setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in train_setup.items()]), master=True, level=Level.INFO)
    logger.log_message("Eval setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in eval_setup.items()]), master=True, level=Level.INFO)
    logger.log_message("Test setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in test_setup.items()]), master=True, level=Level.INFO)

    # Get optimizer and scheduler
    optimizer = get_optimizer(model, config.train.optimizer)
    scheduler = get_scheduler(optimizer, num_train_steps, config.train.scheduler)
    loss_fn = nn.CrossEntropyLoss()
    logger.log_message("Initialized optimizer, scheduler and loss function", master=True, level=Level.INFO)

    # Initialize metrics
    local_train_metrics = Metrics([Step(), Time(), MicroTime(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train/local")
    global_train_metrics = Metrics([Step(), Time(), MicroTime(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train/global")
    eval_metrics = Metrics([Throughput(), Loss(), Perplexity()], name="eval")
    test_metrics = Metrics([Throughput(), Loss(), Perplexity()], name="test")

    # Initialize training communication
    training_shape = (config.train.micro_batch_size, config.data.seq_length, model.config.n_embd)
    training_comm = TrainingComm(world, training_shape, logger)

    # Initialize inference communication
    if config.sample.enable:
        prompt_length = tokenize(config.sample.prompt, tokenizer)["input_ids"].shape[1]
        inference_shape = (config.sample.num_samples, prompt_length+config.sample.max_new_tokens, model.config.n_embd)
        inference_comm = InferenceComm(world, inference_shape)

    # Sample before training
    if config.sample.enable:
        samples = sample_loop(0, model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
        logger.log_samples(samples, step=0)

    # Validate before training
    if config.eval.enable:
        outputs = eval_loop("eval", num_eval_steps, model, loss_fn, eval_dataloader, eval_metrics, world, training_comm, device, config.amp)
        logger.log_metrics(outputs, step=0, master=True)

    # Training loop
    logger.log_message(f"Starting training", master=False, level=Level.DEBUG)
    train_range = range(1, num_train_steps + 1)
    train_bar = tqdm(train_range, position=0, leave=False) if world.is_leader and world.is_last_stage else None
    for train_step in train_range:
        batch = next(train_dataloader)
        local_outputs, global_outputs = train(train_step, num_train_steps, model, batch, loss_fn, optimizer, scheduler, world, training_comm, device, config.swarm, config.train, config.amp)

        # Update local metrics
        if world.is_last_stage:
            local_train_metrics.update(local_outputs)
            curr_metrics = local_train_metrics.compute()
            logger.log_metrics(curr_metrics, step=train_step, master=False)

        # Update global metrics
        if world.is_master and global_outputs is not None:
            global_train_metrics.update(global_outputs)
            curr_metrics = global_train_metrics.compute()
            logger.log_metrics(curr_metrics, step=train_step, master=True)
            train_bar.set_description(get_train_pbar_description(global_train_metrics, prefix="[TRAIN]"))
            train_bar.update()

        # Sample
        if config.sample.enable and config.sample.every_n_steps > 0 and train_step % config.sample.every_n_steps == 0:
            samples = sample_loop(train_step, model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
            logger.log_samples(samples, step=train_step)

        # Validate
        if config.eval.enable and config.eval.every_n_steps > 0 and train_step % config.eval.every_n_steps == 0:
            outputs = eval_loop("eval", num_eval_steps, model, loss_fn, eval_dataloader, eval_metrics, world, training_comm, device, config.amp)
            logger.log_metrics(outputs, step=train_step, master=True)

    # Test
    if config.eval.enable:
        outputs = eval_loop("test", num_test_steps, model, loss_fn, test_dataloader, test_metrics, world, training_comm, device, config.amp)
        logger.log_metrics(outputs, step=train_step, master=True)

    # Sample
    if config.sample.enable:
        samples = sample_loop(train_step, model, tokenizer, world, inference_comm, prompt_length, config.sample, device)
        logger.log_samples(samples, step=train_step)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main(SwarmConfig(**parse_argv()))
