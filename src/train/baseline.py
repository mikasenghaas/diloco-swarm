"""
Single-GPU LLM pre-training.
"""
import autorootcwd
import time
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from datasets import disable_progress_bar
from tqdm import tqdm

from src.logger import Level
from src.utils import set_precision, seed_everything, get_device, get_logger, get_model, get_tokenizer, get_dataset, get_dataloader, get_micro_dataloader, get_optimizer, get_scheduler, non_empty_text, non_headline, tokenize, get_train_pbar_description, get_eval_pbar_description, get_num_steps, get_train_setup, format_int, format_float, get_autocast_context
from src.metrics import Outputs, Step, Examples, Tokens, Norm, Loss, Perplexity, Throughput, LearningRate, Metrics
from src.config import ModelConfig, TokenizerConfig, DataConfig, TrainConfig, EvalConfig, LoggingConfig
from pydantic_config import BaseConfig, parse_argv

class BaselineConfig(BaseConfig):
    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    logging: LoggingConfig = LoggingConfig()

def train(grad_accumulation_steps: int, step: int, model: AutoModelForCausalLM, batch_loader: DataLoader, optimizer: AdamW, scheduler: LambdaLR, device: torch.device) -> Outputs:
    start = time.time()
    model.train()
    model.to(device)
    optimizer.zero_grad()
    batch_loss = torch.Tensor([0.0]).to(device)
    batch_tokens, batch_examples = 0, 0

    autocast = get_autocast_context(device)
    for micro_batch in batch_loader:
        micro_batch = {k: v.to(device) for k, v in micro_batch.items()}
        with autocast:
            outputs = model(micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"], labels=micro_batch["labels"])
        outputs.loss /= grad_accumulation_steps
        outputs.loss.backward()

        batch_loss += outputs.loss.detach()
        batch_examples += micro_batch["input_ids"].shape[0]
        batch_tokens += micro_batch["input_ids"].shape[0] * micro_batch["input_ids"].shape[1]

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    return Outputs(step=step, lr=scheduler.get_last_lr()[0], loss=batch_loss, num_tokens=batch_tokens, num_examples=batch_examples, norm=norm, time=time.time() - start)

def eval(step: int, model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Outputs:
    start = time.time()
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        num_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
        num_examples = batch["input_ids"].shape[0]

        return Outputs(step=step, loss=outputs.loss, num_tokens=num_tokens, num_examples=num_examples, time=time.time() - start)

def main(config: BaselineConfig):
    # Set precision and seed
    set_precision(config.train.precision)
    seed_everything(config.train.seed)

    # Get logger
    logger = get_logger(config.logging)
    logger.log_config(config)

    # Get device
    device = get_device()
    logger.log_message(f"Using device: {device}")

    # Load model
    model = get_model(config.model)
    logger.log_message(f"Loaded model '{config.model.name}' ({format_int(model.num_parameters(), 2)} parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    logger.log_message(f"Loaded tokenizer '{config.tokenizer.name}' ({format_int(len(tokenizer), 0)} tokens)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    logger.log_message(f"Loaded dataset {config.data.path}/{config.data.name} with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")

    # Prepare dataset
    seq_length = config.data.seq_length
    train_data = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, seq_length))
    val_data = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, seq_length))
    test_data = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, seq_length))
    logger.log_message(f"Tokenized dataset with {format_int(len(train_data))} train, {format_int(len(val_data))} validation, {format_int(len(test_data))} test examples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data, batch_size=config.train.batch_size, shuffle=True, cycle=True)
    val_dataloader = get_dataloader(val_data, batch_size=config.eval.batch_size, shuffle=True, cycle=True)
    test_dataloader = get_dataloader(test_data, batch_size=config.eval.batch_size, shuffle=False, cycle=True)
    
    # Compute number of training steps
    num_train_steps = get_num_steps(config.train.max_steps, config.train.max_epochs, len(train_data), config.train.batch_size)
    num_eval_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(val_data), config.eval.batch_size)
    num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.eval.batch_size)

    # Get training, evaluation and testing setup
    train_setup = get_train_setup(num_train_steps, config.train.batch_size, config.data.seq_length, config.train.micro_batch_size, len(train_data))
    eval_setup = get_train_setup(num_eval_steps, config.eval.batch_size, config.data.seq_length, -1, len(val_data))
    test_setup = get_train_setup(num_test_steps, config.eval.batch_size, config.data.seq_length, -1, len(test_data))

    logger.log_message("Train setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in train_setup.items()]))
    logger.log_message("Eval setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in eval_setup.items()]))
    logger.log_message("Test setup:\t" + "\t".join([f"{k.replace('_', ' ').title()}: {format_int(v, 1) if isinstance(v, int) else format_float(v)}" for k, v in test_setup.items()]))

    # Compute grad accumulation steps
    grad_accumulation_steps = config.train.batch_size // config.train.micro_batch_size
    assert config.train.batch_size % grad_accumulation_steps == 0, "Batch size must be divisible by grad accumulation steps"
    logger.log_message(f"Gradient Accumulation Steps: {grad_accumulation_steps}")

    # Set up optimizer
    optimizer = get_optimizer(config.train, model)
    scheduler = get_scheduler(config.train, optimizer, num_train_steps)

    # Start Training
    logger.log_message("Starting training")
    train_metrics = Metrics([Step(), Examples(), Tokens(), Norm(), Loss(), Perplexity(), Throughput(), LearningRate()], name="train")
    eval_metrics = Metrics([Loss(), Perplexity()], name="eval")
    train_bar = tqdm(range(1, num_train_steps+1), position=0, leave=True)
    for train_step in train_bar:
        # Train step
        batch = next(train_dataloader)
        micro_batchloader = get_micro_dataloader(batch, config.train.micro_batch_size)
        outputs = train(grad_accumulation_steps, train_step, model, micro_batchloader, optimizer, scheduler, device)
        
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

        # Checkpoint
        if config.logging.ckpt.enable and config.logging.ckpt.every_n_steps > 0 and train_step % config.logging.ckpt.every_n_steps == 0:
            logger.log_checkpoint(model, tokenizer, train_step)

    # Test
    if config.eval.enable:
        test_metrics = Metrics([Loss(), Perplexity()], name="test")
        num_test_steps = get_num_steps(config.eval.max_steps, config.eval.max_epochs, len(test_data), config.eval.batch_size)
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