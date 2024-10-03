"""
Minimal implementation of pre-training a LLM for text generation.
"""
import autorootcwd
from typing import Dict

import torch
from torch.optim import AdamW, Optimizer
from transformers import AutoModelForCausalLM
from datasets import disable_progress_bar
from tqdm import tqdm

from src.utils import seed_everything, get_device, get_model, get_tokenizer, get_dataset, get_dataloader, non_empty_text, non_headline, tokenize
from pydantic import validate_call
from pydantic_config import parse_argv
from src.config import Config

def train(model: AutoModelForCausalLM, optimizer: Optimizer, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()
    optimizer.step()

    return outputs

def eval(model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs

@validate_call
def main(config: Config):
    # Set seed
    seed_everything(config.train.seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = get_model(config.model)
    print(f"Loaded model '{config.model.name}' ({model.num_parameters() / 1e6:.1f}M parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(config.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded tokenizer '{config.tokenizer.name}' ({len(tokenizer) / 1e3:.1f}K tokens)")

    # Load and split dataset
    train_data = get_dataset(config.data, split="train")
    val_data = get_dataset(config.data, split="validation")
    test_data = get_dataset(config.data, split="test")
    print(f"Loaded dataset {config.data.path}/{config.data.name} with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")

    # Prepare dataset
    train_data_tok = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    val_data_tok = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    test_data_tok = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, config.data.seq_length))
    print(f"Tokenized dataset with {len(train_data_tok)} train, {len(val_data_tok)} validation, {len(test_data_tok)} test samples")
    
    # Prepare data loaders
    train_dataloader = get_dataloader(train_data_tok, batch_size=config.train.batch_size, shuffle=True, cycle=False)
    val_dataloader = get_dataloader(val_data_tok, batch_size=config.eval.batch_size, shuffle=True, cycle=False)
    test_dataloader = get_dataloader(test_data_tok, batch_size=config.eval.batch_size, shuffle=False, cycle=False)
    print(f"Prepared dataloaders with {len(train_dataloader)} train, {len(val_dataloader)} validation, {len(test_dataloader)} test batches")

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay, betas=config.train.adam_betas)

    model.to(device)
    model.train()
    train_loss = 0
    train_bar = tqdm(range(1, min(len(train_dataloader), config.train.max_steps)+1), position=0, leave=True)
    for train_step in train_bar:
        optimizer.zero_grad()

        batch = next(train_dataloader)
        outputs = train(model, optimizer, batch, device)

        train_loss += outputs.loss  
        running_train_loss = train_loss / train_step
        running_train_perplexity = torch.exp(running_train_loss)

        train_bar.set_description(f"[TRAIN] Step: {train_step}/{config.train.max_steps}, Loss: {running_train_loss.item():.4f}, Perplexity: {running_train_perplexity.item():.2f}")

        if train_step % config.eval.every_n_steps == 0 and config.eval.enable:
            model.eval()
            eval_loss = 0
            eval_bar = tqdm(range(1, min(len(val_dataloader), config.eval.max_steps)+1), position=1, leave=False)
            for eval_step in eval_bar:
                batch = next(val_dataloader)
                outputs = eval(model, batch, device)

                eval_loss += outputs.loss
                running_eval_loss = eval_loss / eval_step
                running_eval_perplexity = torch.exp(running_eval_loss)

                eval_bar.set_description(f"[EVAL] Step: {eval_step}/{config.eval.max_steps}, Loss: {running_eval_loss.item():.4f}, Perplexity: {running_eval_perplexity.item():.2f}")
            
            model.train()


    print("Training completed!")

if __name__ == "__main__":
    disable_progress_bar()
    main(Config(**parse_argv()))
