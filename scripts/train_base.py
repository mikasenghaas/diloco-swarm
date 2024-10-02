"""
Minimal implementation of pre-training a fresh LLM for text generation.
"""
import autorootcwd

from typing import Dict
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from torch.optim import AdamW, Optimizer
from tqdm import tqdm

from src.utils import get_device, get_model, get_tokenizer, get_dataset, get_subset, non_empty_text, non_headline, tokenize, collate_fn
from src.args import TrainingArgs

def train(model: AutoModelForCausalLM, optimizer: Optimizer, train_dataloader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.train()
    total_loss = 0
    pbar = tqdm(train_dataloader)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss
        running_loss = total_loss / (batch_idx + 1)
        running_perplexity = torch.exp(running_loss)

        pbar.set_description(f"Loss: {running_loss.item():.4f}, Perplexity: {running_perplexity.item():.4f}")

        loss.backward()
        optimizer.step()

    return {
        "loss": running_loss,
        "perplexity": running_perplexity
    }

def eval(model: AutoModelForCausalLM, val_dataloader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    total_loss = 0
    pbar = tqdm(val_dataloader)
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss
        running_loss = total_loss / (batch_idx + 1)
        running_perplexity = torch.exp(running_loss)

        # compute accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        total_correct += correct.sum().item()
        running_accuracy = total_correct / len(val_dataloader.dataset)

        pbar.set_description(f"Loss: {running_loss:.4f}, Perplexity: {running_perplexity:.4f}, Accuracy: {running_accuracy:.4f}")

    return {
        "loss": running_loss,
        "perplexity": running_perplexity,
        "accuracy": running_accuracy
    }

def main():
    # Get arguments
    training_args = TrainingArgs.from_cli()
    pprint(training_args)
    print("-"*100)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = get_model(training_args.model_name)
    print(f"Loaded model {training_args.model_name} ({model.num_parameters() / 1e6:.2f}M parameters)")

    # Load tokenizer
    tokenizer = get_tokenizer(training_args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded tokenizer {training_args.tokenizer_name} with {len(tokenizer) / 1e3:.2f}K tokens")

    # Load and split dataset
    data = get_dataset(training_args.data_path, training_args.data_name)
    train_data, val_data, test_data = data["train"], data["validation"], data["test"]
    print(f"Loaded dataset {training_args.data_path}/{training_args.data_name} with {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")
    print("-"*100)

    # Subset the train dataset
    train_data = get_subset(train_data, training_args.subset_size)
    print(f"Subsetted train dataset to {len(train_data)} samples ({training_args.subset_size * 100:.2f}% of original)")

    # Prepare dataset
    train_data_tok = train_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, training_args.seq_len))
    val_data_tok = val_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, training_args.seq_len))
    test_data_tok = test_data.filter(non_empty_text).filter(non_headline).map(lambda examples: tokenize(examples, tokenizer, training_args.seq_len))
    print(f"Tokenized dataset with {len(train_data_tok)} train, {len(val_data_tok)} validation, {len(test_data_tok)} test samples")
    
    # Prepare data loaders
    train_dataloader = DataLoader(train_data_tok, batch_size=training_args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data_tok, batch_size=training_args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data_tok, batch_size=training_args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"DataLoader with {len(train_dataloader)} train, {len(val_dataloader)} validation, {len(test_dataloader)} test samples")

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

    print("-"*100)
    print("Starting training!")
    model.to(device)
    for _ in range(training_args.num_epochs):
        train_metrics = train(model, optimizer, train_dataloader, device)
        print(f"Train metrics: {train_metrics}")
        eval_metrics = eval(model, val_dataloader, device)
        print(f"Validation metrics: {eval_metrics}")

    print("Training completed!")
    print("-"*100)

if __name__ == "__main__":
    main()

