"""
How to use Python dataclasses to define arguments for a program.
"""
from dataclasses import dataclass, field
from argparse import ArgumentParser

@dataclass
class TrainingArgs:
    model_name: str = field(default="PrimeIntellect/llama-14m-fresh", metadata={"help": "HuggingFace model name"})
    tokenizer_name: str = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "HuggingFace tokenizer name"})
    seq_len: int = field(default=128, metadata={"help": "Sequence length"})
    data_path: str = field(default="Salesforce/wikitext", metadata={"help": "HuggingFace dataset path"})
    data_name: str = field(default="wikitext-2-raw-v1", metadata={"help": "HuggingFace dataset name"})
    subset_size: float = field(default=0.1, metadata={"help": "Fraction of the dataset to use"})
    num_epochs: int = field(default=3, metadata={"help": "Number of epochs"})
    learning_rate: float = field(default=4e-4, metadata={"help": "Learning rate"})
    batch_size: int = field(default=64, metadata={"help": "Batch size"})

    @classmethod
    def from_cli(cls) -> 'TrainingArgs':
        parser = ArgumentParser()
        for field in cls.__dataclass_fields__.values():
            parser.add_argument(f"--{field.name}", type=field.type, default=field.default, help=f"{field.metadata.get('help', '')} (Default: '{field.default}')")
        return TrainingArgs(**vars(parser.parse_args()))

if __name__ == "__main__":
    training_args = TrainingArgs.from_cli()
    print(training_args)

