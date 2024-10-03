"""
How to use Python dataclasses to define arguments for a program.
"""
from pydantic_config import BaseConfig, parse_argv

class ModelConfig(BaseConfig):
    name: str = "PrimeIntellect/llama-14m-fresh"

class TokenizerConfig(BaseConfig):
    name: str = "meta-llama/Llama-2-7b-hf"
    fast: bool = True

class DataConfig(BaseConfig):
    path: str = "Salesforce/wikitext"
    name: str = "wikitext-2-raw-v1"
    seq_length: int = 128
    num_workers: int = 1
    subset_size: float = 1.0
    
class TrainConfig(BaseConfig):
    seed: int = 42

    warmup_steps: int = 1000
    max_steps: int = 100
    batch_size: int = 4

    lr: float = 4e-4
    weight_decay: float = 0.1
    adam_betas: tuple[float, float] = (0.9, 0.95)

class EvalConfig(BaseConfig):
    enable: bool = True
    every_n_steps: int = 10
    max_steps: int = 100
    batch_size: int = 1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool = True
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool = False
    path: str = "logs"
    log_level: str = "DEBUG"

class WandbLoggingConfig(BaseConfig):
    enable: bool = False
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    run_name: str | None = None

class LoggingConfig(BaseConfig):
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbLoggingConfig = WandbLoggingConfig()

class Config(BaseConfig):
    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    logging: LoggingConfig = LoggingConfig()


if __name__ == "__main__":
    class TestConfig(BaseConfig):
        model: ModelConfig = ModelConfig()
        tokenizer: TokenizerConfig = TokenizerConfig()
        data: DataConfig = DataConfig()
        training: TrainConfig = TrainConfig()
        eval: EvalConfig = EvalConfig()
        logging: LoggingConfig = LoggingConfig()

    config = TestConfig(**parse_argv())
    print(config)
