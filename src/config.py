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

class OptimizerConfig(BaseConfig):
    lr: float = 6e-4
    decay: float = 0.1
    betas: list[float] = [0.9, 0.95]
    
class SchedulerConfig(BaseConfig):
    enable: bool = False
    warmup_steps: int = 100
    num_cycles: float = 0.5
    min_lr_factor: float = 0.1
    last_epoch: int = -1

class TrainConfig(BaseConfig):
    max_epochs: int = 1
    max_steps: int = 10
    micro_batch_size: int = 32
    batch_size: int = 32
    seed: int = 42

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

class EvalConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1
    max_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 1

class CheckpointingConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool = True
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool = True
    log_level: str = "DEBUG"
    name: str = "output.log"

class WandbLoggingConfig(BaseConfig):
    enable: bool = False
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    group: str | None = None
    run_name: str | None = None

class LoggingConfig(BaseConfig):
    log_dir: str = "logs"
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbLoggingConfig = WandbLoggingConfig()
    ckpt: CheckpointingConfig = CheckpointingConfig()

class Config(BaseConfig):
    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    logging: LoggingConfig = LoggingConfig()

if __name__ == "__main__":
    import yaml

    class TestConfig(BaseConfig):
        model: ModelConfig = ModelConfig()
        tokenizer: TokenizerConfig = TokenizerConfig()
        data: DataConfig = DataConfig()
        train: TrainConfig = TrainConfig()
        eval: EvalConfig = EvalConfig()
        logging: LoggingConfig = LoggingConfig()

    config = TestConfig(**parse_argv())
    print(yaml.dump(config.model_dump(), sort_keys=False))
