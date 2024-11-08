from typing import Literal
from pydantic_config import BaseConfig, parse_argv

class ModelConfig(BaseConfig):
    name: str

class DataConfig(BaseConfig):
    path: str
    name: str | None = None
    seq_length: int
    num_workers: int = 1
    subset_size: float = 1.0

class OptimizerConfig(BaseConfig):
    lr: float = 6e-4
    decay: float = 0.1
    betas: list[float] = [0.9, 0.95]
    
class SchedulerConfig(BaseConfig):
    enable: bool
    warmup_steps: int = 100
    num_cycles: float = 0.5
    min_lr_factor: float = 0.1
    last_epoch: int = -1

class AmpConfig(BaseConfig):
    precision: Literal["highest", "high", "medium"] = "high"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

class TrainConfig(BaseConfig):
    max_steps: int
    max_epochs: int
    micro_batch_size: int
    batch_size: int 
    seed: int = 42
    max_norm: float = 1.0

    amp: AmpConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

class EvalConfig(BaseConfig):
    enable: bool
    every_n_steps: int = -1
    max_epochs: int = 1
    max_steps: int = -1

class SampleConfig(BaseConfig):
    enable: bool
    every_n_steps: int = -1
    prompt: str = "Hey, who are you?"
    num_return_sequences: int = 10
    top_k: int = 50
    top_p : float = 0.95
    max_new_tokens: int = 50
    temperature: float = 0.7

class CheckpointingConfig(BaseConfig):
    enable: bool
    every_n_steps: int = -1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool
    log_level: str = "DEBUG"

class WandbLoggingConfig(BaseConfig):
    enable: bool
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    group: str | None = None
    run_name: str | None = None
    cache_dir: str | None = None

class LoggingConfig(BaseConfig):
    log_dir: str = "logs"
    console: ConsoleLoggingConfig
    file: FileLoggingConfig
    wandb: WandbLoggingConfig
    ckpt: CheckpointingConfig

if __name__ == "__main__":
    import yaml

    class TestConfig(BaseConfig):
        model: ModelConfig = ModelConfig()
        data: DataConfig = DataConfig()
        train: TrainConfig = TrainConfig()
        eval: EvalConfig = EvalConfig()
        logging: LoggingConfig = LoggingConfig()

    config = TestConfig(**parse_argv())
    print(yaml.dump(config.model_dump(), sort_keys=False))
