from typing import Literal
from pydantic_config import BaseConfig, parse_argv

class ModelConfig(BaseConfig):
    n_layer: int
    n_head: int
    n_embd: int

class DataConfig(BaseConfig):
    path: str
    seq_length: int
    name: str | None = None
    num_workers: int = 1
    subset_size: float = 1.0

class OptimizerConfig(BaseConfig):
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: list[float] = [0.9, 0.95]
    
class SchedulerConfig(BaseConfig):
    enable: bool = False
    num_warmup_steps: int = 100
    num_cycles: float = 0.5
    min_lr_factor: float = 0.1
    last_epoch: int = -1

class AmpConfig(BaseConfig):
    precision: Literal["highest", "high", "medium"] = "high"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

class TrainConfig(BaseConfig):
    max_steps: int
    micro_batch_size: int
    batch_size: int 
    max_epochs: int = -1
    seed: int = 42
    max_norm: float = 1.0

    amp: AmpConfig = AmpConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

class EvalConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1
    max_epochs: int = 1
    max_steps: int = -1

class SampleConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1
    num_samples: int = 5
    prompt: str = "I am"
    max_new_tokens: int = 20
    temperature : float = 1.0
    top_k : int | None = None

class CheckpointingConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool = True
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool = False
    log_level: str = "DEBUG"

class WandbLoggingConfig(BaseConfig):
    enable: bool = False
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    group: str | None = None
    run_name: str | None = None
    cache_dir: str | None = None

class LoggingConfig(BaseConfig):
    log_dir: str = "logs"
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbLoggingConfig = WandbLoggingConfig()
    ckpt: CheckpointingConfig = CheckpointingConfig()

if __name__ == "__main__":
    import yaml

    class TestConfig(BaseConfig):
        model: ModelConfig
        data: DataConfig
        train: TrainConfig
        eval: EvalConfig
        logging: LoggingConfig

    config = TestConfig(**parse_argv())
    print(yaml.dump(config.model_dump(), sort_keys=False))
