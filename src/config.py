from typing import Literal, List
from pydantic_config import BaseConfig, parse_argv
from pydantic import field_validator

class WorldConfig(BaseConfig):
    num_stages: int = 2

class ModelConfig(BaseConfig):
    n_layer: int
    n_head: int
    n_embd: int
    parameter_sharing: bool = False # Share weights for embeddings and LM head

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
    precision: Literal["highest", "high", "medium"] = "highest"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"

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
    enable: bool = True
    every_n_steps: int = -1
    max_epochs: int = 1
    max_steps: int = -1

class SampleConfig(BaseConfig):
    enable: bool = True
    every_n_steps: int = -1
    num_samples: int = 5
    prompt: str = "I am"
    max_new_tokens: int = 20
    temperature : float = 1.0
    top_k : int | None = 20

class CheckpointingConfig(BaseConfig):
    enable: bool = False
    every_n_steps: int = -1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool = False
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool = True
    log_level: str = "DEBUG"

class WandbLoggingConfig(BaseConfig):
    enable: bool = False
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    tags: List[str] = []
    group: str | None = None
    run_name: str | None = None
    cache_dir: str | None = None

    @field_validator('tags', mode='before')
    def split_string_tags(cls, v):
        if isinstance(v, str):
            return v.split(',')
        return v

class LoggingConfig(BaseConfig):
    log_dir: str = "logs"
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbLoggingConfig = WandbLoggingConfig()
    ckpt: CheckpointingConfig = CheckpointingConfig()