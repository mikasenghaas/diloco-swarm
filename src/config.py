from typing import Literal, List
from pydantic_config import BaseConfig, parse_argv
from pydantic import field_validator

class SwarmConfig(BaseConfig):
    num_stages: int = 2
    sync_every_n_steps: int = 1

class ModelConfig(BaseConfig):
    n_layer: int
    n_head: int
    n_embd: int
    parameter_sharing: bool = False # Share weights for embeddings and LM head

class DataConfig(BaseConfig):
    path: str
    seq_length: int = 1024
    num_workers: int = 1
    subset_size: float = 1.0

class OptimizerConfig(BaseConfig):
    type: Literal["SGD", "AdamW", "None"]
    lr: float
    nesterov: bool = False
    momentum: float = 0.9
    weight_decay: float = 0.1
    betas: list[float] = [0.9, 0.95]
    
class SchedulerConfig(BaseConfig):
    enable: bool = False
    num_warmup_steps: int = 100
    num_cycles: float = 0.5
    min_lr_factor: float = 0.1
    last_epoch: int = -1

class AmpConfig(BaseConfig):
    enable: bool = True
    precision: Literal["highest", "high", "medium"] = "high"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

class TrainConfig(BaseConfig):
    max_epochs: int = 1
    max_steps: int = -1
    micro_batch_size: int = 16
    batch_size: int = 512
    max_micro_batches: int = 1 # Maximum number of micro batches in memory
    seed: int = 42
    max_norm: float = 1.0
    step_timeout: float = 60 # Timeout step after one minute (adjust to hardware!)
    
    inner_optimizer: OptimizerConfig
    outer_optimizer: OptimizerConfig
    scheduler: SchedulerConfig = SchedulerConfig()

class EvalConfig(BaseConfig):
    enable: bool = True
    every_n_steps: int = -1
    eval_size: float = 0.1

class SampleConfig(BaseConfig):
    enable: bool = True
    every_n_steps: int = -1
    num_samples: int = 5
    prompt: str = "I am"
    max_new_tokens: int = 20
    temperature : float = 1.0
    top_k : int | None = 20

class ConsoleLoggingConfig(BaseConfig):
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    log_level: str = "DEBUG"

class WandbLoggingConfig(BaseConfig):
    enable: bool = False
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    tags: List[str] = []
    group: str | None = None
    cache_dir: str | None = None

    @field_validator('tags', mode='before')
    def split_string_tags(cls, v):
        if isinstance(v, str):
            return v.split(',')
        return v

class LoggingConfig(BaseConfig):
    log_dir: str = "logs"
    run_id: str | None = None
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    file: FileLoggingConfig = FileLoggingConfig()
    wandb: WandbLoggingConfig = WandbLoggingConfig()