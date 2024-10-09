from pydantic_config import BaseConfig, parse_argv

class ModelConfig(BaseConfig):
    name: str

class TokenizerConfig(BaseConfig):
    name: str
    fast: bool = True

class DataConfig(BaseConfig):
    path: str
    name: str
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

class TrainConfig(BaseConfig):
    max_steps: int
    max_epochs: int = -1
    micro_batch_size: int = 32
    batch_size: int = 32
    precision: str = "high"
    seed: int = 42
    max_norm: float = 1.0

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

class EvalConfig(BaseConfig):
    enable: bool
    every_n_steps: int = -1
    max_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 1

class CheckpointingConfig(BaseConfig):
    enable: bool
    every_n_steps: int = -1

class ConsoleLoggingConfig(BaseConfig):
    enable: bool
    log_level: str = "INFO"

class FileLoggingConfig(BaseConfig):
    enable: bool
    log_level: str = "DEBUG"
    name: str = "output.log"

class WandbLoggingConfig(BaseConfig):
    enable: bool
    entity: str | None = "mikasenghaas"
    project: str | None = "swarm"
    group: str | None = None
    run_name: str | None = None

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
        tokenizer: TokenizerConfig = TokenizerConfig()
        data: DataConfig = DataConfig()
        train: TrainConfig = TrainConfig()
        eval: EvalConfig = EvalConfig()
        logging: LoggingConfig = LoggingConfig()

    config = TestConfig(**parse_argv())
    print(yaml.dump(config.model_dump(), sort_keys=False))
