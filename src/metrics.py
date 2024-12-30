import math
from typing import List, Dict
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

class Outputs(BaseModel):
    step: int
    time: float
    loss: float
    tokens: int
    num_micro_batches: int = 1
    lr: float | None = None
    norm: float | None = None
    micro_step_time: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Metric(ABC):
    @abstractmethod
    def update(self, outputs: Outputs):
        pass

    @abstractmethod
    def compute(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def reset(self):
        pass

class Step(Metric):
    __name__ = "step"
    def __init__(self):
        self.step = 0

    def update(self, outputs: Outputs):
        self.step = outputs.step

    def compute(self) -> Dict[str, float]:
        return {"current": self.step}

    def reset(self):
        self.step = 0

class Time(Metric):
    """Time per step"""
    __name__ = "time"
    def __init__(self):
        self.times : List[float] = []

    def update(self, outputs: Outputs):
        self.times.append(outputs.time)

    def compute(self) -> Dict[str, float]:
        return {"current": self.times[-1] if self.times else 0}

    def reset(self):
        self.times = []

class MicroTime(Metric):
    """Avg. time per micro step"""
    __name__ = "micro_time"
    def __init__(self):
        self.times : List[float] = []

    def update(self, outputs: Outputs):
        self.times.append(outputs.micro_step_time)

    def compute(self) -> Dict[str, float]:
        return {"current": self.times[-1] if self.times else 0}

    def reset(self):
        self.times = []

class Tokens(Metric):
    __name__ = "tokens"
    def __init__(self):
        self.tokens : List[int] = []

    def update(self, outputs: Outputs):
        self.tokens.append(outputs.tokens)

    def compute(self) -> Dict[str, float]:
        return {"current": self.tokens[-1] if self.tokens else 0, "total": sum(self.tokens)}

    def reset(self):
        self.tokens = []

class NumMicroBatches(Metric):
    __name__ = "num_micro_batches"
    def __init__(self):
        self.num_micro_batches : int = 0

    def update(self, outputs: Outputs):
        self.num_micro_batches += outputs.num_micro_batches

    def compute(self) -> Dict[str, float]:
        return {"current": self.num_micro_batches}

    def reset(self):
        self.num_micro_batches = 0

class Norm(Metric):
    __name__ = "norm"
    def __init__(self):
        self.norms : List[float] = []

    def update(self, outputs: Outputs):
        self.norms.append(outputs.norm)

    def compute(self) -> Dict[str, float]:
        return {"current": self.norms[-1] if self.norms else 0}

    def reset(self):
        self.norms = []

class Loss(Metric):
    __name__ = "loss"
    def __init__(self):
        self.losses : List[float] = []

    def update(self, outputs: Outputs):
        self.losses.append(outputs.loss)

    def compute(self) -> Dict[str, float]:
        return {"current": self.losses[-1] if self.losses else 0, "average": sum(self.losses) / len(self.losses) if self.losses else 0}

    def reset(self):
        self.losses = []

class Perplexity(Metric):
    __name__ = "perplexity"
    def __init__(self):
        self.perplexities : List[float] = []

    def update(self, outputs: Outputs):
        self.perplexities.append(math.exp(outputs.loss) if outputs.loss > 0 else 0)

    def compute(self) -> Dict[str, float]:
        return {"current": self.perplexities[-1] if self.perplexities else 0, "average": sum(self.perplexities) / len(self.perplexities) if self.perplexities else 0}

    def reset(self):
        self.perplexities = []

class Throughput(Metric):
    __name__ = "throughput"
    def __init__(self):
        self.throughputs : List[float] = []

    def update(self, outputs: Outputs):
        throughput = outputs.tokens / outputs.time

        self.throughputs.append(throughput)

    def compute(self) -> Dict[str, float]:
        return {"current": self.throughputs[-1] if self.throughputs else 0, "average": sum(self.throughputs) / len(self.throughputs) if self.throughputs else 0}

    def reset(self):
        self.throughputs = []

class LearningRate(Metric):
    __name__ = "learning_rate"
    def __init__(self):
        self.learning_rates: List[float] = []

    def update(self, outputs: Outputs):
        self.learning_rates.append(outputs.lr)

    def compute(self) -> Dict[str, float]:
        return {"current": self.learning_rates[-1] if self.learning_rates else 0}

    def reset(self):
        self.learning_rates = []

class Metrics:
    def __init__(self, metrics: List[Metric], name: str):
        self.metrics = metrics
        self.name = name

    def update(self, outputs: Outputs):
        for metric in self.metrics:
            metric.update(outputs)

    def compute(self) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for metric in self.metrics:
            metrics.update({f"{self.name}/{metric.__name__}/{key}": value for key, value in metric.compute().items()})
        return metrics

    def reset(self):
        for metric in self.metrics:
            metric.reset()
