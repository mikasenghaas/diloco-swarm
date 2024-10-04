import torch
from typing import List, Dict
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod

class Outputs(BaseModel):
    step: int
    time: float
    loss: torch.Tensor
    logits: torch.Tensor
    lr: float | None = None

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

    def update(self, _: Outputs):
        self.step += 1

    def compute(self) -> Dict[str, float]:
        return {"current": self.step}

    def reset(self):
        self.step = 0

class ExamplesSeen(Metric):
    __name__ = "examples"
    def __init__(self):
        self.num_examples : List[int] = []

    def update(self, outputs: Outputs):
        self.num_examples.append(outputs.logits.shape[0])

    def compute(self) -> Dict[str, float]:
        return {"current": self.num_examples[-1], "total": sum(self.num_examples)}

    def reset(self):
        self.num_examples = []

class TokensSeen(Metric):
    __name__ = "tokens"
    def __init__(self):
        self.num_tokens : List[int] = []

    def update(self, outputs: Outputs):
        self.num_tokens.append(outputs.logits.shape[0] * outputs.logits.shape[1])

    def compute(self) -> Dict[str, float]:
        return {"current": self.num_tokens[-1], "total": sum(self.num_tokens)}

    def reset(self):
        self.num_tokens = []

class Loss(Metric):
    __name__ = "loss"
    def __init__(self):
        self.losses : List[float] = []

    def update(self, outputs: Outputs):
        self.losses.append(outputs.loss.item())

    def compute(self) -> Dict[str, float]:
        return {"current": self.losses[-1], "average": sum(self.losses) / len(self.losses)}

    def reset(self):
        self.losses = []

class Perplexity(Metric):
    __name__ = "perplexity"
    def __init__(self):
        self.perplexities : List[float] = []

    def update(self, outputs: Outputs):
        self.perplexities.append(torch.exp(outputs.loss).item())

    def compute(self) -> Dict[str, float]:
        return {"current": self.perplexities[-1], "average": sum(self.perplexities) / len(self.perplexities)}

    def reset(self):
        self.perplexities = []

class Throughput(Metric):
    __name__ = "throughput"
    def __init__(self):
        self.throughputs : List[float] = []

    def update(self, outputs: Outputs):
        num_tokens = outputs.logits.shape[0] * outputs.logits.shape[1]
        throughput = num_tokens / outputs.time

        self.throughputs.append(throughput)

    def compute(self) -> Dict[str, float]:
        return {"current": self.throughputs[-1], "average": sum(self.throughputs) / len(self.throughputs)}

    def reset(self):
        self.throughputs = []

class LearningRate(Metric):
    __name__ = "learning_rate"
    def __init__(self):
        self.learning_rates: List[float] = []

    def update(self, outputs: Outputs):
        self.learning_rates.append(outputs.lr)

    def compute(self) -> Dict[str, float]:
        return {"current": self.learning_rates[-1]}

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
