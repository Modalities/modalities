from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import torch

from modalities.exceptions import BatchStateError


class TorchDeviceMixin(ABC):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def detach(self):
        raise NotImplementedError


class Batch(ABC):
    """Abstract class that defines the necessary methods any `Batch` implementation needs to implement."""

    pass


@dataclass
class DatasetBatch(Batch, TorchDeviceMixin):
    """A batch of samples and its targets. Used to batch train a model."""

    samples: Dict[str, torch.Tensor]
    targets: Dict[str, torch.Tensor]
    batch_dim: int = 0

    def to(self, device: torch.device):
        self.samples = {k: v.to(device) for k, v in self.samples.items()}
        self.targets = {k: v.to(device) for k, v in self.targets.items()}

    def detach(self):
        self.targets = {k: v.detach() for k, v in self.targets.items()}
        self.samples = {k: v.detach() for k, v in self.samples.items()}

    @property
    def device(self) -> torch.device:
        key = list(self.samples.keys())[0]
        return self.samples[key].device

    def __len__(self) -> int:
        key = list(self.samples.keys())[0]
        return self.samples[key].shape[self.batch_dim]


@dataclass
class InferenceResultBatch(Batch, TorchDeviceMixin):
    """Stores targets and predictions of an entire batch."""

    targets: Dict[str, torch.Tensor]
    predictions: Dict[str, torch.Tensor]
    batch_dim: int = 0

    def to_cpu(self):
        self.to(device=torch.device("cpu"))

    @property
    def device(self) -> torch.device:
        key = list(self.targets.keys())[0]
        return self.targets[key].device

    def to(self, device: torch.device):
        self.predictions = {k: v.to(device) for k, v in self.predictions.items()}
        self.targets = {k: v.to(device) for k, v in self.targets.items()}

    def detach(self):
        self.targets = {k: v.detach() for k, v in self.targets.items()}
        self.predictions = {k: v.detach() for k, v in self.predictions.items()}

    def get_predictions(self, key: str) -> torch.Tensor:
        if key not in self.predictions:
            raise BatchStateError(f"Key {key} not present in predictions!")
        return self.predictions[key]

    def get_targets(self, key: str) -> torch.Tensor:
        if key not in self.targets:
            raise BatchStateError(f"Key {key} not present in targets!")
        return self.targets[key]

    def __len__(self) -> int:
        key = list(self.predictions.keys())[0]
        return self.predictions[key].shape[self.batch_dim]


@dataclass
class EvaluationResultBatch(Batch):
    """Data class for storing the results of a single or multiple batches.
    Also entire epoch results are stored in here."""

    dataloader_tag: str
    train_step_id: int
    losses: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    metrics: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())
    throughput_metrics: Dict[str, torch.Tensor] = field(default_factory=lambda: dict())

    def __str__(self) -> str:
        eval_str = f"Evaluation result on dataset tag {self.dataloader_tag} after {self.train_step_id + 1} steps:"
        eval_str += "\n\nlosses: " + "\n\t".join([f"{k}: {v.mean().item()}" for k, v in self.losses.items()])
        eval_str += "\n\nmetrics: " + "\n\t".join([f"{k}: {v.mean().item()}" for k, v in self.metrics.items()])
        eval_str += "\n\nthroughput metrics: " + "\n\t".join(
            [f"{k}: {v.mean().item()}" for k, v in self.throughput_metrics.items()]
        )
        eval_str += "\n==============================================="
        return eval_str
