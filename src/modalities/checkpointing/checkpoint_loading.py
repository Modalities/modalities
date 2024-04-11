from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer


class CheckpointLoadingIF(ABC):
    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        file_path: Path,
    ) -> Optimizer:
        raise NotImplementedError
