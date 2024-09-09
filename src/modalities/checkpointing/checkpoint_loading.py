from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer


class CheckpointLoadingIF(ABC):
    """Checkpoint loading interface for loading PyTorch models and optimizer checkpoints."""

    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        """
        Loads a model checkpoint from the specified file path.

        Args:
            model (nn.Module): The model to load the checkpoint into.
            file_path (Path): The path to the checkpoint file.

        Returns:
            nn.Module: The loaded model with the checkpoint parameters.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        file_path: Path,
    ) -> Optimizer:
        """
        Loads an optimizer checkpoint from the specified file path.

        Args:
            optimizer (Optimizer): The optimizer to load the checkpoint into.
            model (nn.Module): The model associated with the optimizer.
            file_path (Path): The path to the checkpoint file.

        Returns:
            Optimizer: The loaded optimizer with the checkpoint parameters.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError
