from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.stateful.app_state import AppState


class DistributedCheckpointLoadingIF(ABC):
    """Distributed checkpoint loading interface for loading PyTorch models and optimizer checkpoints."""

    @abstractmethod
    def load_checkpoint_(self, app_state: AppState, checkpoint_directory_path: Path) -> AppState:
        raise NotImplementedError


class FSDP1CheckpointLoadingIF(ABC):
    """Checkpoint loading interface for loading PyTorch models and optimizer checkpoints."""

    @abstractmethod
    def load_model_checkpoint_(self, model: nn.Module, file_path: Path) -> nn.Module:
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
    def load_optimizer_checkpoint_(
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
