from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.config.config import PrecisionEnum
from modalities.util import compute_number_of_trainable_parameters


class TorchCheckpointLoading(CheckpointLoadingIF):
    def __init__(self, device: torch.device, precision: PrecisionEnum):
        self.device = device
        self.precision = precision

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        model_state = torch.load(file_path)
        model.load_state_dict(model_state)
        # set the model to the correct device and precision
        model = model.to(self.precision.value)
        model = model.to(self.device)
        print(
            f"Model loaded with {compute_number_of_trainable_parameters(model)} trainable parameters from {file_path}"
        )
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: nn.Module, file_path: Path) -> Optimizer:
        raise NotImplementedError  # TODO future work
