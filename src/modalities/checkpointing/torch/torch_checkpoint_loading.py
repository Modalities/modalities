from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF


class TorchCheckpointLoading(CheckpointLoadingIF):
    def __init__(self, device: torch.device):
        self.device = device

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        model_state = torch.load(file_path)
        model.load_state_dict(model_state)
        model = model.to(self.device)
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: nn.Module, file_path: Path) -> Optimizer:
        raise NotImplementedError  # TODO future work
