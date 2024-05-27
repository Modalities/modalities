from logging import warning
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.config.config import PrecisionEnum
from modalities.util import compute_number_of_trainable_parameters


class TorchCheckpointLoading(CheckpointLoadingIF):
    def __init__(self, device: torch.device, precision: Optional[PrecisionEnum] = None):
        """Loads a a model checkpoint via the general pytorch checkpoint loading implementation.

        Args:
            device (torch.device): The device to load the model on.
            precision (Optional[PrecisionEnum], optional): If specified, the model checkpoint will
                loaded with the given precision. Otherwise, the precision as specified in the state_dict
                will be used. Defaults to None.
        """
        self.device = device
        self.precision = precision

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        if self.precision is not None:
            model = model.to(self.device, dtype=self.precision.value)
        else:
            model = model.to(self.device)

        model_state = torch.load(file_path, map_location=self.device)
        model_state_dtype = list(model_state.values())[0].dtype

        if self.precision is not None and self.precision.value != model_state_dtype:
            warning(
                f"WARNING: Model checkpoint was stored with precision {model_state_dtype} "
                "but is loaded with precision {self.precision.value}."
            )

        # assign=True makes sure that the model is loaded with the same precision
        # as specified in the state_dict. When precision is set to None,
        # the model is loaded with the precision that is set in the state_dict.
        model.load_state_dict(model_state, assign=self.precision is None)
        # set the model to the correct device and precision
        # model = model.to(self.precision.value)
        print(
            f"Model loaded with {compute_number_of_trainable_parameters(model)} trainable parameters from {file_path}"
        )
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: nn.Module, file_path: Path) -> Optimizer:
        raise NotImplementedError  # TODO future work
