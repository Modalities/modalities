import torch.nn as nn
from torch.optim import AdamW, Optimizer

from modalities.checkpointing.checkpointing import Checkpointing


class OptimizerFactory:
    @staticmethod
    def get_adam_w(lr: float, wrapped_model: nn.Module):
        model_parameters = wrapped_model.parameters()
        optimizer = AdamW(params=model_parameters, lr=lr)
        return optimizer

    @staticmethod
    def get_checkpointed_optimizer(
        checkpointing: Checkpointing, checkpoint_path, wrapped_model: nn.Module, optimizer: Optimizer
    ):
        wrapped_optimizer = checkpointing.load_optimizer_checkpoint(
            file_path=checkpoint_path, optimizer=optimizer, wrapped_model=wrapped_model
        )
        return wrapped_optimizer
