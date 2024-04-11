from abc import ABC, abstractmethod

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointSavingExecutionABC(ABC):
    @abstractmethod
    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, global_train_sample_id: int):
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, global_train_sample_id: int):
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_sample_id: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, global_train_sample_id=global_train_sample_id)

        for global_train_sample_id in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(global_train_sample_id=global_train_sample_id)
