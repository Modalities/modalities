from abc import ABC, abstractmethod

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointSavingExecutionABC(ABC):
    @abstractmethod
    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, num_train_steps_done: int):
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, num_train_steps_done: int):
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        num_train_steps_done: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, num_train_steps_done=num_train_steps_done)

        for num_train_steps_done in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(num_train_steps_done=num_train_steps_done)
