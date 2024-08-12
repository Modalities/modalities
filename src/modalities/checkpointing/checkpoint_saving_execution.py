from abc import ABC, abstractmethod

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointSavingExecutionABC(ABC):
    """Abstract class for saving PyTorch model and optimizer checkpoints."""

    @abstractmethod
    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, num_train_steps_done: int):
        """
        Saves the checkpoint of the model and optimizer.

        Args:
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            num_train_steps_done (int): The number of training steps completed.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, num_train_steps_done: int):
        """
        Deletes the checkpoint.

        Args:
            num_train_steps_done (int): The number of training steps completed.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        num_train_steps_done: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        """
        Runs the checkpoint instruction.

        Args:
            checkpointing_instruction (CheckpointingInstruction): The checkpointing instruction.
            num_train_steps_done (int): The number of training steps done.
            model (nn.Module): The model.
            optimizer (Optimizer): The optimizer.
        """
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, num_train_steps_done=num_train_steps_done)

        for num_train_steps_done in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(num_train_steps_done=num_train_steps_done)
