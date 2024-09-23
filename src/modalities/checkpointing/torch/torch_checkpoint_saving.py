from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class TorchCheckpointSaving(CheckpointSavingExecutionABC):
    CHECKPOINT_STRUCTURE = (
        "eid_{experiment_id}-{entity}-seen_steps_{num_seen_steps}-seen_tokens_{num_seen_tokens}"
        "-target_steps_{num_target_steps}-target_tokens_{num_target_tokens}.bin"
    )

    def __init__(
        self,
        checkpoint_path: Path,
        experiment_id: str,
    ):
        """
        Initializes the TorchCheckpointSaving object.

        Args:
            checkpoint_path (Path): The path to save the checkpoint.
            experiment_id (str): The ID of the experiment.

        Returns:
            None
        """
        self.checkpoint_path = checkpoint_path
        self.experiment_id = experiment_id

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        num_train_steps_done: int,
        target_train_steps: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        """
        Runs the checkpointing instruction for saving checkpoints.

        Args:
            checkpointing_instruction (CheckpointingInstruction): The instruction for checkpointing.
            num_train_steps_done (int): The number of training steps done.
            target_train_steps (int): The target number of training steps.
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented yet. It is reserved for future work.
        """
        raise NotImplementedError  # TODO Future work

    def _save_checkpoint(
        self, model: nn.Module, optimizer: Optimizer, num_train_steps_done: int, target_train_steps: int
    ):
        """
        Saves the checkpoint of the model and optimizer.

        Args:
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            num_train_steps_done (int): The number of training steps completed.
            target_train_steps (int): The target number of training steps.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError  # TODO Future work

    def _delete_checkpoint(self, num_train_steps_done: int, target_train_steps: int):
        """
        Deletes the checkpoint based on the number of performed train steps and target train steps.

        Args:
            num_train_steps_done (int): The number of training steps completed.
            target_train_steps (int): The target number of training steps.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError  # TODO Future work
