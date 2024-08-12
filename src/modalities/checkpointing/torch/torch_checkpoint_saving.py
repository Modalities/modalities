from pathlib import Path

import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class TorchCheckpointSaving(CheckpointSavingExecutionABC):
    CHECKPOINT_STRUCTURE = "eid_{experiment_id}-{entity}-num_samples_{num_samples}.bin"

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
        global_train_sample_id: int,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        """
        Runs the checkpointing instruction for saving checkpoints.

        Args:
            checkpointing_instruction (CheckpointingInstruction): The instruction for checkpointing.
            global_train_sample_id (int): The global train sample ID.
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented yet. It is reserved for future work.
        """
        raise NotImplementedError  # TODO Future work

    def _save_checkpoint(self, model: FSDP, optimizer: Optimizer, global_train_sample_id: int):
        """
        Save the checkpoint of the model and optimizer.

        Args:
            model (FSDP): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            global_train_sample_id (int): The global train sample ID.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented yet. It is reserved for future work.
        """
        raise NotImplementedError  # TODO Future work

    def _delete_checkpoint(self, global_train_sample_id: int):
        """
        Deletes the checkpoint for the given global train sample ID.

        Args:
            global_train_sample_id (int): The global train sample ID.

        Returns:
            None

        Raises:
            NotImplementedError: This method is not implemented yet. It is reserved for future work.
        """
        raise NotImplementedError  # TODO Future work
