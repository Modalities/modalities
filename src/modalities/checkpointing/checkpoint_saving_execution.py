from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction
from modalities.training.training_progress import TrainingProgress


class CheckpointEntityType(Enum):
    """
    Enum class representing the types of entities that can be saved in a checkpoint.
    Attributes:
        MODEL (str): Represents the model entity.
        OPTIMIZER (str): Represents the optimizer entity.
    """

    MODEL = "model"
    OPTIMIZER = "optimizer"


class CheckpointSavingExecutionABC(ABC):
    """Abstract class for saving PyTorch model and optimizer checkpoints."""

    CHECKPOINT_STRUCTURE = (
        "eid_{experiment_id}-{entity}-seen_steps_{num_seen_steps}-seen_tokens_{num_seen_tokens}"
        "-target_steps_{num_target_steps}-target_tokens_{num_target_tokens}.bin"
    )

    @abstractmethod
    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, training_progress: TrainingProgress):
        """
        Saves the checkpoint of the model and optimizer.

        Args:
            model (nn.Module): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            training_progress (TrainingProgress): The training progress.

        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_checkpoint(self, training_progress: TrainingProgress):
        """
        Deletes the checkpoint based on the training progress.

        Args:
            training_progress (TrainingProgress): The training progress.

        Raises:
            NotImplementedError: This abstract method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError

    def run_checkpoint_instruction(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        training_progress: TrainingProgress,
        model: nn.Module,
        optimizer: Optimizer,
    ):
        """
        Runs the checkpoint instruction.

        Args:
            checkpointing_instruction (CheckpointingInstruction): The checkpointing instruction.
            training_progress (TrainingProgress): The training progress.
            model (nn.Module): The model.
            optimizer (Optimizer): The optimizer.
        """
        if checkpointing_instruction.save_current:
            self._save_checkpoint(model=model, optimizer=optimizer, training_progress=training_progress)

        for training_progress_to_delete in checkpointing_instruction.checkpoints_to_delete:
            self._delete_checkpoint(training_progress=training_progress_to_delete)

    def _get_checkpointing_path(
        self,
        experiment_id: str,
        num_seen_steps: int,
        num_seen_tokens: int,
        num_target_steps: int,
        num_target_tokens: int,
        entity_type: CheckpointEntityType,
    ) -> Path:
        entity_file_name = self.CHECKPOINT_STRUCTURE.format(
            experiment_id=experiment_id,
            entity=entity_type.value,
            num_seen_steps=str(num_seen_steps),
            num_seen_tokens=str(num_seen_tokens),
            num_target_steps=str(num_target_steps),
            num_target_tokens=str(num_target_tokens),
        )

        full_path = Path(self.checkpoint_path, experiment_id, entity_file_name)
        return full_path

    def _get_paths_to_delete(self, training_progress: TrainingProgress) -> list[Path]:
        return [
            self._get_checkpointing_path(
                experiment_id=self.experiment_id,
                entity_type=entity_type,
                num_seen_steps=training_progress.num_seen_steps_total,
                num_seen_tokens=training_progress.num_seen_tokens_total,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
            )
            for entity_type in CheckpointEntityType
        ]
