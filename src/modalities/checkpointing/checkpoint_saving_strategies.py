import dataclasses
from abc import ABC, abstractmethod
from typing import Optional

from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction
from modalities.training.training_progress import TrainingProgress


class CheckpointSavingStrategyIF(ABC):
    """Checkpoint saving strategy interface."""

    @abstractmethod
    def get_checkpoint_instruction(
        self,
        training_progress: TrainingProgress,
        evaluation_result: Optional[dict[str, EvaluationResultBatch]] = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Returns the checkpointing instruction.

        Parameters:
            training_progress (TrainingProgress): The training progress.
            evaluation_result (dict[str, EvaluationResultBatch] | None, optional):
            The evaluation result. Defaults to None.
            early_stoppping_criterion_fulfilled (bool, optional):
            Whether the early stopping criterion is fulfilled. Defaults to False.

        Returns:
            CheckpointingInstruction: The checkpointing instruction.
        """
        raise NotImplementedError


class SaveKMostRecentCheckpointsStrategy(CheckpointSavingStrategyIF):
    """Strategy for saving the k most recent checkpoints only."""

    def __init__(self, k: int = -1):
        """Initializes the checkpoint saving strategy.

        Args:
            k (int, optional): The number of most recent checkpoints to save.
                Defaults to -1, which means all checkpoints are saved.
                Set to 0 to not save any checkpoints.
                Set to a positive integer to save the specified number of
                checkpointsStrategy for saving the k most recent checkpoints only.
        """
        self.saved_step_checkpoints: list[TrainingProgress] = []
        self.k = k

    def get_checkpoint_instruction(
        self,
        training_progress: TrainingProgress,
        evaluation_result: dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Generates a checkpointing instruction based on the given parameters.

        Args:
            training_progress (TrainingProgress): The training progress.
            evaluation_result (dict[str, EvaluationResultBatch] | None, optional):
                The evaluation result. Defaults to None.
            early_stoppping_criterion_fulfilled (bool, optional):
                Whether the early stopping criterion is fulfilled. Defaults to False.

        Returns:
            CheckpointingInstruction: The generated checkpointing instruction.
        """
        checkpoints_to_delete: list[TrainingProgress] = []
        save_current = True

        if self.k > 0:
            self.saved_step_checkpoints = [dataclasses.replace(training_progress)] + self.saved_step_checkpoints
            if len(self.saved_step_checkpoints) > self.k:
                # Delete oldest checkpoint
                checkpoints_to_delete = [self.saved_step_checkpoints[-1]]
                self.saved_step_checkpoints = self.saved_step_checkpoints[:-1]
        elif self.k == 0:
            save_current = False
        elif self.k == -1:
            self.saved_step_checkpoints = [dataclasses.replace(training_progress)] + self.saved_step_checkpoints

        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=checkpoints_to_delete)


class SaveEveryKStepsCheckpointingStrategy(CheckpointSavingStrategyIF):
    def __init__(self, k: int):
        """
        Initializes the CheckpointSavingStrategy object.

        Args:
            k (int): The value of k.

        Returns:
            None
        """
        self.k = k

    def get_checkpoint_instruction(
        self,
        training_progress: TrainingProgress,
        evaluation_result: dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Returns a CheckpointingInstruction object.

        Args:
            training_progress (TrainingProgress): The training progress.
            evaluation_result (dict[str, EvaluationResultBatch] | None, optional):
            The evaluation result. Defaults to None.
            early_stoppping_criterion_fulfilled (bool, optional):
            Whether the early stopping criterion is fulfilled. Defaults to False.

        Returns:
            CheckpointingInstruction: The checkpointing instruction object.
        """
        save_current = training_progress.num_seen_steps_total % self.k == 0
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=[])
