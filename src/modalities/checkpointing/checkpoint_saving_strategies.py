from abc import ABC, abstractmethod
from typing import Dict

from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointSavingStrategyIF(ABC):
    @abstractmethod
    def get_checkpoint_instruction(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Returns the checkpointing instruction.

        Parameters:
            num_train_steps_done (int): The number of training steps completed.
            evaluation_result (Dict[str, EvaluationResultBatch] | None, optional):
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
        self.saved_step_checkpoints = []
        self.k = k

    def get_checkpoint_instruction(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Generates a checkpointing instruction based on the given parameters.

        Args:
            num_train_steps_done (int): The number of training steps done.
            evaluation_result (Dict[str, EvaluationResultBatch] | None, optional):
            The evaluation result. Defaults to None.
            early_stoppping_criterion_fulfilled (bool, optional):
            Whether the early stopping criterion is fulfilled. Defaults to False.

        Returns:
            CheckpointingInstruction: The generated checkpointing instruction.
        """
        checkpoints_to_delete = []
        save_current = True

        if self.k > 0:
            self.saved_step_checkpoints = [num_train_steps_done] + self.saved_step_checkpoints
            if len(self.saved_step_checkpoints) > self.k:
                # Delete oldest checkpoint
                checkpoints_to_delete = [self.saved_step_checkpoints[-1]]
                self.saved_step_checkpoints = self.saved_step_checkpoints[:-1]
        elif self.k == 0:
            save_current = False
        elif self.k == -1:
            self.saved_step_checkpoints = [num_train_steps_done] + self.saved_step_checkpoints

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
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        """
        Returns a CheckpointingInstruction object.

        Args:
            num_train_steps_done (int): The number of training steps completed.
            evaluation_result (Dict[str, EvaluationResultBatch] | None, optional):
            The evaluation result. Defaults to None.
            early_stoppping_criterion_fulfilled (bool, optional):
            Whether the early stopping criterion is fulfilled. Defaults to False.

        Returns:
            CheckpointingInstruction: The checkpointing instruction object.
        """
        save_current = num_train_steps_done % self.k == 0
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=[])
