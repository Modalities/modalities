from abc import ABC, abstractmethod
from typing import Dict

from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction


class CheckpointSavingStrategyIF(ABC):
    """
    Checkpoint Interface to get checkpoint instruction.
    """

    @abstractmethod
    def get_checkpoint_instruction(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        raise NotImplementedError


class SaveKMostRecentCheckpointsStrategy(CheckpointSavingStrategyIF):
    def __init__(self, k: int = -1):
        """Strategy for saving the k most recent checkpoints only.
        k=-1: keep all checkpoints
        k=0: don't keep any checkpoint
        k>0: keep k checkpoints
        """
        self.saved_step_checkpoints = []
        self.k = k

    def get_checkpoint_instruction(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
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
        self.k = k

    def get_checkpoint_instruction(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        save_current = num_train_steps_done % self.k == 0
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=[])
