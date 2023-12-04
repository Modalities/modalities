from typing import Dict
from llm_gym.checkpointing.checkpointing import (
    CheckpointingStrategyIF,
    CheckpointingInstruction,
)
from llm_gym.batch import EvaluationResultBatch


class SaveKMostRecentCheckpointsStrategy(CheckpointingStrategyIF):
    def __init__(self, k: int = 0):
        """Strategy for saving the k most recent checkpoints only. If k is defined as 0, then we keep all checkpoints without any deletion.

        Args:
            k (int, optional): Number of most recent checkpoints  we want to keep. Defaults to 0, meaning we don't delete any checkpoints.
        """
        self.saved_batch_id_checkpoints = []
        self.k = k

    def get_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: Dict[str, EvaluationResultBatch] = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        self.saved_batch_id_checkpoints = [global_train_batch_id] + self.saved_batch_id_checkpoints
        checkpoints_to_delete = []
        # we only want to save checkpoints if k > 0 AND if we have more than k checkpoints in the backlog
        if self.k > 0 and len(self.saved_batch_id_checkpoints) > self.k:
            checkpoints_to_delete = [self.saved_batch_id_checkpoints[-1]]

        return CheckpointingInstruction(save_current=True, checkpoints_to_delete=checkpoints_to_delete)


class SaveEveryKStepsCheckpointingStrategy(CheckpointingStrategyIF):
    def __init__(self, k: int):
        self.k = k

    def get_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: Dict[str, EvaluationResultBatch],
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        save_current = (global_train_batch_id + 1) % self.k == 0 and (global_train_batch_id + 1) > 0
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=[])
