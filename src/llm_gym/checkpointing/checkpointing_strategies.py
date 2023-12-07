from typing import Dict
from llm_gym.checkpointing.checkpointing import CheckpointingStrategyIF

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing_instruction import CheckpointingInstruction


class SaveKMostRecentCheckpointsStrategy(CheckpointingStrategyIF):
    def __init__(self, k: int = -1):
        """Strategy for saving the k most recent checkpoints only.
        k=None: keep all checkpoints
        k>0: keep k checkpoints
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

        if self.k > 0 and len(self.saved_batch_id_checkpoints) > self.k:
            # Delete oldest checkpoint
            checkpoints_to_delete = [self.saved_batch_id_checkpoints[-1]]
            self.saved_batch_id_checkpoints = self.saved_batch_id_checkpoints[:-1]

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
