from typing import Dict

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing import CheckpointingStrategyIF
from llm_gym.checkpointing.checkpointing_instruction import CheckpointingInstruction
from llm_gym.exceptions import CheckpointingError


class SaveKMostRecentCheckpointsStrategy(CheckpointingStrategyIF):
    def __init__(self, k: int = -1):
        """Strategy for saving the k most recent checkpoints only.
        k=-1: keep all checkpoints
        k=0: don't keep any checkpoint
        k>0: keep k checkpoints
        """
        self.saved_sample_id_checkpoints = []
        if k < -1:
            raise CheckpointingError("Checkpointing strategy needs k to be in {-1, 0, 1, ..., n}")
        self.k = k

    def get_checkpoint_instruction(
        self,
        global_train_sample_id: int,
        global_num_train_samples: int = None,
        evaluation_result: Dict[str, EvaluationResultBatch] = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        checkpoints_to_delete = []
        save_current = True

        if self.k > 0:
            self.saved_sample_id_checkpoints = [global_train_sample_id] + self.saved_sample_id_checkpoints
            if len(self.saved_sample_id_checkpoints) > self.k:
                # Delete oldest checkpoint
                checkpoints_to_delete = [self.saved_sample_id_checkpoints[-1]]
                self.saved_sample_id_checkpoints = self.saved_sample_id_checkpoints[:-1]
        elif self.k == 0:
            save_current = False
        elif self.k == -1:
            self.saved_sample_id_checkpoints = [global_train_sample_id] + self.saved_sample_id_checkpoints

        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=checkpoints_to_delete)


class SaveEveryKStepsCheckpointingStrategy(CheckpointingStrategyIF):
    def __init__(self, k: int):
        self.k = k

    def get_checkpoint_instruction(
        self,
        global_train_sample_id: int,
        global_num_train_samples: int = None,
        evaluation_result: Dict[str, EvaluationResultBatch] = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        save_current = (global_train_sample_id + 1) % self.k == 0 and (global_train_sample_id + 1) > 0
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=[])
