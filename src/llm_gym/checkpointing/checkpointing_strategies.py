from llm_gym.checkpointing.checkpointing import (
    CheckpointingStrategyIF,
    CheckpointingInstruction,
)
from llm_gym.batch import EvaluationResultBatch


class SaveMostRecentEpochOnlyCheckpointingStrategy(CheckpointingStrategyIF):
    """Class for Save Last Epoch only Checkpointing Strategy."""

    def __init__(self):
        self.saved_batch_id_checkpoints = []

    def get_model_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        checkpoints_to_delete = self.saved_batch_id_checkpoints.copy()
        self.saved_batch_id_checkpoints = [global_train_batch_id]
        return CheckpointingInstruction(
            save_current=True, checkpoints_to_delete=checkpoints_to_delete
        )


class SaveLastEpochOnlyCheckpointingStrategy(CheckpointingStrategyIF):
    """
    Class for Save Last Epoch only Checkpointing Strategy.
    """

    def __init__(self):
        pass

    def get_model_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        checkpoints_to_delete = []
        save_current = (
            global_train_batch_id + 1 == num_batches or early_stoppping_criterion_fulfilled
        )
        return CheckpointingInstruction(
            save_current=save_current, checkpoints_to_delete=checkpoints_to_delete
        )


class SaveAllCheckpointingStrategy(CheckpointingStrategyIF):
    """
    Class for Save All Checkpointing Stratergy.
    """

    def __init__(self):
        pass

    def get_model_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        return CheckpointingInstruction(save_current=True, checkpoints_to_delete=[])
