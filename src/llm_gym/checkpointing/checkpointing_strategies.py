from llm_gym.checkpointing.checkpointing import CheckpointingStrategyIF, CheckpointingInstruction
from llm_gym.batch import EvaluationResultBatch


class SaveMostRecentEpochOnlyCheckpointingStrategy(CheckpointingStrategyIF):
    """
    Class for Save Last Epoch only Checkpointing Strategy.
    """

    def __init__(self):
        pass

    def get_model_checkpoint_instruction(self, current_epoch: int, num_epochs: int,
                                         evaluation_result: EvaluationResultBatch,
                                         early_stoppping_criterion_fulfilled: bool = False) -> CheckpointingInstruction:
        """
        Fetch Checkpoint Instruction 

        :params:
               current_epoch (int): Current epoch number for cerating checkpoints.
               num_epochs (int): Number of epochs to be trained.
               evaluation_result (EvaluationBatchResult): Evaluation results of batches trained on.

        :returns:
            CheckpointingInstruction: Instruction to save and delete checkpoints.
        """
        checkpoints_to_delete = [current_epoch-1] if current_epoch > 0 else []
        return CheckpointingInstruction(save_current=True, checkpoints_to_delete=checkpoints_to_delete)


class SaveLastEpochOnlyCheckpointingStrategy(CheckpointingStrategyIF):
    """
    Class for Save Last Epoch only Checkpointing Strategy.
    """

    def __init__(self):
        pass

    def get_model_checkpoint_instruction(self, current_epoch: int, num_epochs: int,
                                         evaluation_result: EvaluationResultBatch,
                                         early_stoppping_criterion_fulfilled: bool = False) -> CheckpointingInstruction:
        """
        Fetch Checkpoint Instruction 

        :params:
               current_epoch (int): Current epoch number for cerating checkpoints.
               num_epochs (int): Number of epochs to be trained.
               evaluation_result (EvaluationBatchResult): Evaluation results of batches trained on.

        :returns:
            CheckpointingInstruction: Instruction to save and delete checkpoints.
        """
        checkpoints_to_delete = []
        save_current = current_epoch == num_epochs-1 or early_stoppping_criterion_fulfilled
        return CheckpointingInstruction(save_current=save_current, checkpoints_to_delete=checkpoints_to_delete)


class SaveAllCheckpointingStrategy(CheckpointingStrategyIF):
    """
    Class for Save All Checkpointing Stratergy.
    """

    def __init__(self):
        pass

    def get_model_checkpoint_instruction(self, current_epoch: int, num_epochs: int,
                                         evaluation_result: EvaluationResultBatch,
                                         early_stoppping_criterion_fulfilled: bool = False) -> CheckpointingInstruction:
        """
        Fetch Checkpoint Instruction 

        :params:
               current_epoch (int): Current epoch number for cerating checkpoints.
               num_epochs (int): Number of epochs to be trained.
               evaluation_result (EvaluationBatchResult): Evaluation results of batches trained on.

        :returns:
            CheckpointingInstruction: Instruction to save and delete checkpoints.
        """
        return CheckpointingInstruction(save_current=True, checkpoints_to_delete=[])
