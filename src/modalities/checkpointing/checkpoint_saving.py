from enum import Enum
from typing import Dict

import torch.nn as nn
from torch.optim import Optimizer

from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpoint_saving_execution import CheckpointSavingExecutionABC
from modalities.checkpointing.checkpoint_saving_strategies import CheckpointSavingStrategyIF


class CheckpointEntityType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"


class CheckpointSaving:
    """
    Checkpoint class to get checkpoint instruction.
    """

    def __init__(
        self,
        checkpointing_strategy: CheckpointSavingStrategyIF,
        checkpointing_execution: CheckpointSavingExecutionABC,
    ):
        self.checkpointing_strategy = checkpointing_strategy
        self.checkpointing_execution = checkpointing_execution

    def save_checkpoint(
        self,
        global_train_sample_id: int,
        evaluation_result: Dict[str, EvaluationResultBatch],
        model: nn.Module,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        checkpointing_instruction = self.checkpointing_strategy.get_checkpoint_instruction(
            global_train_sample_id=global_train_sample_id,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )

        self.checkpointing_execution.run_checkpoint_instruction(
            checkpointing_instruction=checkpointing_instruction,
            global_train_sample_id=global_train_sample_id,
            model=model,
            optimizer=optimizer,
        )