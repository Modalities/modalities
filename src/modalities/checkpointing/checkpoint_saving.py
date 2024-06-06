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
        checkpoint_saving_strategy: CheckpointSavingStrategyIF,
        checkpoint_saving_execution: CheckpointSavingExecutionABC,
    ):
        self.checkpoint_saving_strategy = checkpoint_saving_strategy
        self.checkpoint_saving_execution = checkpoint_saving_execution

    def save_checkpoint(
        self,
        num_train_steps_done: int,
        evaluation_result: Dict[str, EvaluationResultBatch],
        model: nn.Module,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        checkpointing_instruction = self.checkpoint_saving_strategy.get_checkpoint_instruction(
            num_train_steps_done=num_train_steps_done,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )

        self.checkpoint_saving_execution.run_checkpoint_instruction(
            checkpointing_instruction=checkpointing_instruction,
            num_train_steps_done=num_train_steps_done,
            model=model,
            optimizer=optimizer,
        )
