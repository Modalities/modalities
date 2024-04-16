from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch.nn as nn
from torch.optim import Optimizer

from modalities.batch import EvaluationResultBatch
from modalities.checkpointing.checkpointing_execution import CheckpointingExecutionIF
from modalities.checkpointing.checkpointing_strategies import CheckpointingStrategyIF


class CheckpointingIF(ABC):
    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        file_path: Path,
    ) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(
        self,
        global_train_sample_id: int,
        evaluation_result: Dict[str, EvaluationResultBatch],
        model: nn.Module,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        raise NotImplementedError


class Checkpointing(CheckpointingIF):
    """
    Checkpoint class to get checkpoint instruction.
    """

    def __init__(
        self,
        checkpointing_strategy: CheckpointingStrategyIF,
        checkpointing_execution: CheckpointingExecutionIF,
    ):
        self.checkpointing_strategy = checkpointing_strategy
        self.checkpointing_execution = checkpointing_execution

    def save_checkpoint(
        self,
        global_train_step: int,
        model: nn.Module,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
        evaluation_result: Optional[Dict[str, EvaluationResultBatch]] = None,
    ):
        checkpointing_instruction = self.checkpointing_strategy.get_checkpoint_instruction(
            global_train_step=global_train_step,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )

        self.checkpointing_execution.run_checkpoint_instruction(
            checkpointing_instruction=checkpointing_instruction,
            global_train_sample_id=global_train_step,
            model=model,
            optimizer=optimizer,
        )

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        model = self.checkpointing_execution.load_model_checkpoint(model=model, file_path=file_path)
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, wrapped_model: nn.Module, file_path: Path) -> Optimizer:
        optimizer = self.checkpointing_execution.load_optimizer_checkpoint(
            optimizer=optimizer,
            wrapped_model=wrapped_model,
            file_path=file_path,
        )
        return optimizer
