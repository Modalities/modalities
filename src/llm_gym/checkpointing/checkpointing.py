from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch.nn as nn
from torch.optim import Optimizer

from llm_gym.batch import EvaluationResultBatch
from llm_gym.checkpointing.checkpointing_instruction import CheckpointingInstruction


class CheckpointingStrategyIF(ABC):
    """
    Checkpoint Interface to get checkpoint instruction.
    """

    @abstractmethod
    def get_checkpoint_instruction(
        self,
        global_train_sample_id: int,
        evaluation_result: Dict[str, EvaluationResultBatch] | None = None,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        raise NotImplementedError


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


class CheckpointingExecutionIF(CheckpointingIF):
    @abstractmethod
    def run_checkpoint_instructions(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_sample_id: int,
        model: nn.Module,
        optimizer: Optimizer,
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
        num_ranks: int,
    ):
        self.checkpointing_strategy = checkpointing_strategy
        self.checkpointing_execution = checkpointing_execution
        self.num_ranks = num_ranks

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

        self.checkpointing_execution.run_checkpoint_instructions(
            checkpointing_instruction=checkpointing_instruction,
            global_train_sample_id=global_train_sample_id,
            model=model,
            optimizer=optimizer,
        )

    def load_model_checkpoint(self, model: nn.Module, file_path: Path) -> nn.Module:
        model = self.checkpointing_execution.load_model_checkpoint(model=model, file_path=file_path)
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: nn.Module, file_path: Path) -> Optimizer:
        optimizer = self.checkpointing_execution.load_optimizer_checkpoint(
            optimizer=optimizer,
            model=model,
            file_path=file_path,
        )
        return optimizer
