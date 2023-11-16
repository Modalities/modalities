from typing import List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from llm_gym.batch import EvaluationResultBatch
from llm_gym.models.model import NNModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
import torch.nn as nn
import torch.distributed as dist


@dataclass
class CheckpointingInstruction:
    """
    Instruction to save and delete checkpoints.
    """

    save_current: bool = False
    checkpoints_to_delete: List[int] = field(default_factory=list)


class CheckpointingStrategyIF(ABC):
    """
    Checkpoint Interface to get checkpoint instruction.
    """

    @abstractmethod
    def get_model_checkpoint_instruction(
        self,
        global_train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        early_stoppping_criterion_fulfilled: bool = False,
    ) -> CheckpointingInstruction:
        raise NotImplementedError


class CheckpointingExecutionIF(ABC):
    @abstractmethod
    def run_checkpoint_instructions(
        self,
        checkpointing_instruction: CheckpointingInstruction,
        global_train_batch_id: int,
        model: NNModel,
        optimizer: Optimizer,
    ):
        raise NotImplementedError

    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, global_train_batch_id: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: FSDP, global_train_batch_id: int) -> Optimizer:
        raise NotImplementedError


class CheckpointingIF:
    def save_checkpoint(
        self,
        train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        model: NNModel,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def load_model_checkpoint(self, model: nn.Module, global_train_batch_id: int) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: FSDP, global_train_batch_id: int) -> Optimizer:
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
        train_batch_id: int,
        num_batches: int,
        evaluation_result: EvaluationResultBatch,
        model: NNModel,
        optimizer: Optimizer,
        early_stoppping_criterion_fulfilled: bool = False,
    ):
        global_train_batch_id = (train_batch_id + 1) * self.num_ranks - 1
        checkpointing_instruction = self.checkpointing_strategy.get_model_checkpoint_instruction(
            global_train_batch_id=global_train_batch_id,
            num_batches=num_batches,
            evaluation_result=evaluation_result,
            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled,
        )
        self.checkpointing_execution.run_checkpoint_instructions(
            checkpointing_instruction=checkpointing_instruction,
            global_train_batch_id=global_train_batch_id,
            model=model,
            optimizer=optimizer,
        )

    def load_model_checkpoint(self, model: nn.Module, global_train_batch_id: int) -> nn.Module:
        model = self.checkpointing_execution.load_model_checkpoint(
            model=model, global_train_batch_id=global_train_batch_id
        )
        return model

    def load_optimizer_checkpoint(self, optimizer: Optimizer, model: FSDP, global_train_batch_id: int) -> Optimizer:
        optimizer = self.checkpointing_execution.load_optimizer_checkpoint(
            optimizer=optimizer, model=model, global_train_batch_id=global_train_batch_id
        )
        return optimizer
