from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from llm_gym.batch import EvaluationResultBatch


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
    def get_model_checkpoint_instruction(self, current_epoch: int, num_epochs: int,
                                         evaluation_result: EvaluationResultBatch,
                                         early_stoppping_criterion_fulfilled: bool = False) -> CheckpointingInstruction:
        raise NotImplementedError


class CheckpointingExecutionIF(ABC):
    @abstractmethod
    def run_checkpoint_instructions(self, checkpointing_instruction: CheckpointingInstruction, current_epoch: int):
        raise NotImplementedError


class Checkpointing(CheckpointingStrategyIF, CheckpointingExecutionIF):
    """
    Checkpoint class to get checkpoint instruction.
    """

    def __init__(self, checkpointing_strategy: CheckpointingStrategyIF, checkpointing_execution: CheckpointingExecutionIF):
        self.checkpointing_strategy = checkpointing_strategy
        self.checkpointing_execution = checkpointing_execution

    def run(self, current_epoch: int, num_epochs: int, evaluation_result: EvaluationResultBatch,
            early_stoppping_criterion_fulfilled: bool = False):
        checkpointing_instruction = self.get_model_checkpoint_instruction(current_epoch=current_epoch, num_epochs=num_epochs,
                                                                          evaluation_result=evaluation_result,
                                                                          early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled)
        self.run_checkpoint_instructions(checkpointing_instruction)

    def get_model_checkpoint_instruction(self, current_epoch: int, num_epochs: int,
                                         evaluation_result: EvaluationResultBatch,
                                         early_stoppping_criterion_fulfilled: bool = False) -> CheckpointingInstruction:
        return self.checkpointing_strategy.get_model_checkpoint_instruction(current_epoch=current_epoch, num_epochs=num_epochs,
                                                                            evaluation_result=evaluation_result,
                                                                            early_stoppping_criterion_fulfilled=early_stoppping_criterion_fulfilled)

    def run_checkpoint_instructions(self, checkpointing_instruction: CheckpointingInstruction, current_epoch: int):
        self.checkpointing_execution.run_checkpoint_instructions(checkpointing_instruction=checkpointing_instruction,
                                                                 current_epoch=current_epoch)
