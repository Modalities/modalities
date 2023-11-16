from enum import Enum

import torch

from llm_gym.loss_functions import CLMCrossEntropyLoss
from llm_gym.models.gpt2.gpt2_model import GPT2LLM


class LookupEnum(Enum):
    @classmethod
    def _missing_(cls, value: str) -> type:
        """constructs Enum by member name, if not constructable by value"""
        return cls.__dict__[value]


class ModelTypes(LookupEnum):
    GPT2LLM = GPT2LLM


class LossTypes(LookupEnum):
    CLMCrossEntropyLoss = CLMCrossEntropyLoss


class OptimizerTypes(LookupEnum):
    AdamW = torch.optim.AdamW


class SchedulerTypes(LookupEnum):
    StepLR = torch.optim.lr_scheduler.StepLR
    ConstantLR = torch.optim.lr_scheduler.ConstantLR
    OneCycleLR = torch.optim.lr_scheduler.OneCycleLR
