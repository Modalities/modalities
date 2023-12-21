from enum import Enum, member

import torch
from torch.utils.data import DistributedSampler
from transformers import GPT2TokenizerFast

from llm_gym.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from llm_gym.dataloader.dataset import MemMapDataset, PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron
from llm_gym.dataloader.open_gptx_dataset.mmap_dataset import MMapIndexedDatasetBuilder
from llm_gym.dataloader.open_gptx_dataset.open_gptx_dataset import OpenGPTXDatasetFactory
from llm_gym.loss_functions import CLMCrossEntropyLoss
from llm_gym.models.gpt2.collator import GPT2LLMCollator
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


class TokenizerTypes(LookupEnum):
    GPT2TokenizerFast = GPT2TokenizerFast


class DatasetTypes(LookupEnum):
    MemMapDataset = MemMapDataset
    PackedMemMapDatasetContinuous = PackedMemMapDatasetContinuous
    PackedMemMapDatasetMegatron = PackedMemMapDatasetMegatron
    MMapIndexedDataset = MMapIndexedDatasetBuilder
    OpenGPTXMMapDataset = member(OpenGPTXDatasetFactory.create_dataset)


class SamplerTypes(LookupEnum):
    DistributedSampler = DistributedSampler


class CollatorTypes(LookupEnum):
    GPT2LLMCollator = GPT2LLMCollator


class DataloaderTypes(LookupEnum):
    RepeatingDataLoader = RepeatingDataLoader
    LLMDataLoader = LLMDataLoader
