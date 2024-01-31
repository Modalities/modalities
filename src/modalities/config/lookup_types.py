from enum import Enum

import torch
from torch.utils.data import BatchSampler, DistributedSampler
from transformers import GPT2TokenizerFast

from modalities.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from modalities.checkpointing.checkpointing_strategies import (
    SaveEveryKStepsCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.config.look_up_enum import LookupEnum
from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from modalities.dataloader.dataset import MemMapDataset, PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron
from modalities.dataloader.open_gptx_dataset.mmap_dataset import MMapIndexedDatasetBuilder
from modalities.dataloader.open_gptx_dataset.open_gptx_dataset import OpenGPTXMMapDataset
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.gpt2.collator import GPT2LLMCollator
from modalities.models.gpt2.gpt2_model import GPT2LLM


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
    # TODO: ClassResolver does not work with functions ... therefore there is also no
    # support for factories.
    OpenGPTXMMapDataset = OpenGPTXMMapDataset  # member(OpenGPTXDatasetFactory.create_dataset)


class SamplerTypes(LookupEnum):
    DistributedSampler = DistributedSampler


class BatchSamplerTypes(LookupEnum):
    BatchSampler = BatchSampler


class CollatorTypes(LookupEnum):
    GPT2LLMCollator = GPT2LLMCollator


class DataloaderTypes(LookupEnum):
    RepeatingDataLoader = RepeatingDataLoader
    LLMDataLoader = LLMDataLoader


class CheckpointingStrategyTypes(LookupEnum):
    SaveKMostRecentCheckpointsStrategy = SaveKMostRecentCheckpointsStrategy
    SaveEveryKStepsCheckpointingStrategy = SaveEveryKStepsCheckpointingStrategy


class CheckpointingExectionTypes(LookupEnum):
    FSDPToDiscCheckpointing = FSDPToDiscCheckpointing
