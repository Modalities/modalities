import torch
from torch.utils.data import BatchSampler, DistributedSampler
from transformers import GPT2TokenizerFast

from modalities.checkpointing.checkpointing import Checkpointing
from modalities.checkpointing.checkpointing_factory import CheckpointingExecutionFactory
from modalities.checkpointing.checkpointing_strategies import (
    SaveEveryKStepsCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.config.config_new import (
    CheckpointingConfig,
    CLMCrossEntropyLossConfig,
    FSDPToDiscCheckpointingConfig,
    SaveEveryKStepsCheckpointingStrategyConfig,
    SaveKMostRecentCheckpointsStrategyConfig,
)
from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from modalities.dataloader.dataset import MemMapDataset, PackedMemMapDatasetContinuous, PackedMemMapDatasetMegatron
from modalities.dataloader.open_gptx_dataset.mmap_dataset import MMapIndexedDatasetBuilder
from modalities.dataloader.open_gptx_dataset.open_gptx_dataset import OpenGPTXMMapDataset
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.gpt2.collator import GPT2LLMCollator
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.huggingface.huggingface_models import HuggingFacePretrainedModel
from modalities.registry.registry import Registry
from modalities.running_env.fsdp.fsdp_running_env import FSDPRunningEnv, FSDPRunningEnvConfig


class RegistryFactory:
    @staticmethod
    def get_component_registry() -> Registry:
        components = [
            # models
            ("model", "gpt2", GPT2LLM),
            ("model", "huggingface_pretrained_model", HuggingFacePretrainedModel),
            # losses
            ("loss", "clm_cross_entropy_loss", CLMCrossEntropyLoss),
            # optmizers
            ("optimizer", "adamw", torch.optim.AdamW),
            # schedulers
            ("scheduler", "step_lr", torch.optim.lr_scheduler.StepLR),
            ("scheduler", "constant_lr", torch.optim.lr_scheduler.ConstantLR),
            ("scheduler", "onecycle_lr", torch.optim.lr_scheduler.OneCycleLR),
            # tokenizers
            ("tokenizer", "gpt2_tokenizer_fast", GPT2TokenizerFast),
            ("tokenizer", "llama_tokenizer_fast", GPT2TokenizerFast),
            # dataset types
            ("dataset", "mem_map_dataset", MemMapDataset),
            ("dataset", "packed_mem_map_dataset_continuous", PackedMemMapDatasetContinuous),
            ("dataset", "packed_mem_map_dataset_megatron", PackedMemMapDatasetMegatron),
            ("dataset", "mmap_indexed_dataset", MMapIndexedDatasetBuilder),
            ("dataset", "open_gptx_mmap_dataset", OpenGPTXMMapDataset),
            # samplers
            ("sampler", "distributed_sampler", DistributedSampler),
            # batch samplers
            ("batch_sampler", "batch_sampler", BatchSampler),
            # collators
            ("collator", "gpt2_llm_collator", GPT2LLMCollator),
            # data loaders
            ("data_loader", "llm_data_loader", LLMDataLoader),
            ("data_loader", "repeating_data_loader", RepeatingDataLoader),
            # checkpointing
            ("checkpointing", "default", Checkpointing),
            # checkpointing strategies
            (
                "checkpointing_strategy",
                "save_every_k_steps_checkpointing_strategy",
                SaveEveryKStepsCheckpointingStrategy,
            ),
            ("checkpointing_strategy", "save_k_most_recent_checkpoints_strategy", SaveKMostRecentCheckpointsStrategy),
            # checkpointing execution
            (
                "checkpointing_execution",
                "fsdp_to_disc_checkpointing",
                CheckpointingExecutionFactory.get_fsdp_to_disc_checkpointing,
            ),
            # running env
            ("running_env", "fsdp_running_env", FSDPRunningEnv),
        ]
        registry = Registry()
        for component in components:
            registry.add_entity(*component)
        return registry

    @staticmethod
    def get_config_registry() -> Registry:
        components = [
            # losses
            ("loss", "clm_cross_entropy_loss", CLMCrossEntropyLossConfig),
            # checkpointing
            ("checkpointing", "default", CheckpointingConfig),
            # checkpointing strategies
            (
                "checkpointing_strategy",
                "save_every_k_steps_checkpointing_strategy",
                SaveEveryKStepsCheckpointingStrategyConfig,
            ),
            (
                "checkpointing_strategy",
                "save_k_most_recent_checkpoints_strategy",
                SaveKMostRecentCheckpointsStrategyConfig,
            ),
            # checkpointing execution
            ("checkpointing_execution", "fsdp_to_disc_checkpointing", FSDPToDiscCheckpointingConfig),
            # running env
            ("running_env", "fsdp_running_env", FSDPRunningEnvConfig),
        ]
        registry = Registry()
        for component in components:
            registry.add_entity(*component)
        return registry
