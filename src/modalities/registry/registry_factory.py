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
    AdamWOptimizerConfig,
    BatchSamplerConfig,
    CheckpointingConfig,
    CLMCrossEntropyLossConfig,
    DistributedSamplerConfig,
    FSDPToDiscCheckpointingConfig,
    GPT2LLMCollateFnConfig,
    GPT2TokenizerFastConfig,
    LLMDataLoaderConfig,
    MemMapDatasetConfig,
    OpenGPTXMMapDatasetConfig,
    PackedMemMapDatasetContinuousConfig,
    PackedMemMapDatasetMegatronConfig,
    ResumableBatchSamplerConfig,
    SaveEveryKStepsCheckpointingStrategyConfig,
    SaveKMostRecentCheckpointsStrategyConfig,
)
from modalities.dataloader.dataloader import LLMDataLoader, RepeatingDataLoader
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.dataloader.samplers import ResumableBatchSampler
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.gpt2.collator import GPT2LLMCollateFn
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.huggingface.huggingface_models import HuggingFacePretrainedModel
from modalities.optimizers.optimizer_factory import OptimizerFactory
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
            ("optimizer", "adam_w", OptimizerFactory.get_adam_w),
            # schedulers
            ("scheduler", "step_lr", torch.optim.lr_scheduler.StepLR),
            ("scheduler", "constant_lr", torch.optim.lr_scheduler.ConstantLR),
            ("scheduler", "onecycle_lr", torch.optim.lr_scheduler.OneCycleLR),
            # tokenizers
            ("tokenizer", "gpt2_tokenizer_fast", GPT2TokenizerFast),
            ("tokenizer", "llama_tokenizer_fast", GPT2TokenizerFast),
            # datasets
            ("dataset", "mem_map_dataset", DatasetFactory.get_mem_map_dataset),
            ("dataset", "packed_mem_map_dataset_continuous", DatasetFactory.get_packed_mem_map_dataset_continuous),
            ("dataset", "packed_mem_map_dataset_megatron", DatasetFactory.get_packed_mem_map_dataset_megatron),
            ("dataset", "open_gptx_mmap_dataset", DatasetFactory.get_open_gptx_mmap_dataset),
            # samplers
            ("sampler", "distributed_sampler", DistributedSampler),
            # batch samplers
            ("batch_sampler", "default", BatchSampler),
            ("batch_sampler", "resumable_batch_sampler", ResumableBatchSampler),
            # collators
            ("collate_fn", "gpt_2_llm_collator", GPT2LLMCollateFn),
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
            # models
            ("model", "gpt2", GPT2LLMConfig),
            # losses
            ("loss", "clm_cross_entropy_loss", CLMCrossEntropyLossConfig),
            # optmizers
            ("optimizer", "adam_w", AdamWOptimizerConfig),
            # tokenizers
            ("tokenizer", "gpt2_tokenizer_fast", GPT2TokenizerFastConfig),
            # datasets
            ("dataset", "mem_map_dataset", MemMapDatasetConfig),
            ("dataset", "packed_mem_map_dataset_continuous", PackedMemMapDatasetContinuousConfig),
            ("dataset", "packed_mem_map_dataset_megatron", PackedMemMapDatasetMegatronConfig),
            ("dataset", "open_gptx_mmap_dataset", OpenGPTXMMapDatasetConfig),
            # samplers
            ("sampler", "distributed_sampler", DistributedSamplerConfig),
            # batch samplers
            ("batch_sampler", "default", BatchSamplerConfig),
            ("batch_sampler", "resumable_batch_sampler", ResumableBatchSamplerConfig),
            # collators
            ("collate_fn", "gpt_2_llm_collator", GPT2LLMCollateFnConfig),
            # data loaders
            ("data_loader", "llm_data_loader", LLMDataLoaderConfig),
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
