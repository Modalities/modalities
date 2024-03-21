from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import BatchSampler, DistributedSampler
from transformers import GPT2TokenizerFast

from modalities.checkpointing.checkpointing import Checkpointing
from modalities.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from modalities.checkpointing.checkpointing_strategies import (
    SaveEveryKStepsCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.config.config import (
    AdamOptimizerConfig,
    AdamWOptimizerConfig,
    BatchSamplerConfig,
    CheckpointedModelConfig,
    CheckpointedOptimizerConfig,
    CheckpointingConfig,
    CLMCrossEntropyLossConfig,
    ConstantLRSchedulerConfig,
    CosineAnnealingLRSchedulerConfig,
    DistributedSamplerConfig,
    DummyLRSchedulerConfig,
    DummyProgressSubscriberConfig,
    DummyResultSubscriberConfig,
    FSDPToDiscCheckpointingConfig,
    FSDPWrappedModelConfig,
    GPT2LLMCollateFnConfig,
    GPT2TokenizerFastConfig,
    LLMDataLoaderConfig,
    MemMapDatasetConfig,
    OneCycleLRSchedulerConfig,
    OpenGPTXMMapDatasetConfig,
    PackedMemMapDatasetContinuousConfig,
    PackedMemMapDatasetMegatronConfig,
    RichProgressSubscriberConfig,
    RichResultSubscriberConfig,
    SaveEveryKStepsCheckpointingStrategyConfig,
    SaveKMostRecentCheckpointsStrategyConfig,
    StepLRSchedulerConfig,
    WandBEvaluationResultSubscriberConfig,
)
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.logging_broker.subscriber_impl.subscriber_factory import (
    ProgressSubscriberFactory,
    ResultsSubscriberFactory,
)
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.components.layer_norms import LayerNormConfig, RMSLayerNorm, RMSLayerNormConfig
from modalities.models.gpt2.collator import GPT2LLMCollateFn
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2LLMConfig
from modalities.models.huggingface.huggingface_models import (
    HuggingFacePretrainedModel,
    HuggingFacePretrainedModelConfig,
)
from modalities.models.model_factory import ModelFactory
from modalities.optimizers.lr_schedulers import DummyLRScheduler
from modalities.optimizers.optimizer_factory import OptimizerFactory


@dataclass
class ComponentEntity:
    component_key: str
    variant_key: str
    component_type: Type
    component_config_type: Type[BaseModel]


COMPONENTS = [
    # models
    ComponentEntity("model", "gpt2", GPT2LLM, GPT2LLMConfig),
    ComponentEntity(
        "model", "huggingface_pretrained_model", HuggingFacePretrainedModel, HuggingFacePretrainedModelConfig
    ),
    ComponentEntity("model", "checkpointed", ModelFactory.get_checkpointed_model, CheckpointedModelConfig),
    ComponentEntity("model", "fsdp_wrapped", ModelFactory.get_fsdp_wrapped_model, FSDPWrappedModelConfig),
    # losses
    ComponentEntity("loss", "clm_cross_entropy_loss", CLMCrossEntropyLoss, CLMCrossEntropyLossConfig),
    # optmizers
    ComponentEntity("optimizer", "adam", OptimizerFactory.get_adam, AdamOptimizerConfig),
    ComponentEntity("optimizer", "adam_w", OptimizerFactory.get_adam_w, AdamWOptimizerConfig),
    ComponentEntity(
        "optimizer", "checkpointed", OptimizerFactory.get_checkpointed_optimizer, CheckpointedOptimizerConfig
    ),
    # schedulers
    ComponentEntity("scheduler", "dummy_lr", DummyLRScheduler, DummyLRSchedulerConfig),
    ComponentEntity("scheduler", "step_lr", torch.optim.lr_scheduler.StepLR, StepLRSchedulerConfig),
    ComponentEntity("scheduler", "constant_lr", torch.optim.lr_scheduler.ConstantLR, ConstantLRSchedulerConfig),
    ComponentEntity("scheduler", "onecycle_lr", torch.optim.lr_scheduler.OneCycleLR, OneCycleLRSchedulerConfig),
    ComponentEntity(
        "scheduler", "cosine_annealing_lr", torch.optim.lr_scheduler.CosineAnnealingLR, CosineAnnealingLRSchedulerConfig
    ),
    # tokenizers
    ComponentEntity("tokenizer", "gpt2_tokenizer_fast", GPT2TokenizerFast, GPT2TokenizerFastConfig),
    # ComponentEntity("tokenizer", "llama_tokenizer_fast", GPT2TokenizerFast, None),  # TODO
    # datasets
    ComponentEntity("dataset", "mem_map_dataset", DatasetFactory.get_mem_map_dataset, MemMapDatasetConfig),
    ComponentEntity(
        "dataset",
        "packed_mem_map_dataset_continuous",
        DatasetFactory.get_packed_mem_map_dataset_continuous,
        PackedMemMapDatasetContinuousConfig,
    ),
    ComponentEntity(
        "dataset",
        "packed_mem_map_dataset_megatron",
        DatasetFactory.get_packed_mem_map_dataset_megatron,
        PackedMemMapDatasetMegatronConfig,
    ),
    ComponentEntity(
        "dataset", "open_gptx_mmap_dataset", DatasetFactory.get_open_gptx_mmap_dataset, OpenGPTXMMapDatasetConfig
    ),
    # samplers
    ComponentEntity("sampler", "distributed_sampler", DistributedSampler, DistributedSamplerConfig),
    # batch samplers
    ComponentEntity("batch_sampler", "default", BatchSampler, BatchSamplerConfig),
    # collators
    ComponentEntity("collate_fn", "gpt_2_llm_collator", GPT2LLMCollateFn, GPT2LLMCollateFnConfig),
    # data loaders
    ComponentEntity("data_loader", "default", DataloaderFactory.get_dataloader, LLMDataLoaderConfig),
    # ComponentEntity("data_loader", "repeating_data_loader",(RepeatingDataLoader, None), # TODO
    # checkpointing
    ComponentEntity("checkpointing", "default", Checkpointing, CheckpointingConfig),
    # checkpointing strategies
    ComponentEntity(
        "checkpointing_strategy",
        "save_every_k_steps_checkpointing_strategy",
        SaveEveryKStepsCheckpointingStrategy,
        SaveEveryKStepsCheckpointingStrategyConfig,
    ),
    ComponentEntity(
        "checkpointing_strategy",
        "save_k_most_recent_checkpoints_strategy",
        SaveKMostRecentCheckpointsStrategy,
        SaveKMostRecentCheckpointsStrategyConfig,
    ),
    # checkpointing execution
    ComponentEntity(
        "checkpointing_execution", "fsdp_to_disc_checkpointing", FSDPToDiscCheckpointing, FSDPToDiscCheckpointingConfig
    ),
    # Progress subscriber
    ComponentEntity(
        "progress_subscriber",
        "dummy",
        ProgressSubscriberFactory.get_dummy_progress_subscriber,
        DummyProgressSubscriberConfig,
    ),
    ComponentEntity(
        "progress_subscriber",
        "rich",
        ProgressSubscriberFactory.get_rich_progress_subscriber,
        RichProgressSubscriberConfig,
    ),
    # Results subscriber
    ComponentEntity(
        "results_subscriber", "dummy", ResultsSubscriberFactory.get_dummy_result_subscriber, DummyResultSubscriberConfig
    ),
    ComponentEntity(
        "results_subscriber", "rich", ResultsSubscriberFactory.get_rich_result_subscriber, RichResultSubscriberConfig
    ),
    ComponentEntity(
        "results_subscriber",
        "wandb",
        ResultsSubscriberFactory.get_wandb_result_subscriber,
        WandBEvaluationResultSubscriberConfig,
    ),
    # layer norms
    ComponentEntity("layer_norm", "rms_norm", RMSLayerNorm, RMSLayerNormConfig),
    ComponentEntity("layer_norm", "layer_norm", nn.LayerNorm, LayerNormConfig),
]
