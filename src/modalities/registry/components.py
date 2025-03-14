from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.utils.data import BatchSampler, DistributedSampler, SequentialSampler

from modalities.checkpointing.checkpoint_saving import CheckpointSaving
from modalities.checkpointing.checkpoint_saving_strategies import (
    SaveEveryKStepsCheckpointingStrategy,
    SaveKMostRecentCheckpointsStrategy,
)
from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading, FSDP1CheckpointLoading
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import DCPCheckpointSaving, FSDP1CheckpointSaving
from modalities.checkpointing.stateful.app_state_factory import AppStateFactory
from modalities.checkpointing.torch.torch_checkpoint_loading import TorchCheckpointLoading
from modalities.config.config import (
    ActivationCheckpointedModelConfig,
    AdamOptimizerConfig,
    AdamWOptimizerConfig,
    BatchSamplerConfig,
    CheckpointedOptimizerConfig,
    CheckpointSavingConfig,
    CLMCrossEntropyLossConfig,
    CombinedDatasetConfig,
    CompiledModelConfig,
    ConstantLRSchedulerConfig,
    CosineAnnealingLRSchedulerConfig,
    DCPAppStateConfig,
    DCPCheckpointLoadingConfig,
    DCPCheckpointSavingConfig,
    DistributedSamplerConfig,
    DummyLRSchedulerConfig,
    DummyProgressSubscriberConfig,
    DummyResultSubscriberConfig,
    FSDP1CheckpointedModelConfig,
    FSDP1CheckpointLoadingConfig,
    FSDP1CheckpointSavingConfig,
    FSDP2WrappedModelConfig,
    FSDPWrappedModelConfig,
    GPT2LLMCollateFnConfig,
    LinearLRSchedulerConfig,
    LLMDataLoaderConfig,
    MemMapDatasetConfig,
    OneCycleLRSchedulerConfig,
    PackedMemMapDatasetContinuousConfig,
    PackedMemMapDatasetMegatronConfig,
    PreTrainedHFTokenizerConfig,
    PreTrainedSPTokenizerConfig,
    RawAppStateConfig,
    ResumableDistributedSamplerConfig,
    RichProgressSubscriberConfig,
    RichResultSubscriberConfig,
    SaveEveryKStepsCheckpointingStrategyConfig,
    SaveKMostRecentCheckpointsStrategyConfig,
    SequentialSamplerConfig,
    StepLRSchedulerConfig,
    TorchCheckpointLoadingConfig,
    WandBEvaluationResultSubscriberConfig,
    WeightInitializedModelConfig,
)
from modalities.dataloader.dataloader_factory import DataloaderFactory
from modalities.dataloader.dataset import DummyDatasetConfig
from modalities.dataloader.dataset_factory import DatasetFactory
from modalities.dataloader.samplers import ResumableDistributedSampler
from modalities.logging_broker.subscriber_impl.subscriber_factory import (
    ProgressSubscriberFactory,
    ResultsSubscriberFactory,
)
from modalities.loss_functions import CLMCrossEntropyLoss
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.models.coca.collator import CoCaCollateFnConfig, CoCaCollatorFn
from modalities.models.components.layer_norms import LayerNormConfig, RMSLayerNorm, RMSLayerNormConfig
from modalities.models.gpt2.collator import GPT2LLMCollateFn
from modalities.models.gpt2.gpt2_model import GPT2LLMConfig
from modalities.models.huggingface.huggingface_model import HuggingFacePretrainedModel, HuggingFacePretrainedModelConfig
from modalities.models.model_factory import GPT2ModelFactory, ModelFactory
from modalities.nn.model_initialization.composed_initialization import (
    ComposedInitializationRoutines,
    ComposedModelInitializationConfig,
)
from modalities.optimizers.lr_schedulers import DummyLRScheduler
from modalities.optimizers.optimizer_factory import OptimizerFactory
from modalities.running_env.fsdp.device_mesh import DeviceMeshConfig, get_device_mesh
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer, PreTrainedSPTokenizer
from modalities.training.gradient_clipping.fsdp_gradient_clipper import (
    DummyGradientClipper,
    FSDP1GradientClipper,
    FSDP1LoggingOnlyGradientClipper,
    FSDP2GradientClipper,
    FSDP2LoggingOnlyGradientClipper,
)
from modalities.training.gradient_clipping.fsdp_gradient_clipper_config import (
    DummyGradientClipperConfig,
    FSDPDummyGradientClipperConfig,
    FSDPGradientClipperConfig,
)
from modalities.utils.number_conversion import (
    LocalNumBatchesFromNumSamplesConfig,
    LocalNumBatchesFromNumTokensConfig,
    NumberConversion,
    NumberConversionFromCheckpointPathConfig,
    NumSamplesFromNumTokensConfig,
    NumStepsFromNumSamplesConfig,
    NumStepsFromNumTokensConfig,
    NumStepsFromRawDatasetIndexConfig,
    NumTokensFromNumStepsConfig,
    NumTokensFromPackedMemMapDatasetContinuousConfig,
)


@dataclass
class ComponentEntity:
    """Dataclass to store the component entity.
    The component entity stores the component key, the variant key, the component type and the component config type.
    The component key is used to identify the component type, whereas the variant key is used to identify the component.
    An example of a component entity is the GPT2 model with the component key "model" and the variant key "gpt2".

    Args:
        component_key (str): Key to identify the component type.
        variant_key (str): Variant key to identify the component.
        component_type (Type | Callable): Type of the component.
        component_config_type (Type[BaseModel]): Type of the component config.
    """

    component_key: str
    variant_key: str
    component_type: Type | Callable
    component_config_type: Type[BaseModel]


COMPONENTS = [
    # models
    ComponentEntity("model", "gpt2", GPT2ModelFactory.get_gpt2_model, GPT2LLMConfig),
    ComponentEntity(
        "model", "huggingface_pretrained_model", HuggingFacePretrainedModel, HuggingFacePretrainedModelConfig
    ),
    ComponentEntity(
        "model", "fsdp1_checkpointed", ModelFactory.get_fsdp1_checkpointed_model, FSDP1CheckpointedModelConfig
    ),
    ComponentEntity("model", "fsdp1_wrapped", ModelFactory.get_fsdp_wrapped_model, FSDPWrappedModelConfig),
    ComponentEntity("model", "fsdp2_wrapped", ModelFactory.get_fsdp_2_wrapped_model, FSDP2WrappedModelConfig),
    ComponentEntity(
        "model", "model_initialized", ModelFactory.get_weight_initalized_model, WeightInitializedModelConfig
    ),
    ComponentEntity(
        "model",
        "activation_checkpointed",
        ModelFactory.get_activation_checkpointed_model,
        ActivationCheckpointedModelConfig,
    ),
    ComponentEntity("model", "compiled", ModelFactory.get_compiled_model, CompiledModelConfig),
    ComponentEntity("model", "coca", CoCa, CoCaConfig),
    # Device mesh
    ComponentEntity("device_mesh", "default", get_device_mesh, DeviceMeshConfig),
    # weight initializers
    ComponentEntity(
        "model_initialization",
        "composed",
        ComposedInitializationRoutines.get_composed_model_initializer,
        ComposedModelInitializationConfig,
    ),
    # losses
    ComponentEntity("loss", "clm_cross_entropy_loss", CLMCrossEntropyLoss, CLMCrossEntropyLossConfig),
    # optmizers
    ComponentEntity("optimizer", "adam", OptimizerFactory.get_adam, AdamOptimizerConfig),
    ComponentEntity("optimizer", "adam_w", OptimizerFactory.get_adam_w, AdamWOptimizerConfig),
    ComponentEntity(
        "optimizer", "checkpointed", OptimizerFactory.get_checkpointed_optimizer_, CheckpointedOptimizerConfig
    ),
    # App state
    ComponentEntity("app_state", "raw", AppStateFactory.get_raw_app_state, RawAppStateConfig),
    ComponentEntity("app_state", "dcp", AppStateFactory.get_dcp_checkpointed_app_state, DCPAppStateConfig),
    # schedulers
    ComponentEntity("scheduler", "dummy_lr", DummyLRScheduler, DummyLRSchedulerConfig),
    ComponentEntity("scheduler", "step_lr", torch.optim.lr_scheduler.StepLR, StepLRSchedulerConfig),
    ComponentEntity("scheduler", "constant_lr", torch.optim.lr_scheduler.ConstantLR, ConstantLRSchedulerConfig),
    ComponentEntity("scheduler", "linear_lr", torch.optim.lr_scheduler.LinearLR, LinearLRSchedulerConfig),
    ComponentEntity("scheduler", "onecycle_lr", torch.optim.lr_scheduler.OneCycleLR, OneCycleLRSchedulerConfig),
    ComponentEntity(
        "scheduler", "cosine_annealing_lr", torch.optim.lr_scheduler.CosineAnnealingLR, CosineAnnealingLRSchedulerConfig
    ),
    # tokenizers
    ComponentEntity("tokenizer", "pretrained_hf_tokenizer", PreTrainedHFTokenizer, PreTrainedHFTokenizerConfig),
    ComponentEntity("tokenizer", "pretrained_sp_tokenizer", PreTrainedSPTokenizer, PreTrainedSPTokenizerConfig),
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
    ComponentEntity("dataset", "dummy_dataset", DatasetFactory.get_dummy_dataset, DummyDatasetConfig),
    ComponentEntity("dataset", "combined", DatasetFactory.get_combined_dataset, CombinedDatasetConfig),
    # samplers
    ComponentEntity("sampler", "sequential_sampler", SequentialSampler, SequentialSamplerConfig),
    ComponentEntity("sampler", "distributed_sampler", DistributedSampler, DistributedSamplerConfig),
    ComponentEntity(
        "sampler", "resumable_distributed_sampler", ResumableDistributedSampler, ResumableDistributedSamplerConfig
    ),
    # batch samplers
    ComponentEntity("batch_sampler", "default", BatchSampler, BatchSamplerConfig),
    # collators
    ComponentEntity("collate_fn", "gpt_2_llm_collator", GPT2LLMCollateFn, GPT2LLMCollateFnConfig),
    ComponentEntity("collate_fn", "coca_collator", CoCaCollatorFn, CoCaCollateFnConfig),
    # data loaders
    ComponentEntity("data_loader", "default", DataloaderFactory.get_dataloader, LLMDataLoaderConfig),
    # checkpointing
    ComponentEntity("checkpoint_saving", "default", CheckpointSaving, CheckpointSavingConfig),
    # checkpointing strategies
    ComponentEntity(
        "checkpoint_saving_strategy",
        "save_every_k_steps_checkpointing_strategy",
        SaveEveryKStepsCheckpointingStrategy,
        SaveEveryKStepsCheckpointingStrategyConfig,
    ),
    ComponentEntity(
        "checkpoint_saving_strategy",
        "save_k_most_recent_checkpoints_strategy",
        SaveKMostRecentCheckpointsStrategy,
        SaveKMostRecentCheckpointsStrategyConfig,
    ),
    # checkpoint saving execution
    ComponentEntity("checkpoint_saving_execution", "fsdp1", FSDP1CheckpointSaving, FSDP1CheckpointSavingConfig),
    ComponentEntity("checkpoint_saving_execution", "dcp", DCPCheckpointSaving, DCPCheckpointSavingConfig),
    # checkpoint loading
    ComponentEntity("checkpoint_loading", "fsdp1", FSDP1CheckpointLoading, FSDP1CheckpointLoadingConfig),
    ComponentEntity("checkpoint_loading", "dcp", DCPCheckpointLoading, DCPCheckpointLoadingConfig),
    ComponentEntity("checkpoint_loading", "torch", TorchCheckpointLoading, TorchCheckpointLoadingConfig),
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
    # gradient clippers
    ComponentEntity("gradient_clipper", "fsdp1", FSDP1GradientClipper, FSDPGradientClipperConfig),
    ComponentEntity(
        "gradient_clipper", "fsdp1_logging_only", FSDP1LoggingOnlyGradientClipper, FSDPDummyGradientClipperConfig
    ),
    ComponentEntity("gradient_clipper", "fsdp2", FSDP2GradientClipper, FSDPGradientClipperConfig),
    ComponentEntity(
        "gradient_clipper", "fsdp2_logging_only", FSDP2LoggingOnlyGradientClipper, FSDPDummyGradientClipperConfig
    ),
    ComponentEntity("gradient_clipper", "dummy", DummyGradientClipper, DummyGradientClipperConfig),
    # Number conversion
    ComponentEntity(
        "number_conversion",
        "local_num_batches_from_num_samples",
        NumberConversion.get_local_num_batches_from_num_samples,
        LocalNumBatchesFromNumSamplesConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "local_num_batches_from_num_tokens",
        NumberConversion.get_local_num_batches_from_num_tokens,
        LocalNumBatchesFromNumTokensConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_samples_from_num_tokens",
        NumberConversion.get_num_samples_from_num_tokens,
        NumSamplesFromNumTokensConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_steps_from_num_samples",
        NumberConversion.get_num_steps_from_num_samples,
        NumStepsFromNumSamplesConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_steps_from_num_tokens",
        NumberConversion.get_num_steps_from_num_tokens,
        NumStepsFromNumTokensConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_tokens_from_num_steps",
        NumberConversion.get_num_tokens_from_num_steps,
        NumTokensFromNumStepsConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "last_step_from_checkpoint_path",
        NumberConversion.get_last_step_from_checkpoint_path,
        NumberConversionFromCheckpointPathConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_seen_steps_from_checkpoint_path",
        NumberConversion.get_num_seen_steps_from_checkpoint_path,
        NumberConversionFromCheckpointPathConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "global_num_seen_tokens_from_checkpoint_path",
        NumberConversion.get_global_num_seen_tokens_from_checkpoint_path,
        NumberConversionFromCheckpointPathConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_target_steps_from_checkpoint_path",
        NumberConversion.get_num_target_steps_from_checkpoint_path,
        NumberConversionFromCheckpointPathConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "global_num_target_tokens_from_checkpoint_path",
        NumberConversion.get_global_num_target_tokens_from_checkpoint_path,
        NumberConversionFromCheckpointPathConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_tokens_from_packed_mem_map_dataset_continuous",
        NumberConversion.get_num_tokens_from_packed_mem_map_dataset_continuous,
        NumTokensFromPackedMemMapDatasetContinuousConfig,
    ),
    ComponentEntity(
        "number_conversion",
        "num_steps_from_raw_dataset_index",
        NumberConversion.get_num_steps_from_raw_dataset_index,
        NumStepsFromRawDatasetIndexConfig,
    ),
]
