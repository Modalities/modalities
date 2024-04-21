import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import torch.nn as nn
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, FilePath, GetCoreSchemaHandler, PositiveInt, field_validator, model_validator
from pydantic_core import core_schema
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
from transformers import GPT2TokenizerFast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast

from modalities.checkpointing.checkpointing import Checkpointing
from modalities.checkpointing.checkpointing_execution import CheckpointingExecutionIF
from modalities.checkpointing.checkpointing_strategies import CheckpointingStrategyIF
from modalities.config.lookup_enum import LookupEnum
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.loss_functions import Loss
from modalities.models.gpt2.collator import CollateFnIF
from modalities.running_env.env_utils import MixedPrecisionSettings, has_bfloat_support
from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper
from modalities.util import get_date_of_run, parse_enum_by_name


class PydanticThirdPartyTypeIF:
    def __init__(self, third_party_type):
        self.third_party_type = third_party_type

    def __get_pydantic_core_schema__(
        self,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(self.third_party_type),
            python_schema=core_schema.is_instance_schema(self.third_party_type),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


PydanticCheckpointingType = Annotated[Checkpointing, PydanticThirdPartyTypeIF(Checkpointing)]
PydanticCheckpointingStrategyIFType = Annotated[
    CheckpointingStrategyIF, PydanticThirdPartyTypeIF(CheckpointingStrategyIF)
]
PydanticCheckpointingExecutionIFType = Annotated[
    CheckpointingExecutionIF, PydanticThirdPartyTypeIF(CheckpointingExecutionIF)
]
PydanticPytorchModuleType = Annotated[nn.Module, PydanticThirdPartyTypeIF(nn.Module)]
PydanticTokenizerIFType = Annotated[TokenizerWrapper, PydanticThirdPartyTypeIF(TokenizerWrapper)]
PydanticDatasetIFType = Annotated[Dataset, PydanticThirdPartyTypeIF(Dataset)]
PydanticSamplerIFType = Annotated[Sampler, PydanticThirdPartyTypeIF(Sampler)]
PydanticCollateFnIFType = Annotated[CollateFnIF, PydanticThirdPartyTypeIF(CollateFnIF)]
PydanticLLMDataLoaderIFType = Annotated[LLMDataLoader, PydanticThirdPartyTypeIF(LLMDataLoader)]
PydanticOptimizerIFType = Annotated[Optimizer, PydanticThirdPartyTypeIF(Optimizer)]
PydanticLRSchedulerIFType = Annotated[LRScheduler, PydanticThirdPartyTypeIF(LRScheduler)]
PydanticLossIFType = Annotated[Loss, PydanticThirdPartyTypeIF(Loss)]
PydanticMessageSubscriberIFType = Annotated[MessageSubscriberIF, PydanticThirdPartyTypeIF(MessageSubscriberIF)]


class ProcessGroupBackendType(LookupEnum):
    nccl = "nccl"


class TokenizerTypes(LookupEnum):
    GPT2TokenizerFast = GPT2TokenizerFast
    LlamaTokenizerFast = LlamaTokenizerFast


class PassType(LookupEnum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


class WandbMode(LookupEnum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DISABLED = "DISABLED"


class GradientClippingMode(LookupEnum):
    NONE = "NONE"  # Do not apply gradient clipping.
    VALUE = "value"  # Clip all gradient values independently.
    # For norm based clipping modes, the norm is computed over
    # all gradients together, as if they were concatenated
    # into a single vector.
    P1_NORM = "p1_norm"  # manhattan norm based clipping.
    P2_NORM = "p2_norm"  # Euclidean norm based clipping.
    MAX_NORM = "max_norm"  # Maximum norm based clipping.


class ReferenceConfig(BaseModel):
    instance_key: str
    pass_type: PassType


class CLMCrossEntropyLossConfig(BaseModel):
    target_key: str
    prediction_key: str


# Checkpointing
class SaveEveryKStepsCheckpointingStrategyConfig(BaseModel):
    k: PositiveInt


class SaveKMostRecentCheckpointsStrategyConfig(BaseModel):
    k: Annotated[int, Field(strict=True, ge=-1)]


class FSDPToDiscCheckpointingConfig(BaseModel):
    checkpoint_path: Path
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    experiment_id: str
    block_names: List[str]
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy

    @field_validator("mixed_precision_settings", mode="before")
    def parse_mixed_precision_setting_by_name(cls, name):
        mixed_precision_settings: MixedPrecisionSettings = parse_enum_by_name(
            name=name, enum_type=MixedPrecisionSettings
        )
        if not has_bfloat_support() and (
            mixed_precision_settings == MixedPrecisionSettings.BF_16
            or mixed_precision_settings == MixedPrecisionSettings.BF_16_WORKING
        ):
            raise ValueError("BF16 not supported in the current environment")
        return mixed_precision_settings

    @field_validator("sharding_strategy", mode="before")
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=ShardingStrategy)


class CheckpointingConfig(BaseModel):
    checkpointing_strategy: PydanticCheckpointingStrategyIFType
    checkpointing_execution: PydanticCheckpointingExecutionIFType


class AdamOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


class AdamWOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


class DummyLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType


class StepLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    step_size: Annotated[int, Field(strict=True, gt=0)]
    gamma: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class OneCycleLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    max_lr: Annotated[float, Field(strict=True, gt=0.0)] | List[Annotated[float, Field(strict=True, gt=0.0)]]
    total_steps: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    epochs: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    steps_per_epoch: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    pct_start: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    anneal_strategy: str
    cycle_momentum: bool = True
    base_momentum: Annotated[float, Field(strict=True, gt=0)] | List[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.85
    max_momentum: Annotated[float, Field(strict=True, gt=0.0)] | List[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.95
    div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    final_div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    three_phase: bool = False
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False

    @model_validator(mode="after")
    def check_totals_steps_and_epchs(self) -> "OneCycleLRSchedulerConfig":
        if self.total_steps is None and (self.epochs is None or self.steps_per_epoch is None):
            raise ValueError("Please define total_steps or (epochs and steps_per_epoch).")
        return self


class ConstantLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    factor: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    total_iters: Annotated[int, Field(strict=True, gt=0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class CosineAnnealingLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    t_max: Annotated[int, Field(strict=True, gt=0)]
    eta_min: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1
    verbose: bool = False


class CheckpointedOptimizerConfig(BaseModel):
    checkpointing: PydanticCheckpointingType
    checkpoint_path: Path
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType


class CheckpointedModelConfig(BaseModel):
    checkpointing: PydanticCheckpointingType
    checkpoint_path: Path
    model: PydanticPytorchModuleType


class FSDPWrappedModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    sync_module_states: bool
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy
    block_names: List[str]

    @field_validator("mixed_precision_settings", mode="before")
    def parse_mixed_precision_setting_by_name(cls, name):
        mixed_precision_settings: MixedPrecisionSettings = parse_enum_by_name(
            name=name, enum_type=MixedPrecisionSettings
        )
        if not has_bfloat_support() and (
            mixed_precision_settings == MixedPrecisionSettings.BF_16
            or mixed_precision_settings == MixedPrecisionSettings.BF_16_WORKING
        ):
            raise ValueError("BF16 not supported in the current environment")
        return mixed_precision_settings

    @field_validator("sharding_strategy", mode="before")
    def parse_sharding_strategy_by_name(cls, name):
        return parse_enum_by_name(name=name, enum_type=ShardingStrategy)


class PreTrainedHFTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    max_length: Annotated[int, Field(strict=True, ge=0)]
    truncation: bool = False
    padding: bool | str = False


class PreTrainedSPTokenizerConfig(BaseModel):
    tokenizer_model_file: str


class DistributedSamplerConfig(BaseModel):
    rank: Annotated[int, Field(strict=True, ge=0)]
    num_replicas: Annotated[int, Field(strict=True, ge=0)]
    shuffle: bool
    dataset: PydanticDatasetIFType
    seed: Optional[int] = 0


class MemMapDatasetConfig(BaseModel):
    raw_data_path: FilePath
    index_path: Optional[FilePath] = None
    block_size: Annotated[int, Field(strict=True, gt=0)]
    tokenizer: PydanticTokenizerIFType
    jq_pattern: str
    sample_key: str


class PackedMemMapDatasetContinuousConfig(BaseModel):
    raw_data_path: Path
    block_size: Annotated[int, Field(strict=True, gt=0)]
    sample_key: str


class PackedMemMapDatasetMegatronConfig(BaseModel):
    raw_data_path: Path
    block_size: Annotated[int, Field(strict=True, gt=0)]
    sample_key: str


class MMapIndexedDatasetConfig(BaseModel):
    path: Path
    skip_warmup: bool


class OpenGPTXMMapDatasetConfig(BaseModel):
    num_samples: Annotated[int, Field(strict=True, ge=1)]
    path: FilePath
    sample_key: str
    sequence_len: PositiveInt


class BatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIFType
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    drop_last: Literal[True] = True


class ResumableBatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIFType
    start_index: Annotated[int, Field(strict=True, gt=0)]


class GPT2LLMCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str


class LLMDataLoaderConfig(BaseModel):
    dataloader_tag: str
    dataset: PydanticDatasetIFType
    batch_sampler: PydanticSamplerIFType
    collate_fn: Optional[PydanticCollateFnIFType] = None
    num_workers: Annotated[int, Field(strict=True, ge=0)]
    pin_memory: bool
    shuffle: bool
    skip_num_steps: Optional[int] = 0


class RepeatingDataLoaderConfig(BaseModel):
    dataloader: PydanticLLMDataLoaderIFType
    reshuffle_after_epoch: Optional[bool] = False
    num_epochs: Annotated[int, Field(strict=True, ge=1)]


class DummyProgressSubscriberConfig(BaseModel):
    pass


class RichProgressSubscriberConfig(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: Optional[List[PydanticLLMDataLoaderIFType]] = Field(default_factory=list)
    global_num_seen_steps: int
    local_rank: int


class DummyResultSubscriberConfig(BaseModel):
    pass


class WandBEvaluationResultSubscriberConfig(BaseModel):
    local_rank: int
    project: str
    experiment_id: str
    mode: WandbMode
    directory: Path
    experiment_config: Optional[Dict] = None


class RichResultSubscriberConfig(BaseModel):
    num_ranks: int
    local_rank: int


class CudaEnvConfig(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    world_size: Annotated[int, Field(strict=True, ge=1)]
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class PackedDatasetSettings(BaseModel):
    src_path: FilePath
    dst_path: Optional[Path] = None
    index_path: Optional[FilePath] = None
    jq_pattern: str
    num_cpus: Annotated[int, Field(strict=True, ge=1)] = os.cpu_count()
    eod_token: str


class TrainingSettings(BaseModel):
    class Training(BaseModel):
        class GradientClipping(BaseModel):
            mode: GradientClippingMode
            threshold: Optional[Annotated[float, Field(strict=True, gt=0.0)]] = None

            @model_validator(mode="after")
            def check_mode_none_iff_threshold_none(self) -> BaseModel:
                if self.mode == GradientClippingMode.NONE and self.threshold is not None:
                    raise ValueError("If gradient clipping is deactivated, no threshold should be set.")
                if self.mode != GradientClippingMode.NONE and self.threshold is None:
                    raise ValueError("A threshold value is required when gradient clipping is used.")
                return self

        global_training_log_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
        global_checkpointing_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
        global_evaluation_interval_in_steps: Annotated[int, Field(strict=True, ge=1)]
        do_apply_activation_checkpointing: bool
        gradient_acc_steps: Annotated[int, Field(strict=True, ge=1)]
        local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
        sequence_length: Annotated[int, Field(strict=True, ge=1)]
        gradient_clipping: GradientClipping

    class Paths(BaseModel):
        checkpointing_path: Path

    experiment_id: str
    referencing_keys: Dict[str, str]
    training: Training
    cuda_env: CudaEnvConfig
    paths: Paths


class TrainingComponentsInstantiationModel(BaseModel):
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType
    scheduler: PydanticLRSchedulerIFType
    loss_fn: PydanticLossIFType
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: List[PydanticLLMDataLoaderIFType]
    batch_progress_subscriber: PydanticMessageSubscriberIFType
    evaluation_subscriber: PydanticMessageSubscriberIFType
    checkpointing: PydanticCheckpointingType
    settings: TrainingSettings


class PackedDatasetComponentsModel(BaseModel):
    tokenizer: PydanticTokenizerIFType
    settings: PackedDatasetSettings


class ComponentsInferenceModel(BaseModel):
    wrapped_model: PydanticPytorchModuleType
    cuda_env: CudaEnvConfig


def load_app_config_dict(config_file_path: Path) -> Dict:
    def cuda_env_resolver_fun(var_name: str) -> int:
        int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    def modalities_env_resolver_fun(var_name: str) -> int:
        if var_name == "experiment_id":
            return get_date_of_run()

    def node_env_resolver_fun(var_name: str) -> int:
        if var_name == "num_cpus":
            return os.cpu_count()

    OmegaConf.register_new_resolver("cuda_env", cuda_env_resolver_fun, replace=True)
    OmegaConf.register_new_resolver("modalities_env", modalities_env_resolver_fun, replace=True)
    OmegaConf.register_new_resolver("node_env", node_env_resolver_fun, replace=True)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    return config_dict
