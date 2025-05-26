import os
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, FilePath, PositiveInt, field_validator, model_validator
from torch.distributed.fsdp import ShardingStrategy
from transformers import GPT2TokenizerFast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from typing_extensions import deprecated

from modalities.config.lookup_enum import LookupEnum
from modalities.config.pydantic_if_types import (
    PydanticAppStateType,
    PydanticCheckpointSavingExecutionIFType,
    PydanticCheckpointSavingStrategyIFType,
    PydanticCollateFnIFType,
    PydanticDatasetIFType,
    PydanticDeviceMeshIFType,
    PydanticFSDP1CheckpointLoadingIFType,
    PydanticFSDP1ModuleType,
    PydanticFSDP2ModuleType,
    PydanticLLMDataLoaderIFType,
    PydanticLRSchedulerIFType,
    PydanticModelInitializationIFType,
    PydanticOptimizerIFType,
    PydanticPytorchDeviceType,
    PydanticPytorchModuleType,
    PydanticSamplerIFType,
    PydanticTokenizerIFType,
)
from modalities.config.utils import parse_torch_device
from modalities.running_env.env_utils import (
    FSDP2MixedPrecisionSettings,
    MixedPrecisionSettings,
    PyTorchDtypes,
    has_bfloat_support,
)
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
    SelectiveActivationCheckpointingVariants,
)
from modalities.util import parse_enum_by_name


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


class PrecisionEnum(LookupEnum):
    FP32 = torch.float32
    FP16 = torch.float16
    BF16 = torch.bfloat16


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


class TorchCheckpointLoadingConfig(BaseModel):
    device: PydanticPytorchDeviceType
    precision: Optional[PrecisionEnum] = None

    @field_validator("device", mode="before")
    def parse_device(cls, device) -> PydanticPytorchDeviceType:
        return parse_torch_device(device)


class FSDP1CheckpointLoadingConfig(BaseModel):
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    block_names: list[str]
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


class DCPCheckpointLoadingConfig(BaseModel):
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class FSDP1CheckpointSavingConfig(BaseModel):
    checkpoint_path: Path
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    experiment_id: str


class DCPCheckpointSavingConfig(BaseModel):
    checkpoint_path: Path
    global_rank: Annotated[int, Field(strict=True, ge=0)]
    experiment_id: str


class CheckpointSavingConfig(BaseModel):
    checkpoint_saving_strategy: PydanticCheckpointSavingStrategyIFType
    checkpoint_saving_execution: PydanticCheckpointSavingExecutionIFType


class AdamOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    weight_decay_groups_excluded: list[str]


class AdamWOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    weight_decay_groups_excluded: list[str]


class DummyLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType


class StepLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    step_size: Annotated[int, Field(strict=True, gt=0)]
    gamma: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1


class OneCycleLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    max_lr: Annotated[float, Field(strict=True, gt=0.0)] | list[Annotated[float, Field(strict=True, gt=0.0)]]
    total_steps: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    epochs: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    steps_per_epoch: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    pct_start: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    anneal_strategy: str
    cycle_momentum: bool = True
    base_momentum: Annotated[float, Field(strict=True, gt=0)] | list[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.85
    max_momentum: Annotated[float, Field(strict=True, gt=0.0)] | list[
        Annotated[float, Field(strict=True, gt=0.0)]
    ] = 0.95
    div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    final_div_factor: Annotated[float, Field(strict=True, gt=0.0)]
    three_phase: bool = False
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1

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


class LinearLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    start_factor: Annotated[float, Field(strict=True, gt=0.0, le=1.0)]
    end_factor: Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
    total_iters: Annotated[int, Field(strict=True, gt=0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1


class CosineAnnealingLRSchedulerConfig(BaseModel):
    optimizer: PydanticOptimizerIFType
    t_max: Annotated[int, Field(strict=True, gt=0)]
    eta_min: Annotated[float, Field(strict=True, ge=0.0)]
    last_epoch: Annotated[int, Field(strict=True, ge=-1)] = -1


class FSDP1CheckpointedOptimizerConfig(BaseModel):
    checkpoint_loading: PydanticFSDP1CheckpointLoadingIFType
    checkpoint_path: Path
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType


class FSDP1CheckpointedModelConfig(BaseModel):
    checkpoint_loading: PydanticFSDP1CheckpointLoadingIFType
    checkpoint_path: Path
    model: PydanticPytorchModuleType


@deprecated(
    "With version 0.4, we upgraded FSDP to FSDP 2.0. Use get_fsdp_2_wrapped_model(...) "
    "and FSDP2WrappedModelConfig instead.",
    category=FutureWarning,
)
class FSDPWrappedModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    sync_module_states: bool
    mixed_precision_settings: MixedPrecisionSettings
    sharding_strategy: ShardingStrategy
    block_names: list[str]

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


class FSDP2WrappedModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    block_names: list[str]
    mixed_precision_settings: FSDP2MixedPrecisionSettings
    reshard_after_forward: bool = True
    device_mesh: PydanticDeviceMeshIFType

    @model_validator(mode="after")
    def validate_mixed_precision_settings(self):
        if not has_bfloat_support() and (
            self.mixed_precision_settings.reduce_dtype == PyTorchDtypes.BF_16
            or self.mixed_precision_settings.param_dtype == PyTorchDtypes.BF_16
        ):
            raise ValueError("BF16 not supported in the current environment")
        return self

    @model_validator(mode="after")
    def validate_dp_mesh_existence(self):
        if ParallelismDegrees.DP_SHARD.value not in self.device_mesh.mesh_dim_names:
            raise ValueError(f"Data parallelism key '{ParallelismDegrees.DP_SHARD.value}' not in {self.device_mesh=}")
        return self


class CompiledModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    block_names: list[str]
    fullgraph: Optional[bool] = True
    debug: Optional[bool] = False


class WeightInitializedModelConfig(BaseModel):
    model: PydanticPytorchModuleType
    model_initializer: PydanticModelInitializationIFType

    # avoid warning about protected namespace 'model_', see
    # https://docs.pydantic.dev/2.7/api/config/#pydantic.config.ConfigDict.protected_namespaces
    model_config = ConfigDict(protected_namespaces=())


class ActivationCheckpointedModelConfig(BaseModel):
    model: PydanticFSDP1ModuleType
    activation_checkpointing_modules: Optional[list[str]] = Field(default_factory=list)


class SelectiveActivationCheckpointedModelConfig(BaseModel):
    class FullACParams(BaseModel):
        pass

    class SelectiveLayerACParams(BaseModel):
        ac_freq: int

    class SelectiveOpACParams(BaseModel):
        save_ops_keys: list[str]

    sac_variant: SelectiveActivationCheckpointingVariants
    layers_fqn: str
    model: PydanticPytorchModuleType | PydanticFSDP1ModuleType
    sac_fun_params: Optional[FullACParams | SelectiveLayerACParams | SelectiveOpACParams] = None


class RawAppStateConfig(BaseModel):
    model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType
    lr_scheduler: Optional[PydanticLRSchedulerIFType] = None


class DCPAppStateConfig(BaseModel):
    raw_app_state: PydanticAppStateType
    checkpoint_dir_path: Path


class PreTrainedHFTokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    max_length: Optional[Annotated[int, Field(strict=True, ge=0)]] = None
    truncation: bool = False
    padding: bool | str = False
    special_tokens: Optional[dict[str, str]] = None


class PreTrainedSPTokenizerConfig(BaseModel):
    tokenizer_model_file: str


class SequentialSamplerConfig(BaseModel):
    data_source: PydanticDatasetIFType


class DistributedSamplerConfig(BaseModel):
    rank: Annotated[int, Field(strict=True, ge=0)]
    num_replicas: Annotated[int, Field(strict=True, ge=0)]
    shuffle: bool
    dataset: PydanticDatasetIFType
    seed: Optional[int] = 0
    drop_last: Literal[True] = True


class ResumableDistributedSamplerConfig(BaseModel):
    dataset: PydanticDatasetIFType
    rank: Annotated[int, Field(strict=True, ge=0)]
    num_replicas: Annotated[int, Field(strict=True, ge=0)] = None
    epoch: Annotated[int, Field(strict=True, ge=0)] = 0
    shuffle: Optional[bool] = False
    seed: Optional[int] = 0
    drop_last: Literal[True] = True
    skip_num_global_samples: Annotated[int, Field(strict=True, ge=0)] = 0


class MemMapDatasetConfig(BaseModel):
    raw_data_path: FilePath
    index_path: Optional[FilePath] = None
    tokenizer: PydanticTokenizerIFType
    jq_pattern: str
    sample_key: str


class PackedMemMapDatasetContinuousConfig(BaseModel):
    raw_data_path: Path
    sequence_length: Annotated[int, Field(strict=True, gt=1)]
    sample_key: str


class PackedMemMapDatasetMegatronConfig(BaseModel):
    raw_data_path: Path
    block_size: Annotated[int, Field(strict=True, gt=1)]
    sample_key: str


class CombinedDatasetConfig(BaseModel):
    datasets: list[PydanticDatasetIFType]


class BatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIFType
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    drop_last: Literal[True] = True


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


class DummyProgressSubscriberConfig(BaseModel):
    pass


class RichProgressSubscriberConfig(BaseModel):
    eval_dataloaders: Optional[list[PydanticLLMDataLoaderIFType]] = Field(default_factory=list)
    train_dataloader_tag: str
    num_seen_steps: Annotated[int, Field(strict=True, ge=0)]
    num_target_steps: Annotated[int, Field(strict=True, gt=0)]
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class DummyResultSubscriberConfig(BaseModel):
    pass


class WandBEvaluationResultSubscriberConfig(BaseModel):
    global_rank: int
    project: str
    experiment_id: str
    mode: WandbMode
    directory: Path
    config_file_path: Path


class RichResultSubscriberConfig(BaseModel):
    num_ranks: int
    global_rank: int


class GPT2MFUCalculatorConfig(BaseModel):
    n_layer: Annotated[int, Field(strict=True, gt=0)]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    n_embd: Annotated[int, Field(strict=True, gt=0)]
    world_size: Annotated[int, Field(strict=True, gt=0)]
    wrapped_model: PydanticFSDP1ModuleType | PydanticFSDP2ModuleType


def load_app_config_dict(
    config_file_path: Path,
    experiment_id: Optional[str] = None,
    additional_resolver_funs: Optional[dict[str, Callable]] = None,
) -> dict:
    """Load the application configuration from the given YAML file.
    The function defines custom resolvers for the OmegaConf library to resolve environment variables and
    Modalities-specific variables.

    Args:
        config_file_path (Path): YAML config file.
        experiment_id (str, optional): The experiment_id of the current run. Defaults to None.
        additional_resolver_funs (dict[str, Callable], optional): Additional resolver functions. Defaults to None.

    Returns:
        dict: Dictionary representation of the config file.
    """

    def cuda_env_resolver_fun(var_name: str) -> int:
        int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    def modalities_env_resolver_fun(var_name: str, kwargs: dict[str, Any]) -> str | Path:
        if var_name in kwargs:
            return kwargs[var_name]
        else:
            raise ValueError(f"Unknown modalities_env variable: {var_name}.")

    def node_env_resolver_fun(var_name: str) -> int:
        if var_name == "num_cpus":
            return os.cpu_count()

    OmegaConf.register_new_resolver("cuda_env", cuda_env_resolver_fun, replace=True)
    modalities_env_kwargs = {"config_file_path": config_file_path}
    if experiment_id is not None:
        modalities_env_kwargs["experiment_id"] = experiment_id
    OmegaConf.register_new_resolver(
        "modalities_env", partial(modalities_env_resolver_fun, kwargs=modalities_env_kwargs), replace=True
    )
    OmegaConf.register_new_resolver("node_env", node_env_resolver_fun, replace=True)

    if additional_resolver_funs is not None:
        for resolver_name, resolver_fun in additional_resolver_funs.items():
            OmegaConf.register_new_resolver(resolver_name, resolver_fun, replace=True)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    return config_dict
