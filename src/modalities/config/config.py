import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import torch.nn as nn
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, FilePath, GetCoreSchemaHandler, PositiveInt, field_validator
from pydantic_core import core_schema
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import Optimizer
from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
from transformers import GPT2TokenizerFast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from modalities.checkpointing.checkpointing import CheckpointingIF
from modalities.checkpointing.checkpointing_execution import CheckpointingExecutionIF
from modalities.checkpointing.checkpointing_strategies import CheckpointingStrategyIF
from modalities.config.lookup_enum import LookupEnum
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.logging_broker.subscriber import MessageSubscriberIF
from modalities.loss_functions import Loss
from modalities.models.gpt2.collator import CollateFnIF
from modalities.running_env.env_utils import MixedPrecisionSettings, has_bfloat_support
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


PydanticCheckpointingIFType = Annotated[CheckpointingIF, PydanticThirdPartyTypeIF(CheckpointingIF)]
PydanticCheckpointingStrategyIFType = Annotated[
    CheckpointingStrategyIF, PydanticThirdPartyTypeIF(CheckpointingStrategyIF)
]
PydanticCheckpointingExecutionIFType = Annotated[
    CheckpointingExecutionIF, PydanticThirdPartyTypeIF(CheckpointingExecutionIF)
]
PydanticPytorchModuleType = Annotated[nn.Module, PydanticThirdPartyTypeIF(nn.Module)]
PydanticTokenizerIFType = Annotated[PreTrainedTokenizerFast, PydanticThirdPartyTypeIF(PreTrainedTokenizerFast)]
PydanticDatasetIFType = Annotated[Dataset, PydanticThirdPartyTypeIF(Dataset)]
PydanticSamplerIFType = Annotated[Sampler, PydanticThirdPartyTypeIF(Sampler)]
PydanticCollateFnIFType = Annotated[CollateFnIF, PydanticThirdPartyTypeIF(CollateFnIF)]
PydanticLLMDataLoaderIFType = Annotated[LLMDataLoader, PydanticThirdPartyTypeIF(LLMDataLoader)]
PydanticOptimizerIFType = Annotated[Optimizer, PydanticThirdPartyTypeIF(Optimizer)]
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
    NONE = "NONE"
    VALUE = "value"
    P2_NORM = "p2_norm"
    MAX_NORM = "max_norm"


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


class AdamWOptimizerConfig(BaseModel):
    lr: float
    wrapped_model: PydanticPytorchModuleType


class CheckpointedOptimizerConfig(BaseModel):
    checkpointing: PydanticCheckpointingIFType
    checkpoint_path: Path
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType


class CheckpointedModelConfig(BaseModel):
    checkpointing: PydanticCheckpointingIFType
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


class GPT2TokenizerFastConfig(BaseModel):
    # Note: huggingface tokenizers expect file path as string
    tokenizer_file: str


class DistributedSamplerConfig(BaseModel):
    rank: Annotated[int, Field(strict=True, ge=0)]
    num_replicas: Annotated[int, Field(strict=True, ge=0)]
    shuffle: bool
    dataset: PydanticDatasetIFType


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
    drop_last: bool


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
    collate_fn: PydanticCollateFnIFType
    num_workers: Annotated[int, Field(strict=True, ge=0)]
    pin_memory: bool
    shuffle: bool
    skip_num_batches: Optional[int] = 0


class DummyProgressSubscriberConfig(BaseModel):
    pass


class RichProgressSubscriberConfig(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: Optional[List[PydanticLLMDataLoaderIFType]] = Field(default_factory=list)
    world_size: int
    global_num_seen_samples: int
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


class CudaEnv(BaseModel):
    local_rank: Annotated[int, Field(strict=True, ge=0)]
    world_size: Annotated[int, Field(strict=True, ge=1)]
    global_rank: Annotated[int, Field(strict=True, ge=0)]


class Settings(BaseModel):
    class Training(BaseModel):
        class GradientClipping(BaseModel):
            mode: GradientClippingMode
            threshold: Annotated[float, Field(strict=True, gt=0.0)] = 1.0

        callback_interval_in_samples: Annotated[int, Field(strict=True, ge=1)]
        global_num_training_samples: Annotated[int, Field(strict=True, ge=1)]
        global_num_seen_samples: Annotated[int, Field(strict=True, ge=0)]
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
    cuda_env: CudaEnv
    paths: Paths


class ComponentsModel(BaseModel):
    wrapped_model: PydanticPytorchModuleType
    optimizer: PydanticOptimizerIFType
    loss_fn: PydanticLossIFType
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: List[PydanticLLMDataLoaderIFType]
    batch_progress_subscriber: PydanticMessageSubscriberIFType
    evaluation_subscriber: PydanticMessageSubscriberIFType
    checkpointing: PydanticCheckpointingIFType
    settings: Settings


class ComponentsInferenceModel(BaseModel):
    wrapped_model: PydanticPytorchModuleType
    cuda_env: CudaEnv


def load_app_config_dict(config_file_path: Path) -> Dict:
    def cuda_env_resolver_fun(var_name: str) -> int:
        int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    def modalities_env_resolver_fun(var_name: str) -> int:
        if var_name == "experiment_id":
            return get_date_of_run()

    OmegaConf.register_new_resolver("cuda_env", cuda_env_resolver_fun, replace=True)
    OmegaConf.register_new_resolver("modalities_env", modalities_env_resolver_fun, replace=True)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict
