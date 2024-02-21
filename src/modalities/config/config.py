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
from modalities.util import parse_enum_by_name


class PydanticCheckpointingIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CheckpointingIF),
            python_schema=core_schema.is_instance_schema(CheckpointingIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticCheckpointingStrategyIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CheckpointingStrategyIF),
            python_schema=core_schema.is_instance_schema(CheckpointingStrategyIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticCheckpointingExecutionIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CheckpointingExecutionIF),
            python_schema=core_schema.is_instance_schema(CheckpointingExecutionIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticModelIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(nn.Module),
            python_schema=core_schema.is_instance_schema(nn.Module),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticTokenizerIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(PreTrainedTokenizerFast),
            python_schema=core_schema.is_instance_schema(PreTrainedTokenizerFast),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticDatasetIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(Dataset),
            python_schema=core_schema.is_instance_schema(Dataset),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticSamplerIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(Sampler),
            python_schema=core_schema.is_instance_schema(Sampler),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticCollateFnIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(CollateFnIF),
            python_schema=core_schema.is_instance_schema(CollateFnIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticLLMDataLoaderIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(LLMDataLoader),
            python_schema=core_schema.is_instance_schema(LLMDataLoader),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticOptimizerIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(Optimizer),
            python_schema=core_schema.is_instance_schema(Optimizer),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticLossIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(Loss),
            python_schema=core_schema.is_instance_schema(Loss),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


class PydanticMessageSubscriberIF:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # see: https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types
        return core_schema.json_or_python_schema(
            json_schema=core_schema.is_instance_schema(MessageSubscriberIF),
            python_schema=core_schema.is_instance_schema(MessageSubscriberIF),
            # serialization=core_schema.plain_serializer_function_ser_schema(
            #     lambda instance: instance.x
            # ),
        )


PydanticCheckpointingIFType = Annotated[CheckpointingIF, PydanticCheckpointingIF]
PydanticCheckpointingStrategyIFType = Annotated[CheckpointingStrategyIF, PydanticCheckpointingStrategyIF]
PydanticCheckpointingExecutionIFType = Annotated[CheckpointingExecutionIF, PydanticCheckpointingExecutionIF]
PydanticModelIFType = Annotated[nn.Module, PydanticModelIF]
PydanticTokenizerIFType = Annotated[PreTrainedTokenizerFast, PydanticTokenizerIF]
PydanticDatasetIFType = Annotated[Dataset, PydanticDatasetIF]
PydanticSamplerIFType = Annotated[Sampler, PydanticSamplerIF]
PydanticCollateFnIFType = Annotated[CollateFnIF, PydanticCollateFnIF]
PydanticLLMDataLoaderIFType = Annotated[LLMDataLoader, PydanticLLMDataLoaderIF]
PydanticOptimizerIFType = Annotated[Optimizer, PydanticOptimizerIF]
PydanticLossIFType = Annotated[Loss, PydanticLossIF]
PydanticMessageSubscriberIFType = Annotated[MessageSubscriberIF, PydanticMessageSubscriberIF]


class ProcessGroupBackendType(LookupEnum):
    nccl = "nccl"


class TokenizerTypes(LookupEnum):
    GPT2TokenizerFast = GPT2TokenizerFast
    LlamaTokenizerFast = LlamaTokenizerFast


class PassType(LookupEnum):
    BY_VALUE = "by_value"
    BY_REFERENCE = "by_reference"


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
    wrapped_model: PydanticModelIFType


class CheckpointedOptimizerConfig(BaseModel):
    checkpointing: PydanticCheckpointingIFType
    checkpoint_path: Path
    wrapped_model: PydanticModelIFType
    optimizer: PydanticOptimizerIFType


class CheckpointedModelConfig(BaseModel):
    checkpointing: PydanticCheckpointingIFType
    checkpoint_path: Path
    model: PydanticModelIFType


class FSDPWrappedModelConfig(BaseModel):
    model: PydanticModelIFType
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
    sampler: PydanticSamplerIF
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    drop_last: bool


class ResumableBatchSamplerConfig(BaseModel):
    sampler: PydanticSamplerIF
    start_index: Annotated[int, Field(strict=True, gt=0)]


class GPT2LLMCollateFnConfig(BaseModel):
    sample_key: str
    target_key: str


class LLMDataLoaderConfig(BaseModel):
    dataloader_tag: str
    dataset: PydanticDatasetIF
    batch_sampler: PydanticSamplerIF
    collate_fn: PydanticCollateFnIF
    num_workers: Annotated[int, Field(strict=True, ge=0)]
    pin_memory: bool
    shuffle: bool
    skip_num_batches: Optional[int] = 0


class WandbMode(LookupEnum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DISABLED = "DISABLED"


class DummyProgressSubscriberConfig(BaseModel):
    pass


class RichProgressSubscriberConfig(BaseModel):
    train_dataloader: PydanticLLMDataLoaderIFType
    eval_dataloaders: Dict[str, PydanticLLMDataLoaderIFType]
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
        callback_interval_in_batches: Annotated[int, Field(strict=True, ge=1)]
        global_num_training_samples: Annotated[int, Field(strict=True, ge=1)]
        global_num_seen_samples: Annotated[int, Field(strict=True, ge=0)]
        do_apply_activation_checkpointing: bool
        gradient_acc_step: Annotated[int, Field(strict=True, ge=1)]
        local_train_micro_batch_size: Annotated[int, Field(strict=True, ge=1)]
        sequence_length: Annotated[int, Field(strict=True, ge=1)]

    experiment_id: str
    referencing_keys: Dict[str, str]
    training: Training
    cuda_env: CudaEnv


class ComponentsModel(BaseModel):
    wrapped_model: PydanticModelIFType
    optimizer: PydanticOptimizerIFType
    loss_fn: PydanticLossIFType
    train_dataloader: PydanticLLMDataLoaderIFType
    val_dataloader: PydanticLLMDataLoaderIFType
    test_dataloader: PydanticLLMDataLoaderIFType
    batch_progress_subscriber: PydanticMessageSubscriberIFType
    evaluation_subscriber: PydanticMessageSubscriberIFType
    checkpointing: PydanticCheckpointingIFType

    settings: Settings


class ComponentsInferenceModel(BaseModel):
    wrapped_model: PydanticModelIFType
    cuda_env: CudaEnv


def load_app_config_dict(config_file_path: Path) -> Dict:
    int_env_variable_names = ["LOCAL_RANK", "WORLD_SIZE", "RANK"]

    def resolver_fun(var_name: str) -> int:
        return int(os.getenv(var_name)) if var_name in int_env_variable_names else os.getenv(var_name)

    OmegaConf.register_new_resolver("modalities_env", resolver_fun)

    cfg = OmegaConf.load(config_file_path)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return config_dict
