from typing import Any, Dict, List

import torch.optim as optim
from class_resolver import ClassResolver
from pydantic import BaseModel
from torch.utils.data import BatchSampler, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer

from modalities.checkpointing.checkpointing import CheckpointingExecutionIF, CheckpointingStrategyIF
from modalities.config.config import OptimizerTypes, SchedulerTypes
from modalities.config.lookup_types import (
    BatchSamplerTypes,
    CheckpointingExectionTypes,
    CheckpointingStrategyTypes,
    CodecTypes,
    CollatorTypes,
    DataloaderTypes,
    DatasetTypes,
    LookupEnum,
    LossTypes,
    ModelTypes,
    SamplerTypes,
    TokenizerTypes,
)
from modalities.dataloader.codecs import Codec
from modalities.dataloader.dataloader import LLMDataLoader
from modalities.dataloader.dataset import Dataset
from modalities.loss_functions import CLMCrossEntropyLoss, Loss
from modalities.models.gpt2.collator import GPT2LLMCollator
from modalities.models.gpt2.gpt2_model import GPT2LLM, NNModel
from modalities.running_env.fsdp.fsdp_running_env import FSDPRunningEnv, RunningEnv, RunningEnvTypes


# TODO: this should be a singleton
class ResolverRegister:
    # TODO: args and kwargs only to be backwards compatible
    #       older versions required the appconfig as argument
    def __init__(self, *args, **kwargs):
        self._resolver_register = self._build_resolver_register()

    def build_component_by_key_query(self, register_key: str, type_hint: str, extra_kwargs: Dict = {}) -> Any:
        raise NotImplementedError

    def build_component_by_config(self, config: BaseModel, extra_kwargs: Dict[str, Any] = {}) -> Any:
        assert (
            "type_hint" in config.model_fields.keys()
        ), f"Field 'type_hint' missing but needed for initalisation in {config}"

        assert (
            "config" in config.model_fields.keys()
        ), f"Field 'config' missing but needed for initalisation in {config}"

        kwargs = extra_kwargs.copy()

        for key in config.config.model_fields.keys():
            # get the value corresponding to the key
            # prefer the extra keyword arguments when both specified
            val = getattr(config.config, key)
            val = kwargs.get(key, val)

            # handle nested components
            if isinstance(val, BaseModel) and "type_hint" in val.model_fields and "config" in val.model_fields:
                kwargs[key] = self.build_component_by_config(val)

            else:
                kwargs[key] = val

        return self._build_component(
            register_key=type(config.type_hint),
            register_query=config.type_hint.name,
            extra_kwargs=kwargs,
        )

    def _build_component(self, register_key: LookupEnum, register_query: str, extra_kwargs: Dict[str, Any] = {}):
        assert register_key in self._resolver_register
        return self._resolver_register[register_key].make(
            query=register_query,
            pos_kwargs=extra_kwargs,
        )

    def _build_resolver_register(self) -> List[LookupEnum]:
        return {
            RunningEnvTypes: ClassResolver(
                [t.value for t in RunningEnvTypes],
                base=RunningEnv,
                default=FSDPRunningEnv,
            ),
            ModelTypes: ClassResolver(
                [t.value for t in ModelTypes],
                base=NNModel,
                default=GPT2LLM,
            ),
            OptimizerTypes: ClassResolver(
                [t.value for t in OptimizerTypes],
                base=optim.Optimizer,
                default=optim.AdamW,
            ),
            SchedulerTypes: ClassResolver(
                [t.value for t in SchedulerTypes],
                base=optim.lr_scheduler.LRScheduler,
                default=optim.lr_scheduler.StepLR,
            ),
            LossTypes: ClassResolver(
                [t.value for t in LossTypes],
                base=Loss,
                default=CLMCrossEntropyLoss,
            ),
            SamplerTypes: ClassResolver(
                classes=[t.value for t in SamplerTypes],
                base=Sampler,
                default=DistributedSampler,
            ),
            BatchSamplerTypes: ClassResolver(
                classes=[t.value for t in BatchSamplerTypes],
                base=BatchSampler,
                default=BatchSampler,
            ),
            DataloaderTypes: ClassResolver(
                [t.value for t in DataloaderTypes],
                base=DataLoader,
                default=LLMDataLoader,
            ),
            DatasetTypes: ClassResolver([t.value for t in DatasetTypes], base=Dataset),
            CollatorTypes: ClassResolver([t.value for t in CollatorTypes], base=GPT2LLMCollator),
            TokenizerTypes: ClassResolver([t.value for t in TokenizerTypes], base=PreTrainedTokenizer),
            CodecTypes: ClassResolver([t.value for t in CodecTypes], base=Codec),
            CheckpointingStrategyTypes: ClassResolver(
                [t.value for t in CheckpointingStrategyTypes], base=CheckpointingStrategyIF
            ),
            # TODO: fix type in execution
            CheckpointingExectionTypes: ClassResolver(
                [t.value for t in CheckpointingExectionTypes], base=CheckpointingExecutionIF
            ),
        }
