from typing import Any, Dict

import torch.optim as optim
from class_resolver import ClassResolver
from pydantic import BaseModel

from llm_gym.config.config import AppConfig, OptimizerTypes, SchedulerTypes
from llm_gym.config.lookup_types import LossTypes, ModelTypes, SamplerTypes
from llm_gym.fsdp.fsdp_running_env import FSDPRunningEnv, RunningEnv, RunningEnvTypes
from llm_gym.loss_functions import CLMCrossEntropyLoss, Loss
from llm_gym.models.gpt2.gpt2_model import GPT2LLM, NNModel
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class ResolverRegister:
    def __init__(self, config: AppConfig) -> None:
        self._resolver_register: Dict[str, ClassResolver] = ResolverRegister._create_resolver_register(config=config)

    def build_component_by_config(self, config: BaseModel, extra_kwargs: Dict = {}) -> Any:
        assert (
            "type_hint" in config.model_fields.keys()
        ), f"Field 'type_hint' missing but needed for initalisation in {config}"

        return self._build_component(
            register_key=config.type_hint,
            register_query=config.type_hint.name,
            extra_kwargs=dict(
                # do not model_dump recursively
                **{key: getattr(config.config, key) for key in config.config.model_dump().keys()},
                **extra_kwargs,
            ),
        )

    def build_component_by_key_query(self, register_key: str, type_hint: str, extra_kwargs: Dict = {}) -> Any:
        return self._build_component(register_key=register_key, register_query=type_hint, extra_kwargs=extra_kwargs)

    def _build_component(self, register_key: str, register_query: str, extra_kwargs: Dict = {}):
        return self._resolver_register[register_key].make(
            query=register_query,
            pos_kwargs=extra_kwargs,
        )

    @staticmethod
    def _create_resolver_register(config: AppConfig) -> Dict[str, ClassResolver]:
        expected_resolvers = [
            value["type_hint"]
            for value in config.model_dump().values()
            if isinstance(value, Dict) and "type_hint" in value.keys()
        ]
        resolvers = {
            config.running_env.type_hint: ClassResolver(
                [t.value for t in RunningEnvTypes],
                base=RunningEnv,
                default=FSDPRunningEnv,
            ),
            config.model.type_hint: ClassResolver(
                [t.value for t in ModelTypes],
                base=NNModel,
                default=GPT2LLM,
            ),
            config.optimizer.type_hint: ClassResolver(
                [t.value for t in OptimizerTypes],
                base=optim.Optimizer,
                default=optim.AdamW,
            ),
            config.scheduler.type_hint: ClassResolver(
                [t.value for t in SchedulerTypes],
                base=optim.lr_scheduler.LRScheduler,
                default=optim.lr_scheduler.StepLR,
            ),
            config.loss.type_hint: ClassResolver(
                [t.value for t in LossTypes],
                base=Loss,
                default=CLMCrossEntropyLoss,
            ),
            SamplerTypes: ClassResolver(
                [t.value for t in SamplerTypes],
                base=Sampler,
                default=DistributedSampler,
            ),
        }
        assert set(expected_resolvers) == set(
            resolvers
        ), f"Some resolvers are not registered: {set(expected_resolvers).symmetric_difference(resolvers)}"
        return resolvers

    def add_resolver(self, resolver_key: str, resolver: ClassResolver):
        self._resolver_register[resolver_key] = resolver
