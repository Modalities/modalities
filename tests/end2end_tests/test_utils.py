import os
from pathlib import Path

import pytest
import torch
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.model_factory import ModelFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.util import get_total_number_of_trainable_parameters
from tests.conftest import _ROOT_DIR

# NOTE: We need to run the tests in a torch distributed environment with 2 GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   $(which pytest) path/to/test_utils.py


def get_model_from_config(model_config_dict: dict) -> GPT2LLM:
    """get gpt2 or coca model from config_dict"""

    class InstantationModel(BaseModel):
        model: PydanticPytorchModuleType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(
        config_dict=model_config_dict, components_model_type=InstantationModel
    )

    model = components.model
    return model


def _load_gpt2(
    initialization_type: str, std: float | str, sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
) -> FSDP:
    """load gpt2 model from config and fsdp-wrap it"""
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/gpt2_config_initialization.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    # config_dict = _replace_config_dict(config_dict, initialization_type, std)

    gpt2_model = get_model_from_config(config_dict)
    gpt2_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=sharding_strategy,
    )
    return gpt2_wrapped_model


def _load_model(
    model_name: str,
    initialization: str = "plain",
    std: float | str = 0.02,
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD,
) -> FSDP:
    """load gpt2 or coca model from config and fsdp-wrap it"""
    if model_name == "gpt2":
        model = _load_gpt2(initialization_type=initialization, std=std, sharding_strategy=sharding_strategy)
    else:
        raise Exception(f"model = {model_name} not implemented.")
    return model


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "model_name, sharding_strategy, expected_nr_parameters",
    [
        ("gpt2", ShardingStrategy.NO_SHARD, 145009152),
        ("gpt2", ShardingStrategy.FULL_SHARD, 145009152),
        ("gpt2", ShardingStrategy.HYBRID_SHARD, 145009152),
    ],
)
def test_get_total_number_of_trainable_parameters(
    model_name: str, sharding_strategy: ShardingStrategy, expected_nr_parameters: int
):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        model = _load_model(model_name, sharding_strategy=sharding_strategy)
        assert model.sharding_strategy == sharding_strategy

        nr_parameters = get_total_number_of_trainable_parameters(model)
        assert nr_parameters == expected_nr_parameters
