import os
from pathlib import Path
from typing import Dict

import pytest
import torch
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.coca.coca_model import CoCa, CoCaConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.model_factory import ModelFactory
from modalities.optimizers.optimizer_factory import get_optimizer_groups
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from tests.conftest import _ROOT_DIR

# NOTE: We need to run the tests in a torch distributed environment with 1 GPU.
# CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 \
#   $(which pytest) path/to/test_optimizer_factory.py


def get_gpt2_model_from_config(gpt2_model_config_dict: Dict) -> GPT2LLM:
    class GPT2InstantationModel(BaseModel):
        model: PydanticPytorchModuleType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(
        config_dict=gpt2_model_config_dict, components_model_type=GPT2InstantationModel
    )

    model = components.model
    return model


def _load_gpt2() -> FSDP:
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/gpt2_config_optimizer.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    gpt2_model = get_gpt2_model_from_config(config_dict)
    gpt2_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return gpt2_wrapped_model


def _load_coca() -> FSDP:
    config_file_path = _ROOT_DIR / Path("tests/models/coca/coca_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    coca_config = CoCaConfig.model_validate(config_dict)
    coca_model = CoCa(**dict(coca_config))
    coca_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        coca_model,
        sync_module_states=True,
        block_names=["TransformerBlock", "VisionTransformerBlock"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return coca_wrapped_model


# number of parameters for each optimizer group
GPT2_LINEAR = 66130944
GPT2_EMBEDDING = 768 * (50304 + 2048)  # n_embd * (vocab_size + sequence_length)
GPT2_LAYERNORM = 768 * 50  # n_embd * num_layer_norms
COCA_ALL = 184502784


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "model_name, weight_decay, weight_decay_groups_excluded, success,"
    "num_decayed_parameters, num_nondecayed_parameters",
    [
        ("gpt2", 0, [], True, 0, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, [], True, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM, 0),
        ("gpt2", 1e-1, ["embedding"], True, GPT2_LINEAR + GPT2_LAYERNORM, GPT2_EMBEDDING),
        ("gpt2", 1e-1, ["embedding", "layernorm"], True, GPT2_LINEAR, GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, ["linear", "embedding", "layernorm"], True, 0, GPT2_LINEAR + GPT2_EMBEDDING + GPT2_LAYERNORM),
        ("gpt2", 1e-1, ["non-existing-group"], False, None, None),
        ("coca", 0, [], True, 0, COCA_ALL),
        ("coca", 1e-1, [], True, COCA_ALL, 0),
        ("coca", 1e-1, ["non-existing-group"], False, None, None),
    ],
)
def test_get_optimizer_groups(
    model_name, weight_decay, weight_decay_groups_excluded, success, num_decayed_parameters, num_nondecayed_parameters
):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        if model_name == "gpt2":
            model = _load_gpt2()
        elif model_name == "coca":
            model = _load_coca()

        if not success:
            with pytest.raises(Exception):
                get_optimizer_groups(model, weight_decay, weight_decay_groups_excluded)
        else:
            optimizer_groups = get_optimizer_groups(model, weight_decay, weight_decay_groups_excluded)
            test_num_decayed_parameters = sum(
                p.numel() for group in optimizer_groups for p in group["params"] if group["weight_decay"] > 0
            )
            test_num_nondecayed_parameters = sum(
                p.numel() for group in optimizer_groups for p in group["params"] if group["weight_decay"] == 0
            )

            assert (
                test_num_decayed_parameters == num_decayed_parameters
            ), f"#(decayed parameters) = {test_num_decayed_parameters} should be {num_decayed_parameters}"
            assert (
                test_num_nondecayed_parameters == num_nondecayed_parameters
            ), f"#(non-decayed parameters) = {test_num_nondecayed_parameters} should be {num_nondecayed_parameters}"
