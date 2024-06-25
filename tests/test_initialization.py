import os
from pathlib import Path
from typing import Dict

import numpy as np
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
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from tests.conftest import _ROOT_DIR

# NOTE: We need to run the tests in a torch distributed environment with 1 GPU.
# CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 \
#   $(which pytest) path/to/test_initialization.py


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


# architecture
GPT2_NLAYERS = 12
GPT2_FFN_HIDDEN = 2048
GPT2_VOCAB_SIZE = 50304
GPT2_SEQUENCE_LENGTH = 2048
GPT2_HIDDEN_DIM = 768
COCA_NLAYERS = 6 + 6  # text + multimodal

# parameters
AVG = {
    "gpt2": {
        "weight-normal": 0.0,
        "weight-projection": 0.0,
        "weight-norm": 1.0,
        "bias": 0.0,
    },
    "coca": {
        "weight-normal": 0.0,
        "weight-projection": 0.0,
        "weight-norm": 1.0,
        "bias": 0.0,
    },
}
STD = {
    "gpt2": {
        "weight-normal": 0.02,
        "weight-projection": 0.02 / np.sqrt(2 * GPT2_NLAYERS),
        "weight-norm": 0,
        "bias": 0,
    },
    "coca": {
        "weight-normal": 0.02,
        "weight-projection": 0.02 / np.sqrt(2 * COCA_NLAYERS),
        "weight-norm": 0,
        "bias": 0,
    },
}

# number of parameters for each parameter group
ALL = {
    "gpt2": 106374912,
    "coca": 184502784,
}
GPT2_LINEAR_PROJECTION = (GPT2_HIDDEN_DIM**2 + GPT2_HIDDEN_DIM * GPT2_FFN_HIDDEN) * GPT2_NLAYERS  # 25952256
GPT2_EMBEDDING = GPT2_HIDDEN_DIM * (GPT2_VOCAB_SIZE + GPT2_SEQUENCE_LENGTH)
GPT2_NORM = GPT2_HIDDEN_DIM * (GPT2_NLAYERS * 2 + 1)  # second term = num_layer_norms = (12*2+1) = 25
GPT2_BIAS = 89856
GPT2_LINEAR_NO_PROJECTION = ALL["gpt2"] - GPT2_LINEAR_PROJECTION - GPT2_EMBEDDING - GPT2_NORM - GPT2_BIAS  # 40107264


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "model_name, initialization, groups",
    [
        (
            "gpt2",
            "scaled",
            {
                "weight-normal": GPT2_LINEAR_NO_PROJECTION + GPT2_EMBEDDING,
                "weight-projection": GPT2_LINEAR_PROJECTION,
                "weight-norm": GPT2_NORM,
                "bias": GPT2_BIAS,
                "other": 0,
            },
        ),
        (
            "coca",
            "scaled",
            {
                "weight-normal": 169332480,
                "weight-projection": 14745600,
                "weight-norm": 34560,
                "bias": 191232,
                "other": 198912,
            },
        ),
    ],
)
def test_initialization_groups(model_name, initialization, groups):
    """
    verifies that, for a given model architectrue and a given initialization,
    the different model parameter initialization groups
    - have the expected number of parameters
    - have the expected mean and std
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        if model_name == "gpt2":
            model = _load_gpt2()
        elif model_name == "coca":
            model = _load_coca()
        print(model)  # for debugging

        params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        init = {}
        if initialization == "scaled":
            init["weight-normal"] = torch.cat(
                [
                    torch.flatten(parameter.detach())
                    for name, parameter in params.items()
                    if name.endswith(".weight")
                    and not name.endswith("c_proj.weight")
                    and not name.endswith("norm.weight")
                    and not name.endswith("norm1.weight")
                    and not name.endswith("norm2.weight")
                    and not name.endswith("ln_1.weight")
                    and not name.endswith("ln_2.weight")
                    and not name.endswith("ln_3.weight")
                    and not name.endswith("ln_4.weight")
                    and not name.endswith("ln_f.weight")
                ]
            )
            init["weight-projection"] = torch.cat(
                [
                    torch.flatten(parameter.detach())
                    for name, parameter in params.items()
                    if name.endswith("c_proj.weight")
                ]
            )
            init["weight-norm"] = torch.cat(
                [
                    torch.flatten(parameter.detach())
                    for name, parameter in params.items()
                    if name.endswith("norm.weight")
                    or name.endswith("norm1.weight")
                    or name.endswith("norm2.weight")
                    or name.endswith("ln_1.weight")
                    or name.endswith("ln_2.weight")
                    or name.endswith("ln_3.weight")
                    or name.endswith("ln_4.weight")
                    or name.endswith("ln_f.weight")
                ]
            )
            init["bias"] = torch.cat(
                [
                    torch.flatten(parameter.detach())
                    for name, parameter in params.items()
                    if name.endswith(".bias") and "conv" not in name
                ]
            )
            other = [
                torch.flatten(parameter.detach())
                for name, parameter in params.items()
                if not name.endswith(".weight") and (not name.endswith(".bias") or "conv" in name)
            ]
            init["other"] = torch.cat(other) if len(other) else None
            count_all = 0
            for key in ["weight-normal", "weight-projection", "weight-norm", "bias", "other"]:
                # check number of parameters in each group
                len_init_key = len(init[key]) if init[key] is not None else 0
                assert len_init_key == groups[key], f"len(init[{key}]) = {len_init_key} should be {groups[key]}"
                count_all += len_init_key

                # check mean and std for each group
                if key != "other":
                    avg = torch.mean(init[key])
                    std = torch.std(init[key])
                    avg_test = torch.tensor(AVG[model_name][key], device=avg.device, dtype=avg.dtype)
                    std_test = torch.tensor(STD[model_name][key], device=std.device, dtype=std.dtype)
                    torch.testing.assert_close(
                        avg, avg_test, msg=f"average for {key} = {avg} should be close to {avg_test}"
                    )
                    torch.testing.assert_close(
                        std, std_test, msg=f"standard deviation for {key} = {std} should be close to {std_test}"
                    )

            # check total number of parameters
            assert count_all == ALL[model_name], f"total number of parameters = {count_all} should be {ALL[model_name]}"

        else:
            raise Exception(f"Initialization = {initialization} not implemented.")
