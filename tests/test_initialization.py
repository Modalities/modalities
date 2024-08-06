import math
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pytest
import torch
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.coca.coca_model import CoCa
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


def get_model_from_config(model_config_dict: Dict) -> GPT2LLM | CoCa:
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


def _replace_config_dict(_config_dict: Dict, _initialization_type: str, _std: str) -> Dict:
    """dynamically replace initialization_type, std and dependent fields in config_dict"""
    _config_dict["model"]["config"]["model_initializer"]["config"]["weight_init_type"] = _initialization_type  # replace
    _config_dict["model"]["config"]["model_initializer"]["config"]["std"] = _std  # replace
    if _initialization_type == "plain":
        _config_dict["model"]["config"]["model_initializer"]["config"]["num_layers"] = None  # replace
    if _std != "auto":
        _config_dict["model"]["config"]["model_initializer"]["config"]["hidden_dim"] = None  # replace
    return _config_dict


def _load_gpt2(initialization_type: str, std: float | str) -> FSDP:
    """load gpt2 model from config and fsdp-wrap it"""
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/gpt2_config_initialization.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config_dict = _replace_config_dict(config_dict, initialization_type, std)

    gpt2_model = get_model_from_config(config_dict)
    gpt2_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return gpt2_wrapped_model


def _load_coca(initialization_type: str, std: float | str) -> FSDP:
    """load coca model from config and fsdp-wrap it"""
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/coca_config_initialization.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config_dict = _replace_config_dict(config_dict, initialization_type, std)

    coca_model = get_model_from_config(config_dict)
    coca_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        coca_model,
        sync_module_states=True,
        block_names=["TransformerBlock", "VisionTransformerBlock"],
        mixed_precision_settings=MixedPrecisionSettings.FP_16,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return coca_wrapped_model


def _load_model(model_name: str, initialization: str = "plain", std: float | str = 0.02) -> FSDP:
    """load gpt2 or coca model from config and fsdp-wrap it"""
    if model_name == "gpt2":
        model = _load_gpt2(initialization_type=initialization, std=std)
    elif model_name == "coca":
        model = _load_coca(initialization_type=initialization, std=std)
    else:
        raise Exception(f"model = {model_name} not implemented.")
    return model


# REGEX EXPRESSIONS THAT DEFINE INITIALIZATION GROUPS
INITIALIZATION_GROUPS = ["embedding", "weight-normal", "weight-projection", "weight-norm", "bias", "other"]
MAPPING_GPT2 = {
    "embedding": [r"wte.weight$", r"wpe.weight$"],
    "weight-projection": [r"c_proj\.weight$"],
    "weight-norm": [r"norm\.weight$"],
    "weight-normal": [r"\.weight$"],
    "bias": [r"\.bias$"],
    "other": [],
}
MAPPING_COCA = {
    "embedding": [],  # TODO
    "weight-projection": [r"c_proj\.weight$"],  # TODO
    "weight-norm": [r"norm[12]\.weight$", r"ln_[1234f]\.weight$"],  # TODO
    "weight-normal": [r"\.weight$"],
    "other": [r"conv", r".*(?<!bias)$"],  # (contains conv) or (does not end with .bias)
    "bias": [r".bias$"],
}


def get_group_params(model: FSDP, model_name: str) -> Dict[str, Optional[torch.Tensor]]:
    """
    divide all model parameters into initialization groups
    """
    if model_name == "gpt2":
        mapping = MAPPING_GPT2
    elif model_name == "coca":
        mapping = MAPPING_COCA
    else:
        raise Exception(f"Model = {model_name} not implemented.")

    params = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}

    group_params = {}
    excluded_regex_expressions = []
    for group_name, regex_expressions in mapping.items():
        list_of_flattened_params = [
            torch.flatten(parameter.detach())
            for name, parameter in params.items()
            if any([bool(re.search(regex_expression, name)) for regex_expression in regex_expressions])
            and not any(
                [
                    bool(re.search(excluded_regex_expression, name))
                    for excluded_regex_expression in excluded_regex_expressions
                ]
            )
        ]
        group_params[group_name] = torch.cat(list_of_flattened_params) if len(list_of_flattened_params) else None
        excluded_regex_expressions.extend(regex_expressions)
    return group_params


# NUMBER OF PARAMETERS
GPT2_NLAYERS = 12
GPT2_FFN_HIDDEN = 2048
GPT2_VOCAB_SIZE = 50304
GPT2_SEQUENCE_LENGTH = 2048
GPT2_HIDDEN_DIM = 768
GPT2_ALL = 106375680
GPT2_EMBEDDING = GPT2_HIDDEN_DIM * (
    GPT2_VOCAB_SIZE + GPT2_SEQUENCE_LENGTH
)  # parameters for token embeddings and positional embeddings
GPT2_WEIGHT_PROJECTION = (
    GPT2_HIDDEN_DIM * GPT2_HIDDEN_DIM + GPT2_HIDDEN_DIM * GPT2_FFN_HIDDEN
) * GPT2_NLAYERS  # 25952256
GPT2_WEIGHT_NORM = GPT2_HIDDEN_DIM * (GPT2_NLAYERS * 2 + 1)  # second term = num_layer_norms = (12*2+1) = 25
GPT2_BIAS = 89856
GPT2_OTHER = 0
GPT2_WEIGHT_NORMAL = GPT2_ALL - GPT2_WEIGHT_PROJECTION - GPT2_EMBEDDING - GPT2_WEIGHT_NORM - GPT2_BIAS  # 40107264

COCA_NLAYERS = 6 + 6  # text + multimodal
COCA_ALL = 184502784
COCA_EMBEDDING = 0  # TODO
COCA_WEIGHT_PROJECTION = 14745600
COCA_WEIGHT_NORM = 34560
COCA_BIAS = 191232
COCA_OTHER = 198912
COCA_WEIGHT_NORMAL = 169332480

NR_PARAMETERS = {
    "gpt2": {
        "all": GPT2_ALL,
        "embedding": GPT2_EMBEDDING,
        "weight-projection": GPT2_WEIGHT_PROJECTION,
        "weight-norm": GPT2_WEIGHT_NORM,
        "weight-normal": GPT2_WEIGHT_NORMAL,
        "bias": GPT2_BIAS,
        "other": GPT2_OTHER,
    },
    "coca": {
        "all": COCA_ALL,
        "embedding": COCA_EMBEDDING,
        "weight-normal": COCA_WEIGHT_NORMAL,
        "weight-projection": COCA_WEIGHT_PROJECTION,
        "weight-norm": COCA_WEIGHT_NORM,
        "bias": COCA_BIAS,
        "other": COCA_OTHER,
    },
}


# THEORETICAL AVERAGES AND STANDARD DEVIATIONS
def get_avg_theory(group: str) -> float:
    if group == "weight-norm":
        return 1.0
    else:
        return 0.0


def get_std_theory(group: str, initialization: str, model_name: str, std: float | str) -> float:
    if std == "auto":
        if model_name == "gpt2":
            std = math.sqrt(2 / (5 * GPT2_HIDDEN_DIM))
        else:
            raise Exception(f"std_theory not implemented for model_name = {model_name} and std = auto")

    if group in ["weight-norm", "bias"]:
        return 0.0
    elif group == "weight-normal":
        return std
    elif group == "weight-projection":
        if initialization == "plain":
            return std
        elif initialization in ["scaled", "scaled_embed"]:
            if model_name == "gpt2":
                return std / math.sqrt(2 * GPT2_NLAYERS)
            elif model_name == "coca":
                return std / math.sqrt(2 * COCA_NLAYERS)
            else:
                raise Exception(f"std_theory not implemented for model_name = {model_name}")
        else:
            raise Exception(f"std_theory not implemented for initialization = {initialization}")
    elif group == "embedding":
        if initialization == "scaled_embed":
            return math.sqrt(0.4)  # see https://arxiv.org/abs/2312.16903
        else:
            return std
    else:
        raise Exception(f"std_theory not implemented for group = {group}")


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "model_name",
    [
        ("gpt2"),
        ("coca"),
    ],
)
def test_nr_parameters_per_initialization_group(model_name):
    """
    verifies that, for a given model architecture,
    the different model parameter initialization groups
    have the expected number of parameters
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        model = _load_model(model_name)
        print(model)  # for debugging

        group_params = get_group_params(model, model_name)

        nr_parameters_all = 0
        for group in INITIALIZATION_GROUPS:
            # check number of parameters in each group
            nr_parameters_group = len(group_params[group]) if group_params[group] is not None else 0
            assert nr_parameters_group == NR_PARAMETERS[model_name][group], (
                f"nr_parameters for {model_name}/{group} = {nr_parameters_group} "
                + f"should be {NR_PARAMETERS[model_name][group]}"
                + f"Actual - Expected: {nr_parameters_group - NR_PARAMETERS[model_name][group]}"
            )
            nr_parameters_all += nr_parameters_group

        # check total number of parameters
        assert nr_parameters_all == NR_PARAMETERS[model_name]["all"], (
            f"total number of parameters for {model_name} = {nr_parameters_all} "
            + f"should be {NR_PARAMETERS[model_name]['all']}"
        )


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "model_name, initialization, std, success",
    [
        # std = 0.02
        ("gpt2", "plain", 0.02, True),
        ("gpt2", "scaled", 0.02, True),
        ("gpt2", "scaled_embed", 0.02, True),
        ("coca", "plain", 0.02, True),
        ("coca", "scaled", 0.02, False),  # scaled not implemented for coca
        ("coca", "scaled_embed", 0.02, False),  # scaled_embed not implemented for coca
        # std = 'auto'
        ("gpt2", "plain", "auto", True),
        ("gpt2", "scaled", "auto", True),
        ("gpt2", "scaled_embed", "auto", True),
        ("coca", "plain", "auto", False),  # auto not implemented for coca
        ("coca", "scaled", "auto", False),  # auto not implemented for coca
        ("coca", "scaled_embed", "auto", False),  # scaled_embed not implemented for coca
    ],
)
def test_statistical_distribution_for_each_initialization_group(
    model_name: str, initialization: str, std: float | str, success: bool
):
    """
    verifies that, for a given model architectrue and a given initialization,
    the different model parameter initialization groups
    have the expected avg and std
    """
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        if not success:
            with pytest.raises(Exception):
                model = _load_model(model_name, initialization, std)
        else:
            model = _load_model(model_name, initialization, std)

            print(model)  # for debugging

            group_params = get_group_params(model, model_name)

            for group in INITIALIZATION_GROUPS:
                # check mean and std for each group
                if group != "other" and group_params[group] is not None:
                    avg_test = torch.mean(group_params[group])
                    std_test = torch.std(group_params[group])
                    avg_theory = torch.tensor(get_avg_theory(group), device=avg_test.device, dtype=avg_test.dtype)
                    std_theory = torch.tensor(
                        get_std_theory(group, initialization, model_name, std),
                        device=std_test.device,
                        dtype=std_test.dtype,
                    )
                    torch.testing.assert_close(
                        avg_test,
                        avg_theory,
                        msg=f"average for {model_name}/{group} = {avg_test} should be close to {avg_theory}",
                        atol=3e-4,  # default value for torch.float32: 1e-5 (see https://pytorch.org/docs/stable/testing.html)
                        rtol=0,  # default value for torch.float32: 1.3e-6
                    )
                    torch.testing.assert_close(
                        std_test,
                        std_theory,
                        msg=f"standard deviation for {model_name}/{group} = {std_test} should be close to {std_theory}",
                        atol=2e-4,  # default value for torch.float32: 1e-5 (see https://pytorch.org/docs/stable/testing.html)
                        rtol=0,  # default value for torch.float32: 1.3e-6
                    )
