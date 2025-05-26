import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.fsdp import StateDictType

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticFSDP1ModuleType, PydanticFSDP2ModuleType
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@dataclass
class WeightInitFSDPX:
    weight_init_type: str
    std: float
    use_weight_tying: bool


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test requires 2 GPUs.")
class TestWeightInitFSDPX:
    GPT2_HIDDEN_DIM = 768
    GPT2_NLAYERS = 12

    # REGEX EXPRESSIONS THAT DEFINE INITIALIZATION GROUPS
    INITIALIZATION_GROUPS = ["embedding", "weight-normal", "weight-projection", "weight-norm", "bias", "other"]

    @staticmethod
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path, weight_init_params",
        [
            # FSDP1 with tied weights
            (
                22359,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "plain",
                    0.02,
                    True,
                ),
            ),
            (
                22360,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled",
                    0.02,
                    True,
                ),
            ),
            (
                22361,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    0.02,
                    True,
                ),
            ),
            (
                22362,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "plain",
                    "auto",
                    True,
                ),
            ),
            (
                22363,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled",
                    "auto",
                    True,
                ),
            ),
            (
                22364,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    "auto",
                    True,
                ),
            ),
            # FSDP1 without tied weights
            (
                22359,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "plain",
                    0.02,
                    False,
                ),
            ),
            (
                22360,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled",
                    0.02,
                    False,
                ),
            ),
            (
                22361,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    0.02,
                    False,
                ),
            ),
            (
                22362,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "plain",
                    "auto",
                    False,
                ),
            ),
            (
                22363,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled",
                    "auto",
                    False,
                ),
            ),
            (
                22364,
                "test_yaml_configs/gpt2_config_initialization_fsdp1.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    "auto",
                    False,
                ),
            ),
            # FSDP2 with tied weights
            (
                22365,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "plain",
                    0.02,
                    True,
                ),
            ),
            (
                22366,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled",
                    0.02,
                    True,
                ),
            ),
            (
                22367,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    0.02,
                    True,
                ),
            ),
            (
                22368,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "plain",
                    "auto",
                    True,
                ),
            ),
            (
                22369,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled",
                    "auto",
                    True,
                ),
            ),
            (
                22370,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    "auto",
                    True,
                ),
            ),
            # FSDP2 without tied weights
            (
                22365,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "plain",
                    0.02,
                    False,
                ),
            ),
            (
                22366,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled",
                    0.02,
                    False,
                ),
            ),
            (
                22367,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    0.02,
                    False,
                ),
            ),
            (
                22368,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "plain",
                    "auto",
                    False,
                ),
            ),
            (
                22369,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled",
                    "auto",
                    False,
                ),
            ),
            (
                22370,
                "test_yaml_configs/gpt2_config_initialization_fsdp2.yaml",
                WeightInitFSDPX(
                    "scaled_embed",
                    "auto",
                    False,
                ),
            ),
        ],
    )
    def test_weight_distribution(
        rdvz_port: int, relative_config_path: str, temporary_folder_path: Path, weight_init_params: WeightInitFSDPX
    ):
        working_dir = Path(os.path.dirname(__file__))
        # load, update and save tmp config
        config_file_path = working_dir / relative_config_path
        config = TestWeightInitFSDPX._load_yaml_config(config_file_path=config_file_path)
        config_updated = TestWeightInitFSDPX._update_config(config=config, weight_init_params=weight_init_params)
        tmp_config_file_path = temporary_folder_path / "config.yaml"
        TestWeightInitFSDPX._save_yaml_config(config_file_path=tmp_config_file_path, config=config_updated)

        # run the test in a distributed environment
        world_size = 2
        mp.spawn(
            TestWeightInitFSDPX._test_weight_init_thread,
            args=(world_size, rdvz_port, tmp_config_file_path, weight_init_params),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_weight_init_thread(
        process_id: int,
        world_size: int,
        rdvz_port: int,
        tmp_config_file_path: Path,
        weight_init_params: WeightInitFSDPX,
    ):
        class CustomComponentInstantiationModel(BaseModel):
            tested_model: PydanticFSDP1ModuleType | PydanticFSDP2ModuleType

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(tmp_config_file_path)
            # build the components (indluduing the custom component)
            components: CustomComponentInstantiationModel = main_obj.build_components(
                components_model_type=CustomComponentInstantiationModel
            )
            tested_model = components.tested_model

            # replicate all parameters on all ranks and run the tests
            if isinstance(tested_model, FSDP1):
                state_dict = TestWeightInitFSDPX._get_fdsp1_state_dict(model=tested_model)

            elif isinstance(tested_model, FSDP2):
                state_dict = TestWeightInitFSDPX._get_fdsp2_state_dict(model=tested_model)

            else:
                raise Exception(f"model type {type(tested_model)} not supported")

        TestWeightInitFSDPX.assert_correct_weight_distribution(
            state_dict=state_dict,
            weight_init_params=weight_init_params,
        )

    @staticmethod
    def assert_correct_weight_distribution(state_dict: dict[str, Any], weight_init_params: WeightInitFSDPX):
        # verifies that, for a given model (state_dict) and a given initialization,
        # the different model parameter initialization
        # group have the expected avg and std

        group_params = TestWeightInitFSDPX._get_group_params(state_dict=state_dict)
        for group in TestWeightInitFSDPX.INITIALIZATION_GROUPS:
            if group != "other" and group_params[group] is not None:
                avg_test = torch.mean(group_params[group])
                std_test = torch.std(group_params[group])
                avg_theory = torch.tensor(
                    TestWeightInitFSDPX._get_avg_theory(group), device=avg_test.device, dtype=avg_test.dtype
                )
                std_theory = torch.tensor(
                    TestWeightInitFSDPX._get_std_theory(
                        group=group,
                        initialization=weight_init_params.weight_init_type,
                        std=weight_init_params.std,
                    ),
                    device=std_test.device,
                    dtype=std_test.dtype,
                )
                torch.testing.assert_close(
                    avg_test,
                    avg_theory,
                    msg=f"average for {group} = {avg_test} should be close to {avg_theory}",
                    atol=3e-4,  # default value for torch.float32: 1e-5 (see https://pytorch.org/docs/stable/testing.html)
                    rtol=0,  # default value for torch.float32: 1.3e-6
                )
                torch.testing.assert_close(
                    std_test,
                    std_theory,
                    msg=f"standard deviation for {group} = {std_test} should be close to {std_theory}",
                    atol=2e-4,  # default value for torch.float32: 1e-5 (see https://pytorch.org/docs/stable/testing.html)
                    rtol=0,  # default value for torch.float32: 1.3e-6
                )
            if group == "other":
                # other group should be empty
                assert group_params[group] is None, f"other group should be empty, but got {group_params[group]}"

    @staticmethod
    def _load_yaml_config(config_file_path: Path) -> dict:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _save_yaml_config(config_file_path: Path, config: dict):
        with open(config_file_path, "w") as f:
            yaml.safe_dump(config, f)

    @staticmethod
    def _update_config(config: dict, weight_init_params: WeightInitFSDPX) -> dict:
        config["model_raw"]["config"]["n_embd"] = TestWeightInitFSDPX.GPT2_HIDDEN_DIM
        config["model_raw"]["config"]["n_layer"] = TestWeightInitFSDPX.GPT2_NLAYERS

        if "model_initializer" in config["tested_model"]["config"]:  # FSDP2 case
            initialized_model_config = config["tested_model"]["config"]
        else:  # FSDP1 case
            initialized_model_config = config["initialized_model"]["config"]

        initialized_model_config["model_initializer"]["config"][
            "weight_init_type"
        ] = weight_init_params.weight_init_type
        initialized_model_config["model_initializer"]["config"]["std"] = weight_init_params.std
        if weight_init_params.weight_init_type == "plain":
            initialized_model_config["model_initializer"]["config"]["num_layers"] = None  # replace
        if weight_init_params.std != "auto":
            initialized_model_config["model_initializer"]["config"]["hidden_dim"] = None  # replace

        config["model_raw"]["config"]["use_weight_tying"] = weight_init_params.use_weight_tying
        return config

    @staticmethod
    def _get_group_params(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """
        Divide all model parameters into initialization groups
        """
        mapping = {
            "embedding": [r"wte.weight$", r"wpe.weight$", r"lm_head.weight$"],
            "weight-projection": [r"c_proj\.weight$"],
            "weight-norm": [r"norm\.weight$"],
            "weight-normal": [r"\.weight$"],
            "bias": [r"\.bias$"],
            "other": [],
        }
        params = {name: parameter for name, parameter in state_dict.items()}

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

    @staticmethod
    def _get_avg_theory(group: str) -> float:
        # returns the expected average weight value for the given group
        if group == "weight-norm":
            return 1.0
        else:
            return 0.0

    @staticmethod
    def _get_std_theory(group: str, initialization: str, std: float | str) -> float:
        # returns the expected standard deviation of the weight values for the given group
        if std == "auto":
            std = math.sqrt(2 / (5 * TestWeightInitFSDPX.GPT2_HIDDEN_DIM))

        if group in ["weight-norm", "bias"]:
            return 0.0
        elif group == "weight-normal":
            return std
        elif group == "weight-projection":
            if initialization == "plain":
                return std
            elif initialization in ["scaled", "scaled_embed"]:
                return std / math.sqrt(2 * TestWeightInitFSDPX.GPT2_NLAYERS)
            else:
                raise Exception(f"std_theory not implemented for initialization = {initialization}")
        elif group == "embedding":
            if initialization == "scaled_embed":
                return math.sqrt(0.4)  # see https://arxiv.org/abs/2312.16903
            else:
                return std
        else:
            raise Exception(f"std_theory not implemented for group = {group}")

    @staticmethod
    def _get_fdsp1_state_dict(model: FSDP1) -> dict[str, Any]:
        # returns the state dict of the FSDP1 wrapped model
        # with the parameters replicated on all ranks
        model_save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
        with FSDP1.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=model_save_policy,
        ):
            model_state = model.state_dict()
        return model_state

    @staticmethod
    def _get_fdsp2_state_dict(model: FSDP2) -> dict[str, Any]:
        # returns the state dict of the FSDP2 wrapped model
        # with the parameters replicated on all ranks
        model_state = get_state_dict(
            model=model, optimizers=[], options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        )[0]
        return model_state
