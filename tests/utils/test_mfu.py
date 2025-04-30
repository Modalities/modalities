import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel
from torch.types import Number

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import (
    PydanticFSDP1ModuleType,
    PydanticFSDP2ModuleType,
    PydanticMFUCalculatorABCType,
)
from modalities.running_env.env_utils import MixedPrecisionSettings, PyTorchDtypes
from modalities.utils.mfu import GPT2MFUCalculator
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


# MODEL ARCHITECTURE FROM CONFIG
N_LAYER = 12
D_MODEL = 768
VOCAB_SIZE = 50304
SEQUENCE_LENGTH = 2048

#   LINEAR                      + EMBEDDING                                + LAYER NORM
N = 12 * N_LAYER * (D_MODEL**2) + (VOCAB_SIZE + SEQUENCE_LENGTH) * D_MODEL + (2 * N_LAYER + 1) * D_MODEL
ATTENTION = 12 * N_LAYER * D_MODEL * SEQUENCE_LENGTH
EXPECTED_THEORETICAL_FLOPS_PER_TOKEN = 6 * N + ATTENTION  # 977453568


class TestMFU:
    @staticmethod
    def _load_yaml_config(config_file_path: Path) -> dict:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def _update_config_theoretical_gpu_peak_performance(
        config: dict,
        mixed_precision_settings: MixedPrecisionSettings,
    ) -> dict:
        if "device_mesh" in config:  # FSDP2
            config["fsdp_model"]["config"]["mixed_precision_settings"]["param_dtype"] = mixed_precision_settings.name
            config["fsdp_model"]["config"]["mixed_precision_settings"]["reduce_dtype"] = mixed_precision_settings.name
        else:  # FSDP1
            config["test_model"]["config"]["mixed_precision_settings"] = mixed_precision_settings.name
        return config

    @staticmethod
    def _update_config_test_compute_mfu(config: dict, sequence_length: int) -> dict:
        config["model_raw"]["config"]["sequence_length"] = sequence_length
        return config

    @staticmethod
    def _save_yaml_config(config_file_path: Path, config: dict):
        with open(config_file_path, "w") as f:
            yaml.safe_dump(config, f)

    @staticmethod
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test requires 2 GPUs.")
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path, mixed_precision_settings, world_size_fake, "
        "simulated_gpu_type, expected_theoretical_gpu_peak_performance, warning_msg",
        [
            # A100
            (
                22380,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.BF_16,
                2,
                "NVIDIA A100",
                624e12,
                None,
            ),
            (
                22381,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.FP_16,
                2,
                "NVIDIA A100",
                624e12,
                None,
            ),
            (
                22382,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.BF_16,
                10,
                "NVIDIA A100",
                312e13,
                None,
            ),
            (
                22383,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.FP_16,
                10,
                "NVIDIA A100",
                312e13,
                None,
            ),
            # we only support bf16 for FSDP2
            (
                22380,
                "../test_yaml_configs/gpt2_config_mfu_fsdp2.yaml",
                PyTorchDtypes.BF_16,
                2,
                "NVIDIA A100",
                624e12,
                "MFU is computed based on the assumption that bf16 precision is used.",
            ),
            (
                22380,
                "../test_yaml_configs/gpt2_config_mfu_fsdp2.yaml",
                PyTorchDtypes.FP_16,
                2,
                "NVIDIA A100",
                624e12,
                "MFU is computed based on the assumption that bf16 precision is used.",
            ),
            (
                22380,
                "../test_yaml_configs/gpt2_config_mfu_fsdp2.yaml",
                PyTorchDtypes.FP_32,
                2,
                "NVIDIA A100",
                624e12,
                "MFU is computed based on the assumption that bf16 precision is used.",
            ),
            # H100
            (
                22384,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.BF_16,
                2,
                "NVIDIA H100",
                1978e12,
                None,
            ),
            (
                22385,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.FP_16,
                2,
                "NVIDIA H100",
                1978e12,
                None,
            ),
            (
                22386,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.BF_16,
                10,
                "NVIDIA H100",
                989e13,
                None,
            ),
            (
                22387,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.FP_16,
                10,
                "NVIDIA H100",
                989e13,
                None,
            ),
            # unsupported precision
            (
                22388,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.BF_16_WORKING,
                1,
                "NVIDIA A100",
                None,
                None,
            ),
            (
                22389,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.FP_32,
                1,
                "NVIDIA A100",
                None,
                None,
            ),
            (
                22390,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.MIXED_PRECISION_MEGATRON,
                1,
                "NVIDIA A100",
                None,
                None,
            ),
            (
                22391,
                "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml",
                MixedPrecisionSettings.NO_MIXED_PRECISION,
                1,
                "NVIDIA A100",
                None,
                None,
            ),
        ],
    )
    def test_get_theoretical_gpu_peak_performance(
        rdvz_port: int,
        relative_config_path: str,
        temporary_folder_path: Path,
        mixed_precision_settings: MixedPrecisionSettings,
        world_size_fake: int,
        simulated_gpu_type: str,
        expected_theoretical_gpu_peak_performance: int,
        warning_msg: Optional[str],
    ):
        working_dir = Path(os.path.dirname(__file__))
        # load, update and save tmp config
        config_file_path = working_dir / relative_config_path
        config = TestMFU._load_yaml_config(config_file_path=config_file_path)
        config_updated = TestMFU._update_config_theoretical_gpu_peak_performance(
            config=config, mixed_precision_settings=mixed_precision_settings
        )
        tmp_config_file_path = temporary_folder_path / "config.yaml"
        TestMFU._save_yaml_config(config_file_path=tmp_config_file_path, config=config_updated)

        # run the test in a distributed environment
        world_size_actual = 2
        mp.spawn(
            TestMFU._test_get_theoretical_gpu_peak_performance_thread,
            args=(
                world_size_fake,
                world_size_actual,
                rdvz_port,
                tmp_config_file_path,
                simulated_gpu_type,
                expected_theoretical_gpu_peak_performance,
                warning_msg,
            ),
            nprocs=world_size_actual,
            join=True,
        )

    @staticmethod
    def _test_get_theoretical_gpu_peak_performance_thread(
        process_id: int,
        world_size_fake: int,
        world_size_actual: int,
        rdvz_port: int,
        tmp_config_file_path: Path,
        simulated_gpu_type: str,
        expected_theoretical_gpu_peak_performance: int,
        warning_msg: Optional[str],
    ):
        torch.cuda.get_device_name = Mock()
        torch.cuda.get_device_name.return_value = simulated_gpu_type

        class CustomComponentInstantiationModel(BaseModel):
            test_model: PydanticFSDP1ModuleType | PydanticFSDP2ModuleType

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size_actual,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(tmp_config_file_path)
            components: CustomComponentInstantiationModel = main_obj.build_components(
                components_model_type=CustomComponentInstantiationModel
            )
            wrapped_model = components.test_model
            if warning_msg is not None:
                with pytest.warns(UserWarning, match=warning_msg):
                    theoretical_gpu_peak_performance = GPT2MFUCalculator._get_theoretical_gpu_peak_performance(
                        wrapped_model, world_size_fake
                    )
            else:
                theoretical_gpu_peak_performance = GPT2MFUCalculator._get_theoretical_gpu_peak_performance(
                    wrapped_model, world_size_fake
                )
            assert theoretical_gpu_peak_performance == expected_theoretical_gpu_peak_performance

    @staticmethod
    @pytest.mark.parametrize(
        "expected_theoretical_flops_per_token",
        [
            (EXPECTED_THEORETICAL_FLOPS_PER_TOKEN),
        ],
    )
    def test_get_theoretical_flops_per_token(
        expected_theoretical_flops_per_token: int,
    ):
        theoretical_flops_per_token = GPT2MFUCalculator._get_theoretical_flops_per_token(
            num_params=N,
            n_layer=N_LAYER,
            sequence_length=SEQUENCE_LENGTH,
            n_embd=D_MODEL,
        )
        assert theoretical_flops_per_token == expected_theoretical_flops_per_token

    @staticmethod
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="This test requires 2 GPUs.")
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path, num_samples_per_second, expected_mfu",
        [
            # (2, 4, 6, 8, 6.0),  # 2*4*6/8 = 6
            # (2, 4, None, 8, -1.0),
            # (2, 4, 6, None, -1.0),
            # 125M model, see 3rd last row here:
            # https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md
            (22301, "../test_yaml_configs/gpt2_config_mfu_fsdp1.yaml", 532, 0.4275),
            (22302, "../test_yaml_configs/gpt2_config_mfu_fsdp2.yaml", 532, 0.4275),
        ],
    )
    def test_compute_mfu(
        rdvz_port: int,
        temporary_folder_path: Path,
        relative_config_path: str,
        num_samples_per_second: int,
        expected_mfu: Number,
    ):
        working_dir = Path(os.path.dirname(__file__))
        # load, update and save tmp config
        config_file_path = working_dir / relative_config_path
        config = TestMFU._load_yaml_config(config_file_path=config_file_path)
        config_updated = config  # TestMFU._update_config_test_compute_mfu(config=config)
        tmp_config_file_path = temporary_folder_path / "config.yaml"
        TestMFU._save_yaml_config(config_file_path=tmp_config_file_path, config=config_updated)

        # run the test in a distributed environment
        world_size = torch.cuda.device_count()
        mp.spawn(
            TestMFU._test_compute_mfu_thread,
            args=(rdvz_port, world_size, tmp_config_file_path, num_samples_per_second, expected_mfu),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_compute_mfu_thread(
        process_id: int,
        rdvz_port: int,
        world_size: int,
        tmp_config_file_path: Path,
        num_samples_per_second: int,
        expected_mfu: float,
    ):
        class CustomComponentInstantiationModel(BaseModel):
            mfu_calculator: PydanticMFUCalculatorABCType

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(tmp_config_file_path)
            components: CustomComponentInstantiationModel = main_obj.build_components(
                components_model_type=CustomComponentInstantiationModel
            )
            mfu_calculator = components.mfu_calculator
            mfu_value = mfu_calculator.compute(num_samples_per_second=torch.tensor(num_samples_per_second))

        torch.testing.assert_close(
            mfu_value, torch.tensor(expected_mfu), atol=0.001, rtol=0
        )  # only absolute difference matters
