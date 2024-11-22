import os
from pathlib import Path
from typing import Optional

import pytest
import torch
from pydantic import BaseModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.types import Number

from modalities.__main__ import load_app_config_dict
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ProcessGroupBackendType, PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.model_factory import ModelFactory
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.utils.mfu import compute_mfu, get_theoretical_flops_per_token, get_theoretical_gpu_peak_performance
from tests.conftest import _ROOT_DIR

# NOTE: We need to run the tests in a torch distributed environment with 1 GPU.
# CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 \
#   $(which pytest) path/to/test_mfu.py


def get_model_from_config(model_config_dict: dict) -> GPT2LLM:
    """get gpt2 model from config_dict"""

    class InstantationModel(BaseModel):
        model: PydanticPytorchModuleType

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)

    components = component_factory.build_components(
        config_dict=model_config_dict, components_model_type=InstantationModel
    )

    model = components.model
    return model


def _load_gpt2(mixed_precision_settings: MixedPrecisionSettings) -> FSDP:
    """load gpt2 model from config and fsdp-wrap it"""
    config_file_path = _ROOT_DIR / Path("tests/test_yaml_configs/gpt2_config_mfu.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    # config_dict = _replace_config_dict(config_dict, _initialization_type='scaled', _std='auto')

    gpt2_model = get_model_from_config(config_dict)
    gpt2_wrapped_model = ModelFactory.get_fsdp_wrapped_model(
        gpt2_model,
        sync_module_states=True,
        block_names=["GPT2Block"],
        mixed_precision_settings=mixed_precision_settings,
        sharding_strategy=ShardingStrategy.NO_SHARD,
    )
    return gpt2_wrapped_model


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "mixed_precision_settings, world_size, expected_theoretical_gpu_peak_performance",
    [
        (MixedPrecisionSettings.BF_16, 2, 624e12),
        (MixedPrecisionSettings.FP_16, 2, 624e12),
        (MixedPrecisionSettings.BF_16, 10, 312e13),
        (MixedPrecisionSettings.FP_16, 10, 312e13),
        (MixedPrecisionSettings.BF_16_WORKING, 1, None),
        (MixedPrecisionSettings.FP_32, 1, None),
        (MixedPrecisionSettings.MIXED_PRECISION_MEGATRON, 1, None),
        (MixedPrecisionSettings.NO_MIXED_PRECISION, 1, None),
    ],
)
def test_get_theoretical_gpu_peak_performance(
    mixed_precision_settings: MixedPrecisionSettings,
    world_size: int,
    expected_theoretical_gpu_peak_performance: int,
):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        model = _load_gpt2(mixed_precision_settings)
        theoretical_gpu_peak_performance = get_theoretical_gpu_peak_performance(model, world_size)
        assert theoretical_gpu_peak_performance == expected_theoretical_gpu_peak_performance


# MODEL ARCHITECTURE FROM CONFIG
N_LAYER = 12
D_MODEL = 768
VOCAB_SIZE = 50304
SEQUENCE_LENGTH = 2048

#   LINEAR                      + EMBEDDING                                + LAYER NORM
N = 12 * N_LAYER * (D_MODEL**2) + (VOCAB_SIZE + SEQUENCE_LENGTH) * D_MODEL + (2 * N_LAYER + 1) * D_MODEL
ATTENTION = 12 * N_LAYER * D_MODEL * SEQUENCE_LENGTH
EXPECTED_THEORETICAL_FLOPS_PER_TOKEN = 6 * N + ATTENTION  # 977453568


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This test requires 1 GPU and a torchrun distributed environment.",
)
@pytest.mark.parametrize(
    "expected_theoretical_flops_per_token",
    [
        (EXPECTED_THEORETICAL_FLOPS_PER_TOKEN),
    ],
)
def test_get_theoretical_flops_per_token(
    expected_theoretical_flops_per_token: int,
):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        model = _load_gpt2(MixedPrecisionSettings.BF_16)
        theoretical_flops_per_token, _ = get_theoretical_flops_per_token(model)
        assert theoretical_flops_per_token == expected_theoretical_flops_per_token


@pytest.mark.parametrize(
    "num_samples_per_second, sequence_length, theoretical_flops_per_token, "
    "theoretical_gpu_peak_performance, expected_mfu",
    [
        (2, 4, 6, 8, 6.0),  # 2*4*6/8 = 6
        (2, 4, None, 8, -1.0),
        (2, 4, 6, None, -1.0),
        # 125M model, see 3rd last row here:
        # https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/benchmarking/README.md
        (532, 2048, EXPECTED_THEORETICAL_FLOPS_PER_TOKEN, 312e12 * 8, 0.4275),
    ],
)
def test_compute_mfu(
    num_samples_per_second: int,
    sequence_length: int,
    theoretical_flops_per_token: Optional[Number],
    theoretical_gpu_peak_performance: Optional[Number],
    expected_mfu: Number,
):
    mfu = compute_mfu(
        torch.tensor(num_samples_per_second),
        sequence_length,
        theoretical_flops_per_token,
        theoretical_gpu_peak_performance,
    )
    torch.testing.assert_close(mfu, torch.tensor(expected_mfu), atol=0.001, rtol=0)  # only absolute difference matters
