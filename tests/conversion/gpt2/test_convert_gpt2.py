from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from modalities.config.config import ConfigDictType, ProcessGroupBackendType, load_app_config_dict
from modalities.conversion.gpt2.conversion_model import (
    _build_single_node_dcp_config,
    check_converted_dcp_model,
    check_converted_model,
)
from modalities.conversion.gpt2.convert_gpt2 import convert_gpt2, convert_gpt2_dcp
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.running_env.cuda_env import MultiProcessingCudaEnv
from tests.conversion.gpt2.helper import check_same_weight_model


def test_converting_gpt2_does_not_change_weights(converted_model: PreTrainedModel, original_model: GPT2LLM):
    check_same_weight_model(converted_model, original_model)


def test_converting_gpt2_does_not_change_outputs(
    converted_model: PreTrainedModel, original_model: GPT2LLM, vocab_size: int
):
    check_converted_model(
        hf_model=converted_model, modalities_model=original_model, num_testruns=1, vocab_size=vocab_size
    )


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 8 GPUs.")
def test_converting_dcp_gpt2_does_not_change_weights(converted_dcp_model: PreTrainedModel, dcp_checkpoint: str):
    new_config: ConfigDictType = _build_single_node_dcp_config(dcp_checkpoint)
    with MultiProcessingCudaEnv(ProcessGroupBackendType.nccl, 0, 0, 1, 24570, device_id=0):
        modalities_model = get_model_from_config(new_config, model_type=ModelTypeEnum.DCP_CHECKPOINTED_MODEL)
        check_same_weight_model(converted_dcp_model, modalities_model)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 8 GPUs.")
def test_converting_dcp_gpt2_does_not_change_outputs(run_convert_gpt2_dcp: None, output_dir: Path, dcp_checkpoint: str):
    check_converted_dcp_model(
        hf_model_dir=str(output_dir), dcp_dir=dcp_checkpoint, num_testruns=1, device_id_modalities=0, device_hf="cuda:1"
    )


@pytest.fixture
def converted_model(run_convert_gpt2: None, output_dir: Path) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True).to(
        dtype=torch.bfloat16
    )


@pytest.fixture
def converted_dcp_model(run_convert_gpt2_dcp: None, output_dir: Path) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)


@pytest.fixture
def run_convert_gpt2(gpt2_config_path: Path, output_dir: Path):
    convert_gpt2(str(gpt2_config_path), str(output_dir))


@pytest.fixture
def run_convert_gpt2_dcp(dcp_checkpoint: str, output_dir: Path):
    convert_gpt2_dcp(dcp_checkpoint, str(output_dir))


@pytest.fixture
def original_model(gpt2_config_path: Path) -> GPT2LLM:
    modalities_config = load_app_config_dict(gpt2_config_path)
    return get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)


@pytest.fixture
def vocab_size(gpt2_config_path: Path) -> int:
    modalities_config = load_app_config_dict(gpt2_config_path)
    return modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"
