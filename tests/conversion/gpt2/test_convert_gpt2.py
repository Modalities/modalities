from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.conversion_model import check_converted_model
from modalities.conversion.gpt2.convert_gpt2 import convert_gpt2
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from tests.conversion.gpt2.helper import check_same_weight_model


def test_converting_gpt2_does_not_change_weights(tmp_path: Path, gpt2_config_path: str):
    output_dir = tmp_path / "output"
    convert_gpt2(gpt2_config_path, output_dir)
    modalities_config = load_app_config_dict(gpt2_config_path)
    original_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    converted_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    check_same_weight_model(converted_model, original_model)


def test_converting_gpt2_does_not_change_outputs(tmp_path: Path, gpt2_config_path: str):
    output_dir = tmp_path / "output"
    convert_gpt2(gpt2_config_path, output_dir)
    modalities_config = load_app_config_dict(gpt2_config_path)
    original_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    converted_model = AutoModelForCausalLM.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=True
    ).to(dtype=torch.bfloat16)
    vocab_size = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]
    check_converted_model(converted_model, original_model, 1, vocab_size)
