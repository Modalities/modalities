"""
usage: convert_gpt2.py [-h] [--num_testruns NUM_TESTRUNS] [--device_modalities DEVICE_MODALITIES]
                       [--device_hf DEVICE_HF] modalities_config output_dir

Convert GPT-2 model checkpoint to Huggingface transformers format.

positional arguments:
  modalities_config     Path to the modalities config file.
  output_dir            Directory to save the converted model.

options:
  -h, --help            show this help message and exit
  --num_testruns NUM_TESTRUNS
                        Number of test runs to perform.
  --device_modalities DEVICE_MODALITIES
                        Device for the modalities model.
  --device_hf DEVICE_HF
                        Device for the Hugging Face model.
"""

import argparse
import shutil

import torch
import torch.nn as nn
from tqdm import tqdm

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block, PositionTypes
from modalities.models.model import SwiGLU
from modalities.models.utils import ModelTypeEnum, get_model_from_config


def convert_gpt2(
    modalities_config_path: str,
    output_dir: str,
    num_testruns: int = 0,
    device_modalities: str = "cpu",
    device_hf: str = "cpu",
) -> None:
    """Takes a modalities gpt2 model and converts it to a Huggingface transformers model.
       The provided config yaml file should contain the model_raw section with the model configuration.
       Additionally, the checkpointed_model section should be present and contain the path to the model checkpoint.
       Optionally, the function can run a number of test runs to compare the converted model with the original one.

    Args:
        modalities_config_path (str): Path to the modalities config file.
        output_dir (str): Directory to save the converted model.
        num_testruns (int, optional): Number of test runs to perform. Defaults to 0.
        device_modalities (str, optional): Device for the modalities model. Defaults to "cpu".
        device_hf (str, optional): Device for the Hugging Face model. Defaults to "cpu".
    """
    modalities_config = load_app_config_dict(modalities_config_path)
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)

    if num_testruns > 0:
        test_converted_model(
            hf_model.to(device_hf),
            modalities_model.to(device_modalities),
            num_testruns,
            modalities_config["model_raw"]["config"]["vocab_size"],
        )

    hf_model.config.auto_map = {
        "AutoConfig": "configuration_gpt2.GPT2Config",
        "AutoModel": "modeling_gpt2.GPT2Model",
        "AutoModelForCausalLM": "modeling_gpt2.GPT2ForCausalLM",
    }
    hf_model.save_pretrained(output_dir)
    _transfer_model_code(output_dir)


def convert_model_checkpoint(modalities_config: dict) -> tuple[GPT2ForCausalLM, GPT2LLM]:
    """Converts the modalities model to a Huggingface transformers model.
       Both the loaded modalities model and the converted Huggingface model are returned
       so that they can be compared.

    Args:
        modalities_config (dict): Modalities config dictionary.

    Returns:
        tuple[GPT2ForCausalLM, GPT2LLM]: Converted Hugging Face model and the original modalities model.
    """
    gpt2_config = convert_model_config(modalities_config)
    hf_model = GPT2ForCausalLM(gpt2_config).to(dtype=torch.bfloat16)
    modalities_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    _copy_weights_model(hf_model, modalities_model)
    return hf_model, modalities_model


def convert_model_config(modalities_config: dict) -> GPT2Config:
    """Converts the modalities model configuration to a Huggingface transformers configuration.
       For this the model_raw section of the modalities config is used.
       Corresponding entries are mapped to the Huggingface configuration.

    Args:
        modalities_config (dict): Modalities config dictionary.

    Returns:
        GPT2Config: Converted Huggingface model configuration.
    """
    config = modalities_config["model_raw"]["config"]

    assert config["poe_type"] == PositionTypes.NOPE
    assert config["activation_type"] == "swiglu"

    return GPT2Config(
        vocab_size=config["vocab_size"],
        hidden_size=config["n_embd"],
        pad_token_id=None,
        num_hidden_layers=config["n_layer"],
        num_key_value_heads=config["n_head_kv"],
        num_attention_heads=config["n_head_q"],
        intermediate_size=SwiGLU._get_hidden_dim(ffn_hidden=config["ffn_hidden"]),
        mlp_bias=config["bias"],
        hidden_act="silu",
        layer_norm_eps=config["ffn_norm"]["config"]["eps"],
        layer_norm_elementwise_affine=config["ffn_norm"]["config"].get(
            "elementwise_affine",
            True,
            # TODO:
            # Temporary solution: double-check that these are the correct default values.
            # Permanent solution: read default values from where they are defined.
        ),
        layer_norm_bias=config["ffn_norm"]["config"].get("bias", True),  # TODO: see comment above
        max_position_embeddings=config["sequence_length"],
        rope_theta=config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
        _attn_implementation="sdpa",
        output_attentions=False,
    )


def test_converted_model(hf_model: GPT2ForCausalLM, modalities_model: GPT2LLM, num_testruns: int, vocab_size: int):
    """Tests the converted model by inputting a random token sequence and comparing the output logits of both models.

    Args:
        hf_model (GPT2ForCausalLM): Huggingface transformers model.
        modalities_model (GPT2LLM): Modalities model.
        num_testruns (int): Number of test runs to perform.
        vocab_size (int): Vocabulary size of the model. (Required for generating random input tokens.)
    """
    for _ in tqdm(range(num_testruns), desc="Testing converted model"):
        input_ids = torch.randint(0, vocab_size, (1, 1024), device=hf_model.device)
        inputs = {modalities_model.sample_key: input_ids.to(modalities_model.transformer.wte.weight.device)}

        with torch.no_grad():
            llama_logits = hf_model(input_ids=input_ids).logits.to("cpu")
            modalities_logits = modalities_model(inputs)[modalities_model.prediction_key].to("cpu")

        assert llama_logits.shape == modalities_logits.shape
        assert torch.equal(llama_logits, modalities_logits)


def _copy_weights_model(hf_model_model: GPT2ForCausalLM, modalities_model: GPT2LLM):
    """Copies the weights of the modalities model to the Huggingface transformers model.

    Args:
        hf_model_model (GPT2ForCausalLM): The uninitialized Huggingface transformers model.
                                          The weights will be copied here.
        modalities_model (GPT2LLM): The modalities model from which the weights will be copied.
    """
    hf_model_model.model.embed_tokens.weight.data.copy_(modalities_model.transformer.wte.weight.data)
    for hf_layer, modalities_layer in zip(hf_model_model.model.layers, modalities_model.transformer.h):
        _copy_weights_attention(hf_layer, modalities_layer)
        _copy_weights_mlp(hf_layer, modalities_layer)
        _copy_weights_layer_norms(hf_layer, modalities_layer)
    _copy_weights_base_modules(hf_model_model.lm_head, modalities_model.lm_head)
    _copy_weights_base_modules(hf_model_model.model.norm, modalities_model.transformer.lm_head_norm)


def _copy_weights_attention(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    _copy_weights_base_modules(hf_layer.self_attn.q_proj, modalities_layer.attn.q_attn)
    _copy_weights_base_modules(hf_layer.self_attn.k_proj, modalities_layer.attn.k_attn)
    _copy_weights_base_modules(hf_layer.self_attn.v_proj, modalities_layer.attn.v_attn)
    _copy_weights_base_modules(hf_layer.self_attn.o_proj, modalities_layer.attn.c_proj)


def _copy_weights_mlp(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    _copy_weights_base_modules(hf_layer.mlp.down_proj, modalities_layer.mlp.W_2)
    _copy_weights_base_modules(hf_layer.mlp.gate_proj, modalities_layer.mlp.W)
    _copy_weights_base_modules(hf_layer.mlp.up_proj, modalities_layer.mlp.V)


def _copy_weights_layer_norms(hf_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    _copy_weights_base_modules(hf_layer.input_layernorm, modalities_layer.attention_norm)
    _copy_weights_base_modules(hf_layer.post_attention_layernorm, modalities_layer.ffn_norm)


def _copy_weights_base_modules(m1: nn.Linear | nn.LayerNorm, m2: nn.Linear | nn.LayerNorm):
    assert m1.weight.shape == m2.weight.shape
    assert (m1.bias is None and m2.bias is None) or m1.bias.shape == m2.bias.shape
    m1.weight.data.copy_(m2.weight.data)
    if m1.bias is not None:
        m1.bias.data.copy_(m2.bias.data)


def _transfer_model_code(output_dir: str):
    """Copies the required model code to the output directory.
       This allows the converted model to be used without the modalities package via:
       >>> from transformers import AutoModelForCausalLM
       >>> model = AutoModelForCausalLM.from_pretrained("path/to/converted/model", trust_remote_code=True)

    Args:
        output_dir (str): Directory of the converted model.
    """
    source_dir = os.path.dirname(__file__)
    modeling_gpt2_path = os.path.join(source_dir, "modeling_gpt2.py")
    configuration_gpt2_path = os.path.join(source_dir, "configuration_gpt2.py")
    shutil.copy(modeling_gpt2_path, output_dir)
    shutil.copy(configuration_gpt2_path, output_dir)

    target_modeling_file = os.path.join(output_dir, "modeling_gpt2.py")
    with open(target_modeling_file, "r") as file:
        content = file.read()
    content = content.replace("modalities.conversion.gpt2.configuration_gpt2", ".configuration_gpt2")
    with open(target_modeling_file, "w") as file:
        file.write(content)


if __name__ == "__main__":
    import os

    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    parser = argparse.ArgumentParser(description="Convert GPT-2 model checkpoint to Huggingface transformers format.")
    parser.add_argument("modalities_config", type=str, help="Path to the modalities config file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the converted model.")
    parser.add_argument("--num_testruns", type=int, default=0, help="Number of test runs to perform.")
    parser.add_argument("--device_modalities", type=str, default="cpu", help="Device for the modalities model.")
    parser.add_argument("--device_hf", type=str, default="cpu", help="Device for the Hugging Face model.")

    args = parser.parse_args()

    convert_gpt2(
        args.modalities_config,
        args.output_dir,
        args.num_testruns,
        args.device_modalities,
        args.device_hf,
    )
