import torch
import torch.nn as nn
from tqdm import tqdm

from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block, PositionTypes
from modalities.models.model import SwiGLU
from modalities.models.utils import ModelTypeEnum, get_model_from_config


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


def _check_conversion_criteria(model_config: dict) -> None:
    """Checks that the modalities config fulfills criteria necessary for conversion

    Args:
        model_config (dict): model or model_raw part of the Modalities config dictionary.

    Returns:
        None
    """
    assert model_config["poe_type"] == PositionTypes.NOPE
    assert model_config["bias"] is False
    assert model_config["activation_type"] == "swiglu"
    assert model_config["attention_implementation"] in ["pytorch_flash", "manual"]

    for norm in ["attention_norm", "ffn_norm", "lm_head_norm"]:
        assert model_config[norm]["variant_key"] == "layer_norm"
        assert model_config[norm]["config"].get("elementwise_affine", True) is True  # True = default setting
        assert model_config[norm]["config"].get("bias", True) is True  # True = default setting


def convert_model_config(modalities_config: dict) -> GPT2Config:
    """Converts the modalities model configuration to a Huggingface transformers configuration.
       For this the model_raw or model section of the modalities config is used.
       Corresponding entries are mapped to the Huggingface configuration.

    Args:
        modalities_config (dict): Modalities config dictionary.

    Returns:
        GPT2Config: Converted Huggingface model configuration.
    """
    config = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]

    _check_conversion_criteria(config)

    if config["attention_implementation"] == "pytorch_flash":
        attention_impl = "sdpa"
    elif config["attention_implementation"] == "manual":
        attention_impl = "eager"
    else:
        raise ValueError(f"Unknown or unsupported attention implementation {config['attention_implementation']}.")

    return GPT2Config(
        vocab_size=config["vocab_size"],
        hidden_size=config["n_embd"],
        pad_token_id=None,
        num_hidden_layers=config["n_layer"],
        num_key_value_heads=config["n_head_kv"],
        num_attention_heads=config["n_head_q"],
        intermediate_size=SwiGLU._get_hidden_dim(ffn_hidden=config["ffn_hidden"]),
        attention_bias=config["bias"],
        mlp_bias=config["bias"],
        hidden_act="silu",
        layer_norm_eps=config["ffn_norm"]["config"]["eps"],
        layer_norm_elementwise_affine=config["ffn_norm"]["config"].get(
            "elementwise_affine",
            True,
        ),
        layer_norm_bias=config["ffn_norm"]["config"].get("bias", True),
        max_position_embeddings=config["sequence_length"],
        rope_theta=config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
        _attn_implementation=attention_impl,
        output_attentions=False,
    )


def check_converted_model(hf_model: GPT2ForCausalLM, modalities_model: GPT2LLM, num_testruns: int, vocab_size: int):
    """Tests the converted model by inputting a random token sequence and comparing the output logits of both models.

    Args:
        hf_model (GPT2ForCausalLM): Huggingface transformers model.
        modalities_model (GPT2LLM): Modalities model.
        num_testruns (int): Number of test runs to perform.
        vocab_size (int): Vocabulary size of the model. (Required for generating random input tokens.)
    """
    for _ in tqdm(range(num_testruns), desc="Testing converted model"):
        input_ids = torch.randint(0, vocab_size, (1, modalities_model.sequence_length), device=hf_model.device)
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
