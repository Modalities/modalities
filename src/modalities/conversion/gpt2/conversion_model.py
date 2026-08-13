import torch
import torch.nn as nn
from tqdm import tqdm

from modalities.conversion.gpt2.configuration_gpt2 import GPT2Config
from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.components.layer_norms import LayerNormConfig
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
    model_config = modalities_config["model_raw" if "model_raw" in modalities_config else "model"]
    checkpoint_path = None
    if "checkpointed_model" in modalities_config:
        checkpoint_path = modalities_config["checkpointed_model"].get("config", {}).get("checkpoint_path")

    if checkpoint_path and not ("variant_key" in modalities_config.get("checkpointed_model", {})):
        # Load state dict manually if variant_key is missing
        if "model" not in modalities_config and "model_raw" in modalities_config:
            modalities_config["model"] = modalities_config["model_raw"]
        modalities_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.MODEL)
        from pathlib import Path
        if Path(checkpoint_path).is_dir():
            from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
            from torch.distributed.checkpoint.filesystem import FileSystemReader
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
            sd = {}
            planner = _EmptyStateDictLoadPlanner(keys=["app.model"], allow_partial_load=True)
            _load_state_dict(sd, storage_reader=FileSystemReader(checkpoint_path), planner=planner, no_dist=True)
            model_sd = sd.get("app", {}).get("model", sd)
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_sd = ckpt
            for key in ("model_state_dict", "state_dict", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    model_sd = ckpt[key]
                    break
        
        out = {}
        for k, v in model_sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            out[k] = v
        missing, unexpected = modalities_model.load_state_dict(out, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        modalities_model = get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)
    _copy_weights_model(hf_model, modalities_model)
    return hf_model, modalities_model


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

    ffn_norm_key = "ffn_norm_config"
    norm_type = config[ffn_norm_key].get("norm_type", "layer_norm")

    qk_norm_cfg = config.get("attention_config", {}).get("qk_norm_config")
    use_qk_norm = qk_norm_cfg is not None
    qk_norm_dim = None
    if use_qk_norm:
        qk_cfg = qk_norm_cfg.get("config", {})
        qk_norm_dim = qk_cfg.get("ndim", qk_cfg.get("normalized_shape"))

    return GPT2Config(
        vocab_size=config["vocab_size"],
        hidden_size=config["n_embd"],
        pad_token_id=None,
        num_hidden_layers=config["n_layer"],
        num_key_value_heads=config["n_head_kv"],
        num_attention_heads=config["n_head_q"],
        intermediate_size=SwiGLU._get_hidden_dim(
            ffn_hidden=config["ffn_hidden"], enforce_swiglu_hidden_dim_multiple_of=256
        ),
        attention_bias=config["bias"],
        mlp_bias=config["bias"],
        hidden_act="silu",
        norm_type=norm_type,
        layer_norm_eps=_get_layer_norm_value(config[ffn_norm_key]["config"], "eps"),
        layer_norm_elementwise_affine=_get_layer_norm_value(config[ffn_norm_key]["config"], "elementwise_affine"),
        layer_norm_bias=_get_layer_norm_value(config[ffn_norm_key]["config"], "bias"),
        max_position_embeddings=config["sequence_length"],
        rope_theta=config["attention_config"]["qkv_transforms"][0]["config"]["base_freq"],
        _attn_implementation=_map_attention_type(config),
        use_qk_norm=use_qk_norm,
        qk_norm_dim=qk_norm_dim,
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

        modalities_model.to(dtype=hf_model.dtype, device=hf_model.device)
        with torch.no_grad():
            llama_logits = hf_model(input_ids=input_ids).logits.to("cpu")
            modalities_logits = modalities_model(inputs)[modalities_model.prediction_key].to("cpu")

        assert llama_logits.shape == modalities_logits.shape
        assert torch.equal(llama_logits, modalities_logits)


def _check_conversion_criteria(model_config: dict) -> None:
    """Checks that the modalities config fulfills criteria necessary for conversion

    Args:
        model_config (dict): model or model_raw part of the Modalities config dictionary.

    Returns:
        None
    """
    assert model_config["poe_type"] == PositionTypes.NOPE
    assert model_config["activation_type"] == "swiglu"
    assert model_config["attention_implementation"] in ["pytorch_flash", "manual"]

    norms = ["attention_norm_config", "ffn_norm_config", "lm_head_norm_config"]
    for norm in norms:
        assert model_config[norm]["norm_type"] in ["layer_norm", "rms_norm", "pytorch_rms_norm"]

    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "bias") for norm in norms)) == 1
    ), "All norms must have the same bias setting."
    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "elementwise_affine") for norm in norms)) == 1
    ), "All norms must have the same elementwise_affine setting."
    assert (
        len(set(_get_layer_norm_value(model_config[norm]["config"], "eps") for norm in norms)) == 1
    ), "All norms must have the same eps setting."


def _get_layer_norm_value(config: dict, field: str) -> bool | float | int:
    default = LayerNormConfig.model_fields[field].default
    return config.get(field, default)


def _map_attention_type(config: dict):
    impl = config.get("attention_implementation", "default")
    if impl in ("pytorch_flash", "default"):
        return "sdpa"
    elif impl == "manual":
        return "eager"
    else:
        raise ValueError(f"Unknown attention_implementation: {impl}")


def _copy_weights_model(hf_model: GPT2ForCausalLM, modalities_model: GPT2LLM):
    """Copies the weights of the modalities model to the Huggingface transformers model.

    Args:
        hf_model (GPT2ForCausalLM): The uninitialized Huggingface transformers model.
                                    The weights will be copied here.
        modalities_model (GPT2LLM): The modalities model from which the weights will be copied.
    """
    hf_model.model.embed_tokens.weight.data.copy_(modalities_model.transformer.wte.weight.data)
    for hf_layer, modalities_layer_idx in zip(hf_model.model.layers, modalities_model.transformer.h):
        _copy_weights_attention(hf_layer, modalities_model.transformer.h[modalities_layer_idx])
        _copy_weights_mlp(hf_layer, modalities_model.transformer.h[modalities_layer_idx])
        _copy_weights_layer_norms(hf_layer, modalities_model.transformer.h[modalities_layer_idx])
    _copy_weights_base_modules(hf_model.lm_head, modalities_model.transformer.lm_head)
    _copy_weights_base_modules(hf_model.model.norm, modalities_model.transformer.lm_head_norm)


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
    if getattr(hf_layer.self_attn, "q_norm", None) is not None:
        _copy_weights_base_modules(hf_layer.self_attn.q_norm, modalities_layer.attn.q_norm)
        _copy_weights_base_modules(hf_layer.self_attn.k_norm, modalities_layer.attn.k_norm)


def _copy_weights_base_modules(m1: nn.Linear | nn.LayerNorm | nn.Module, m2: nn.Linear | nn.LayerNorm | nn.Module):
    assert m1.weight.shape == m2.weight.shape
    m1_bias = getattr(m1, "bias", None)
    m2_bias = getattr(m2, "bias", None)
    assert (m1_bias is None and m2_bias is None) or m1_bias.shape == m2_bias.shape
    m1.weight.data.copy_(m2.weight.data)
    if m1_bias is not None:
        m1_bias.data.copy_(m2_bias.data)
