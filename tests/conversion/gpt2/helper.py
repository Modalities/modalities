import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from modalities.conversion.gpt2.modeling_gpt2 import GPT2DecoderLayer, GPT2ForCausalLM
from modalities.models.gpt2.gpt2_model import GPT2LLM, GPT2Block


@torch.no_grad()
def check_same_weight_model(converted_model: GPT2ForCausalLM, modalities_model: GPT2LLM):
    converted_model.to(device=modalities_model.transformer.h["0"].attn.q_attn.weight.device)
    assert torch.equal(converted_model.model.embed_tokens.weight, to_local(modalities_model.transformer.wte.weight))
    for i, (llama_layer, modalities_layer_idx) in enumerate(
        zip(converted_model.model.layers, modalities_model.transformer.h)
    ):
        check_same_weight_attention(llama_layer, modalities_model.transformer.h[modalities_layer_idx])
        check_same_weight_mlp(llama_layer, modalities_model.transformer.h[modalities_layer_idx])
        check_same_weight_layer_norms(llama_layer, modalities_model.transformer.h[modalities_layer_idx])
    check_same_weight_base_modules(converted_model.lm_head, modalities_model.transformer.lm_head)
    check_same_weight_base_modules(converted_model.model.norm, modalities_model.transformer.lm_head_norm)


def check_same_weight_attention(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.self_attn.q_proj, modalities_layer.attn.q_attn)
    check_same_weight_base_modules(llama_layer.self_attn.k_proj, modalities_layer.attn.k_attn)
    check_same_weight_base_modules(llama_layer.self_attn.v_proj, modalities_layer.attn.v_attn)
    check_same_weight_base_modules(llama_layer.self_attn.o_proj, modalities_layer.attn.c_proj)


def check_same_weight_mlp(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.mlp.down_proj, modalities_layer.mlp.W_2)
    check_same_weight_base_modules(llama_layer.mlp.gate_proj, modalities_layer.mlp.W)
    check_same_weight_base_modules(llama_layer.mlp.up_proj, modalities_layer.mlp.V)


def check_same_weight_layer_norms(llama_layer: GPT2DecoderLayer, modalities_layer: GPT2Block):
    check_same_weight_base_modules(llama_layer.input_layernorm, modalities_layer.attention_norm)
    check_same_weight_base_modules(llama_layer.post_attention_layernorm, modalities_layer.ffn_norm)


def check_same_weight_base_modules(l1: nn.Linear | nn.LayerNorm, l2: nn.Linear | nn.LayerNorm):
    assert torch.equal(l1.weight, to_local(l2.weight))
    assert (l1.bias is None and l2.bias is None) or torch.equal(l1.bias, to_local(l2.bias))


@torch.no_grad()
def to_local(tensor: torch.Tensor | DTensor) -> torch.Tensor:
    """Convert a tensor or distributed tensor to a local tensor."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor
