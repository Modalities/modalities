import copy
from typing import List, Type

from torch import nn

from modalities.models.lora.lora_layers import LoRAEmbedding, LoRALinear


def convert_model(
    model: nn.Module, r: int, alpha: int, layer_types: List[Type[nn.Module]]
):
    # todo implement with different layer types
    # "attn", "linear", "embedding", "conv1d", "conv2d", "conv3d"
    old_head = model.lm_head
    model.lm_head = convert_layer(old_head, r, alpha)
    del old_head
    return model


def convert_layer(layer: nn.Module, r: int, alpha: int) -> nn.Module:
    if isinstance(layer, nn.Embedding):
        result = convert_embedding(layer, r, alpha)
    elif isinstance(layer, nn.Linear):
        result = convert_linear(layer, r, alpha)
    else:
        raise NotImplementedError(
            f"We cannot convert a {type(layer)} into a LoRA layer."
        )
    return transfer_default_attributes(layer, result)


def convert_embedding(
    embedding_layer: nn.Embedding, r: int, alpha: int
) -> LoRAEmbedding:
    lora_embedding = LoRAEmbedding(
        num_embeddings=embedding_layer.num_embeddings,
        embedding_dim=embedding_layer.embedding_dim,
        r=r,
        lora_alpha=alpha,
    )
    lora_embedding.weight = copy.deepcopy(embedding_layer.weight)
    return lora_embedding


def convert_linear(linear_layer: nn.Linear, r: int, alpha: int) -> LoRALinear:
    lora_linear = LoRALinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        r=r,
        lora_alpha=alpha,
    )
    lora_linear.weight = copy.deepcopy(linear_layer.weight)
    lora_linear.bias = copy.deepcopy(linear_layer.bias)
    return lora_linear


def transfer_default_attributes(
    reference_layer: nn.Module, result_layer: nn.Module
) -> nn.Module:
    result_layer.training = reference_layer.training
    result_layer.dump_patches = reference_layer.dump_patches
    result_layer.call_super_init = reference_layer.call_super_init
    return result_layer
