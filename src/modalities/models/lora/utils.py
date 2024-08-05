import copy
from typing import Union, List

from torch import nn

from modalities.models.lora.lora_layers import (
    LoRAEmbedding,
    LoRALinear,
    LoRALayer,
    LoRAConv1d,
    LoRAConv2d,
    LoRAConv3d,
)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def convert_to_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    list_allowed_conversion_types: List[str],
):
    recursive_layer_conversion(model, r, alpha, list_allowed_conversion_types)
    mark_only_lora_as_trainable(model=model)
    return model


def recursive_layer_conversion(
    module: nn.Module,
    r: int,
    alpha: int,
    list_allowed_conversion_types: List[str],
):
    for name, child in module.named_children():
        # If it's a leaf module (i.e., has no children), replace it with Linear
        if len(list(child.children())) == 0:
            if (
                type(child).__name__ in list_allowed_conversion_types
                or type(module).__name__ in list_allowed_conversion_types
            ):
                converted_child = convert_layer(child, r=r, alpha=alpha)
                setattr(module, name, converted_child)
        else:
            # Recursively apply to child modules
            recursive_layer_conversion(child, r, alpha, list_allowed_conversion_types)


def convert_layer(layer: nn.Module, r: int, alpha: int) -> nn.Module:
    if isinstance(layer, nn.Embedding):
        result = convert_embedding(layer, r, alpha)
    elif isinstance(layer, nn.Linear):
        result = convert_linear(layer, r, alpha)
    elif (
        isinstance(layer, nn.Conv1d)
        or isinstance(layer, nn.Conv2d)
        or isinstance(layer, nn.Conv3d)
    ):
        result = convert_convXd(layer, r, alpha)
    else:
        # todo log
        print(f"{layer} was not converted.")
        return layer
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


def convert_convXd(
    convXd_layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d], r: int, alpha: int
) -> Union[LoRAConv1d, LoRAConv2d, LoRAConv3d]:
    conv_to_lora_conv = {
        nn.Conv1d: LoRAConv1d,
        nn.Conv2d: LoRAConv2d,
        nn.Conv3d: LoRAConv3d,
    }
    if type(convXd_layer) not in conv_to_lora_conv:
        raise TypeError(f"{type(convXd_layer)} is not supported!")
    lora_conv_class = conv_to_lora_conv[type(convXd_layer)]
    lora_convXd = lora_conv_class(
        in_channels=convXd_layer.in_channels,
        out_channels=convXd_layer.out_channels,
        kernel_size=convXd_layer.kernel_size,
        r=r,
        lora_alpha=alpha,
    )
    lora_convXd.conv = copy.deepcopy(convXd_layer)
    return lora_convXd


def transfer_default_attributes(
    reference_layer: nn.Module, result_layer: nn.Module
) -> nn.Module:
    result_layer.training = reference_layer.training
    result_layer.dump_patches = reference_layer.dump_patches
    result_layer.call_super_init = reference_layer.call_super_init
    return result_layer
