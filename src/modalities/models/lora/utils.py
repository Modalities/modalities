import copy

from torch import nn

from modalities.models.lora.lora_layers import LoRAEmbedding, LoRALinear, LoRALayer


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


def conversion_lora(
    model: nn.Module,
    r: int,
    alpha: int,
):
    replace_modules_in_attention(model=model, r=r, alpha=alpha)
    mark_only_lora_as_trainable(model=model)
    return model

def replace_modules_in_attention(
    model: nn.Module,
    r: int,
    alpha: int,
):
    # todo implement with different layer types
    # "attn", "linear", "embedding", "conv1d", "conv2d", "conv3d"
    # also implement with different model key name 'attn'
    for name, module in model.named_children():
        if "attention" in type(module).__name__.lower() and isinstance(module, nn.Module):
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear) or isinstance(sub_module, nn.Embedding):
                    new_linear = convert_layer(sub_module, r, alpha)
                    setattr(module, sub_name, new_linear)
        else:
            replace_modules_in_attention(module, r, alpha)

def convert_layer(layer: nn.Module, r: int, alpha: int) -> nn.Module:
    if isinstance(layer, nn.Embedding):
        result = convert_embedding(layer, r, alpha)
    elif isinstance(layer, nn.Linear):
        result = convert_linear(layer, r, alpha)
    else:
        # todo handle this
        return layer
        # raise NotImplementedError(
        #     f"We cannot convert a {type(layer)} into a LoRA layer."
        # )
    return transfer_default_attributes(layer, result)


def convert_embedding(embedding_layer: nn.Embedding, r: int, alpha: int) -> LoRAEmbedding:
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


def transfer_default_attributes(reference_layer: nn.Module, result_layer: nn.Module) -> nn.Module:
    result_layer.training = reference_layer.training
    result_layer.dump_patches = reference_layer.dump_patches
    result_layer.call_super_init = reference_layer.call_super_init
    return result_layer


if __name__ == "__main__":

    def recursively_transform(obj):
        if isinstance(obj, dict):
            return {key: recursively_transform(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [recursively_transform(element) for element in obj]
        elif isinstance(obj, int):
            return obj + 1
        else:
            return obj

    # Example usage:
    data = {"a": 1, "b": [2, {"c": 3}], "d": {"e": 4, "f": [5, 6]}}

    transformed_data = recursively_transform(data)
    print(transformed_data)
