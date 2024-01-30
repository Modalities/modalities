from enum import Enum
from typing import List


from pydantic import BaseModel, confloat, conint, model_validator

from modalities.config.lookup_types import QueryKeyValueTransformType

# GPT2 implementation taken from nanogpt https://github.com/karpathy/nanoGPT


class AttentionType(str, Enum):
    DEFAULT_ATTENTION = "default_attention"
    PYTORCH_FLASH_ATTENTION = "pytorch_flash_attention"


class ActivationType(str, Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class AttentionConfig(BaseModel):
    class QueryKeyValueTransformConfig(BaseModel):
        class IdentityTransformConfig(BaseModel):
            pass

        class RotaryTransformConfig(BaseModel):
            n_embd: int = confloat(ge=0.0)
            block_size: int = confloat(ge=0.0)

        type_hint: QueryKeyValueTransformType
        config: IdentityTransformConfig | RotaryTransformConfig

    attention_type: AttentionType
    scaling_factor: conint(ge=1)
    qkv_transform: List[QueryKeyValueTransformConfig]


class WeightInitailizationConfig(BaseModel):
    mean: confloat(ge=0.0)
    std: confloat(ge=0.0)


class GPT2Config(BaseModel):
    sample_key: str
    prediction_key: str
    block_size: conint(ge=1)
    vocab_size: conint(ge=1)  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: conint(ge=1)
    n_head: conint(ge=1)
    n_embd: conint(ge=1)
    ffn_hidden: conint(ge=1)
    dropout: confloat(ge=0.0)
    bias: bool  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    attention: AttentionConfig
    activation: ActivationType
    epsilon: confloat(ge=0.0)
    weight_init: WeightInitailizationConfig

    @model_validator(mode="after")
    def validate_sizes(self) -> "GPT2Config":
        for param, param_name in zip(
            [self.ffn_hidden, self.vocab_size, self.n_embd], ["ffn_hidden", "vocab_size", "n_embd"]
        ):
            if param % 128 != 0:
                # See https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
                raise ValueError(f"{param_name} with value {param} should be divisible by 128 for efficient training.")
        return self
