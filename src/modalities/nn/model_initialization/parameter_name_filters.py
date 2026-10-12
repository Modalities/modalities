from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WeightInitTypes(Enum):
    PLAIN = "plain"
    SCALED = "scaled"
    SCALED_EMBED = "scaled_embed"


class SupportWeightInitModels(Enum):
    GPT2 = "gpt2"
    COCA = "coca"
    QWEN_MOE = "qwen_moe"


class RegexFilter(BaseModel):
    weights: list[str]
    biases: Optional[list[str]] = Field(default_factory=list)


NAMED_PARAMETER_INIT_GROUPS = {
    SupportWeightInitModels.GPT2: {
        # as per https://arxiv.org/abs/2312.16903
        # plain: All linear and embedding layers and embedding layers
        #        are initialized by sampling from a normal distribution.
        #        NOTE: layer norms are ALWAYS initialized beforehand at instantiation time!
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[
                # attention projection weights
                r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn|c_proj)\.weight",
                # hidden feed forward in attention block
                r"transformer\.h\.\w+\.mlp\.(W|V|W_2)\.weight",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.(c_fc|c_proj)\.weight",  # gelu
                # embedding weights
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                # lm_head weights (although usually tied with transformer.wte.weight)
                r"transformer\.lm_head\.weight",
            ],
            biases=[
                # NOTE: some bias terms might not be present due to user configuration
                r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn|c_proj)\.bias",
                r"transformer\.h\.\w+\.mlp\.(W|V|W_2)\.bias",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.(c_fc|c_proj)\.bias",  # gelu
                r"transformer\.lm_head\.bias",  # typically not present
            ],
        ),
        # scaled: In the attention block, we scale the final projection in the FFN (W_2)
        #         and the projection before the FFN (W_o), as defined in
        #         https://arxiv.org/abs/2312.16903
        WeightInitTypes.SCALED: RegexFilter(
            weights=[
                r"transformer\.h\.\d+\.attn\.c_proj\.weight",
                r"transformer\.h\.\w+\.mlp\.W_2.weight",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.c_proj\.weight",  # gelu
            ]
        ),
        WeightInitTypes.SCALED_EMBED: RegexFilter(
            weights=[
                # embedding weights
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                r"transformer\.lm_head\.weight",
            ]
        ),
    },
    SupportWeightInitModels.COCA: {
        # we reject all bias and weight parameters belonging to norms
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[r"^(?!.*norm)(?!.*ln_).*\.weight$"], biases=[r"^(?!.*norm)(?!.*ln_).*\.bias$"]
        ),
        WeightInitTypes.SCALED: RegexFilter(weights=[], biases=[]),
        WeightInitTypes.SCALED_EMBED: RegexFilter(weights=[], biases=[]),
    },
    # MoE model (modalities.models.moe.qwen_model.QwenModel). The optional
    # "_checkpoint_wrapped_module." infix appears when activation checkpointing is applied before initialization.
    SupportWeightInitModels.QWEN_MOE: {
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[
                r"layers\.\d+\.(_checkpoint_wrapped_module\.)?attn\.(q_proj|k_proj|v_proj|o_proj)\.weight",
                r"layers\.\d+\.(_checkpoint_wrapped_module\.)?ffn\.router\.weight",
                r"layers\.\d+\.(_checkpoint_wrapped_module\.)?ffn\.experts\.(w1|w2|w3)",
                r"token_emb\.weight",
                r"lm_head\.weight",
            ],
        ),
        # scaled: output projections of the attention and of the experts
        WeightInitTypes.SCALED: RegexFilter(
            weights=[
                r"layers\.\d+\.(_checkpoint_wrapped_module\.)?attn\.o_proj\.weight",
                r"layers\.\d+\.(_checkpoint_wrapped_module\.)?ffn\.experts\.w3",
            ]
        ),
        WeightInitTypes.SCALED_EMBED: RegexFilter(
            weights=[
                r"token_emb\.weight",
                r"lm_head\.weight",
            ]
        ),
    },
}
