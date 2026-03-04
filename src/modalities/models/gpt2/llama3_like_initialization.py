import math
import re
from functools import partial
from typing import Annotated

import torch.nn as nn
from pydantic import BaseModel, Field

from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.utils.logger_utils import get_logger

logger = get_logger(name="llama3 initialization")


class Llama3InitializerConfig(BaseModel):
    num_layers: Annotated[int, Field(strict=True, gt=0)]
    n_embd: Annotated[int, Field(strict=True, gt=0)]


class Llama3Initializer(ModelInitializationIF):
    """
    Follows weight initialization distributions and parameterization for Llama3 as described in TorchTitan.
    """

    def __init__(self, num_layers: int, n_embd: int) -> None:
        super().__init__()

        self.regex_to_init = {
            # embedding weights
            r"transformer\.wte\.weight": partial(nn.init.normal_, mean=0.0, std=1),
            r"transformer\.wpe\.weight": partial(nn.init.normal_, mean=0.0, std=1),
            # lm head weights
            r"transformer\.lm_head\.weight": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=1 / math.sqrt(n_embd),
                a=-3 / math.sqrt(n_embd),
                b=3 / math.sqrt(n_embd),
            ),
            # qkv projections
            r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn)\.weight": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02,
                a=-2,
                b=2,
            ),
            r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn)\.bias": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02,
                a=-2,
                b=2,
            ),
            # final attention projection in attention block
            r"transformer\.h\.\d+\.attn\.c_proj\.weight": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * num_layers),
                a=-2,
                b=2,
            ),
            r"transformer\.h\.\d+\.attn\.c_proj\.bias": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * num_layers),
                a=-2,
                b=2,
            ),
            # SwiGLU
            r"transformer\.h\.\w+\.mlp\.(W)\.weight": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02,
                a=-2,
                b=2,
            ),
            r"transformer\.h\.\w+\.mlp\.(W)\.bias": nn.init.zeros_,
            r"transformer\.h\.\w+\.mlp\.(V|W_2)\.weight": partial(
                nn.init.trunc_normal_,
                mean=0.0,
                std=0.02 / math.sqrt(2 * num_layers),
                a=-2,
                b=2,
            ),
            r"transformer\.h\.\w+\.mlp\.(V|W_2)\.bias": nn.init.zeros_,
        }

    def initialize_in_place(self, model: nn.Module):
        self._init_by_fqn_regex(model, self.regex_to_init)

    @staticmethod
    def _init_by_fqn_regex(model: nn.Module, regex_to_init: dict[str, partial]):
        for parameter_name, p in model.named_parameters():
            match_count = 0
            for weight_regex in regex_to_init.keys():
                if re.fullmatch(weight_regex, parameter_name):
                    init_fn = regex_to_init[weight_regex]
                    init_fn(p)
                    match_count += 1
            if match_count == 0:
                logger.warning(f"Parameter {parameter_name} did not match any regex for initialization")
            elif match_count > 1:
                raise ValueError(
                    f"Parameter {parameter_name} matched multiple regexes for initialization, which is not allowed"
                )
