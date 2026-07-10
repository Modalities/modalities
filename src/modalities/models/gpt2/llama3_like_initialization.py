import math
import re
from typing import Annotated, Callable

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.utils.logger_utils import get_logger

logger = get_logger(name="llama3 initialization")


class Llama3InitializerConfig(BaseModel):
    num_layers: Annotated[int, Field(strict=True, gt=0)]
    n_embd: Annotated[int, Field(strict=True, gt=0)]
    depth_init: bool = True


class Llama3Initializer(ModelInitializationIF):
    """
    Follows weight initialization distributions and parameterization for Llama3 as described in TorchTitan.
    """

    def __init__(self, num_layers: int, n_embd: int, depth_init: bool) -> None:
        """
        Initializes the Llama3Initializer.
        Args:
            num_layers: The number of transformer layers in the model. Used to calculate std for certain parameters.
            n_embd: The embedding dimension of the model. Used to calculate std and truncation for certain parameters.
            depth_init: Whether to use depth-aware initialization for certain parameters, where the std
                        is scaled based on the layer's depth in the model. If False, a constant std is
                        used for all layers baed on num_layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.n_embd = n_embd
        self.depth_init = depth_init

    def _build_regex_to_init(self, use_weight_tying: bool) -> dict[str, tuple[Callable, dict]]:
        regex_to_init: dict[str, tuple[Callable, dict]] = {
            # qkv projections
            r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn)\.weight": (
                trunc_normal_,
                {
                    "mean": 0.0,
                    "std": 0.02,
                    "a": -2,
                    "b": 2,
                },
            ),
            # final attention projection in attention block
            r"transformer\.h\.\d+\.attn\.c_proj\.weight": (
                trunc_normal_,
                {
                    "mean": 0.0,
                    "std": (
                        (lambda layer_id: 0.02 / math.sqrt(2 * (layer_id + 1)))
                        if self.depth_init
                        else 0.02 / math.sqrt(2 * self.num_layers)
                    ),
                    "a": -2,
                    "b": 2,
                },
            ),
            # SwiGLU
            r"transformer\.h\.\d+\.mlp\.(W)\.weight": (
                trunc_normal_,
                {
                    "mean": 0.0,
                    "std": 0.02,
                    "a": -2,
                    "b": 2,
                },
            ),
            r"transformer\.h\.\d+\.mlp\.(V|W_2)\.weight": (
                trunc_normal_,
                {
                    "mean": 0.0,
                    "std": (
                        (lambda layer_id: 0.02 / math.sqrt(2 * (layer_id + 1)))
                        if self.depth_init
                        else 0.02 / math.sqrt(2 * self.num_layers)
                    ),
                    "a": -2,
                    "b": 2,
                },
            ),
        }

        # Initialization of the output projection (the matrix that produces the logits): small std
        # 1/sqrt(n_embd) so the logits are well-scaled at init.
        output_projection_init = (
            trunc_normal_,
            {
                "mean": 0.0,
                "std": 1 / math.sqrt(self.n_embd),
                "a": -3 / math.sqrt(self.n_embd),
                "b": 3 / math.sqrt(self.n_embd),
            },
        )
        if use_weight_tying:
            # With weight tying, transformer.wte.weight IS the output projection (lm_head shares the
            # same tensor), so it must use the small output std instead of the embedding std of 1.
            # Otherwise the tied matrix produces logits ~sqrt(n_embd)x too large at init, causing the
            # initial loss/grad norm to explode.
            regex_to_init[r"transformer\.wte\.weight"] = output_projection_init
        else:
            # Untied: wte is the embedding (std=1) and lm_head is the separate output projection.
            regex_to_init[r"transformer\.wte\.weight"] = (nn.init.normal_, {"mean": 0.0, "std": 1})
            regex_to_init[r"transformer\.lm_head\.weight"] = output_projection_init
        return regex_to_init

    def initialize_in_place(self, model: nn.Module):
        # The FQN regexes are specific to GPT2LLM, which is also the single source of truth for whether
        # the word embeddings are tied -- so we infer tying from the model rather than tracking a
        # separate flag that could disagree with it (wrong-std tied output projection / uninitialized
        # lm_head). Reject model types we cannot initialize.
        if not isinstance(model, GPT2LLM):
            raise TypeError(
                f"Llama3Initializer only supports GPT2LLM (its FQN regexes are specific to it), "
                f"but received {type(model).__name__}."
            )
        regex_to_init = self._build_regex_to_init(use_weight_tying=model.has_tied_word_embeddings)
        self._init_by_fqn_regex(model, regex_to_init)

    @staticmethod
    def _init_by_fqn_regex(model: nn.Module, regex_to_init: dict[str, tuple[Callable, dict]]):
        hits = {k: 0 for k in regex_to_init.keys()}

        for parameter_name, p in model.named_parameters():
            if parameter_name.endswith("bias"):
                raise ValueError(
                    f"Bias initialization is not allowed for Llama3Initializer. Found bias parameter: {parameter_name}"
                )
            match_count = 0
            for weight_regex in regex_to_init.keys():
                parameter_name = parameter_name.replace(
                    "_orig_mod.", ""
                )  # remove FQN modification from torch.compile if present
                if re.fullmatch(weight_regex, parameter_name):
                    init_fn, arg_dict = regex_to_init[weight_regex]
                    if arg_dict["std"] is not None and callable(arg_dict["std"]):
                        # If std is a function, call it with the layer_id
                        layer_id_match = re.search(r"transformer\.h\.(\d+)\.", parameter_name)
                        if layer_id_match is not None:
                            layer_id = int(layer_id_match.group(1))
                            arg_dict = arg_dict.copy()  # create a copy of the arg_dict to avoid mutating the original
                            arg_dict["std"] = arg_dict["std"](layer_id)
                        else:
                            raise ValueError(
                                f"Could not extract layer_id from parameter name {parameter_name} "
                                "for dynamic std calculation"
                            )
                    init_fn(p, **arg_dict)
                    match_count += 1
                    hits[weight_regex] += 1

            if match_count == 0:
                logger.warning(f"Parameter {parameter_name} did not match any regex for initialization")
            elif match_count > 1:
                raise ValueError(
                    f"Parameter {parameter_name} matched multiple regexes for initialization, which is not allowed"
                )

        for k, count in hits.items():
            if count == 0:
                raise ValueError(
                    f"Regex {k} did not match any FQNs. The model specification probably does not match LLama3."
                )


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
):
    """
    Fills the input tensor with values sampled from a truncated normal distribution.
    Values are drawn from a normal distribution with the given mean and standard
    deviation. Any sampled values outside the range defined by a and b are resampled
    until they fall within the bounds.

    To avoid numerical instability in torch.nn.init.trunc_normal_, the initialization
    is always performed using float32 precision. The result is then cast back to the
    original data type of the input tensor.

    Args:
        tensor: an n dimensional torch Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the lower bound for truncation
        b: the upper bound for truncation

    Returns:
        The input tensor filled with values from the truncated normal distribution.
    """
    # This function is copied from from Meta's open-source project TorchTitan,
    # licensed under the BSD 3-Clause License.
    tmp = tensor.float()
    nn.init.trunc_normal_(tmp, mean=mean, std=std, a=a, b=b)
    tensor.copy_(tmp)
