import math
import re
import warnings
from enum import Enum
from typing import Annotated

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, model_validator

from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.nn.model_initialization.parameter_name_filters import RegexFilter


class MultiDeviceGeneratorPolicy(str, Enum):
    IGNORE = "ignore"
    WARN = "warn"
    ERROR = "error"


# Wrappers that insert themselves into a parameter's fully qualified name without changing which
# logical parameter it is. The initialization filters are written against the plain model FQNs, so
# these prefixes are stripped before matching. Without this, applying activation checkpointing (or
# torch.compile) before initialization would silently prevent every per-layer regex from matching,
# leaving the model with its default rather than its configured initialization.
_FQN_WRAPPER_PREFIXES = (
    "_orig_mod.",  # torch.compile
    "_checkpoint_wrapped_module.",  # activation checkpointing
    "_fsdp_wrapped_module.",  # FSDP1
)


def normalize_parameter_name(parameter_name: str) -> str:
    """
    Removes wrapper prefixes from a parameter's fully qualified name.

    Args:
        parameter_name (str): The fully qualified parameter name, possibly containing wrapper
            segments such as ``_checkpoint_wrapped_module.``.

    Returns:
        str: The name as it would appear on the unwrapped model.
    """
    for prefix in _FQN_WRAPPER_PREFIXES:
        parameter_name = parameter_name.replace(prefix, "")
    return parameter_name


class PlainInitializationConfig(BaseModel):
    mean: float
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    parameter_name_regexes: list[str]  # here we filter for the parameter names, e.g., "c_proj.weight"
    hidden_dim: int | None = None

    @model_validator(mode="after")
    def check_std_and_hidden_dim(self):
        # in case of initialization with "auto", we need to specify the hidden_dim
        if self.std == "auto" and self.hidden_dim is None:
            raise ValueError("hidden_dim must be specified when std is 'auto'")
        # in case of initialization with float std, we must not specify the hidden_dim
        if isinstance(self.std, float) and self.hidden_dim is not None:
            raise ValueError("hidden_dim must not be specified when std is a float value")
        return self


class ScaledInitializationConfig(BaseModel):
    mean: float
    std: Annotated[float, Field(strict=True, ge=0.0)]
    num_layers: Annotated[int, Field(strict=True, gt=0)]
    parameter_name_regexes: list[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class ScaledEmbedInitializationConfig(BaseModel):
    mean: float
    parameter_name_regexes: list[str]  # here we filter for the parameter names, e.g., "c_proj.weight"


class NamedParameterwiseNormalInitialization(ModelInitializationIF):
    def __init__(
        self,
        mean: float,
        std: float,
        parameter_name_regexes: RegexFilter,
        seed: int | None = None,
        multi_device_generator_policy: MultiDeviceGeneratorPolicy = MultiDeviceGeneratorPolicy.WARN,
    ):
        self.mean = mean
        self.std = std
        self.parameter_name_regexes = parameter_name_regexes
        self.seed = torch.initial_seed() if seed is None else seed
        self.multi_device_generator_policy = multi_device_generator_policy
        self._generators: dict[str, torch.Generator] = {}

    def _get_generator(self, parameter: torch.Tensor) -> torch.Generator:
        device_key = str(parameter.device)
        generator = self._generators.get(device_key)
        if generator is None:
            if len(self._generators) > 0:
                message = (
                    "NamedParameterwiseNormalInitialization created generators for multiple devices in one process "
                    f"(existing={list(self._generators.keys())}, new={device_key})."
                )
                if self.multi_device_generator_policy == MultiDeviceGeneratorPolicy.ERROR:
                    raise RuntimeError(message)
                if self.multi_device_generator_policy == MultiDeviceGeneratorPolicy.WARN:
                    warnings.warn(message, stacklevel=2)
            generator = torch.Generator(device=parameter.device)
            generator.manual_seed(self.seed)
            self._generators[device_key] = generator
        return generator

    def initialize_in_place(self, model: nn.Module):
        weight_regexes = self.parameter_name_regexes.weights
        bias_regexes = self.parameter_name_regexes.biases or []
        for parameter_name, p in model.named_parameters():
            # Strip FQN modifications introduced by torch.compile, activation checkpointing and
            # FSDP1 so that the filters can be written against the plain model.
            parameter_name = normalize_parameter_name(parameter_name)
            for weight_regex in weight_regexes:
                if re.fullmatch(weight_regex, parameter_name):
                    nn.init.normal_(p, mean=self.mean, std=self.std, generator=self._get_generator(p))
            for bias_regex in bias_regexes:
                if re.fullmatch(bias_regex, parameter_name):
                    nn.init.zeros_(p)


class InitializationRoutines:
    @staticmethod
    def get_plain_initialization(
        mean: float,
        std: float | str,
        parameter_name_regexes: RegexFilter,
        hidden_dim: int | None = None,
        seed: int | None = None,
        multi_device_generator_policy: MultiDeviceGeneratorPolicy = MultiDeviceGeneratorPolicy.WARN,
    ) -> NamedParameterwiseNormalInitialization:
        """Initializes the weights of a model by sampling from a normal distribution.
        NOTE: This class supports the initialization of nn.Linear and nn.Embedding layers.
        For other layer types, the initialization must be subclassed and extended

        Args:
            mean (float): mean of the normal distribution
            std (float): standard deviation of the normal distribution. If set to "auto", appropiate
                value selected as per plain initialization described in https://arxiv.org/abs/2312.16903
            hidden_dim (Optional[int]): hidden dimension of the attention layer. Defaults to None.
            parameter_name_regexes (list[str]): List of parameter name regexes to which the initialization
                should be applied
            seed (Optional[int]): Random seed for initialization. Defaults to None.
            multi_device_generator_policy (MultiDeviceGeneratorPolicy): Behavior when more than one
                device-local RNG generator is created in the same process.
        """
        # auto: choose std automatically
        if std == "auto":
            if hidden_dim is None:
                raise ValueError("ERROR! weight_init.std = auto not implemented")
            # as per  https://arxiv.org/abs/2312.16903
            std = math.sqrt(2 / (5 * hidden_dim))
        assert isinstance(std, float)

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean,
            std=std,
            parameter_name_regexes=parameter_name_regexes,
            seed=seed,
            multi_device_generator_policy=multi_device_generator_policy,
        )
        return initialization

    @staticmethod
    def get_scaled_initialization(
        mean: float,
        std: float,
        num_layers: int,
        parameter_name_regexes: RegexFilter,
        seed: int | None = None,
        multi_device_generator_policy: MultiDeviceGeneratorPolicy = MultiDeviceGeneratorPolicy.WARN,
    ) -> ModelInitializationIF:
        """Implementation of scaled weight initialization. As defined in https://arxiv.org/abs/2312.16903

        Args:
            mean (float): Mean of the normal distribution
            std (float): Standard deviation of the normal distribution used to initialize the other weights
            num_layers (int): Number of layers in the model which we use to downscale std with
            parameter_name_regexes (RegexFilter): List of parameter name regexes to which the initialization
                should be applied
            seed (Optional[int]): Random seed for initialization. Defaults to None.
            multi_device_generator_policy (MultiDeviceGeneratorPolicy): Behavior when more than one
                device-local RNG generator is created in the same process.

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        # see https://arxiv.org/abs/2312.16903
        scaled_std = std / math.sqrt(2 * num_layers)

        initialization = NamedParameterwiseNormalInitialization(
            mean=mean,
            std=scaled_std,
            parameter_name_regexes=parameter_name_regexes,
            seed=seed,
            multi_device_generator_policy=multi_device_generator_policy,
        )
        return initialization

    @staticmethod
    def get_scaled_embed_initialization(
        mean: float,
        parameter_name_regexes: RegexFilter,
        seed: int | None = None,
        multi_device_generator_policy: MultiDeviceGeneratorPolicy = MultiDeviceGeneratorPolicy.WARN,
    ) -> ModelInitializationIF:
        """Implementation of scaled weight initialization for embeddings, see https://arxiv.org/abs/2312.16903
        We fix the standard deviation to sqrt(0.4).

        Args:
            mean (float): Mean of the normal distribution
            parameter_name_regexes (list[str], optional): List of parameter name regexes to which the initialization
                should be applied Defaults to None.
            seed (Optional[int]): Random seed for initialization. Defaults to None.
            multi_device_generator_policy (MultiDeviceGeneratorPolicy): Behavior when more than one
                device-local RNG generator is created in the same process.

        Returns:
            WeightInitializationIF: Weight initialization object
        """
        std = math.sqrt(0.4)
        initialization = NamedParameterwiseNormalInitialization(
            mean=mean,
            std=std,
            parameter_name_regexes=parameter_name_regexes,
            seed=seed,
            multi_device_generator_policy=multi_device_generator_policy,
        )
        return initialization
