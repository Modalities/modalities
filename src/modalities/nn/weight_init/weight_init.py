import warnings
from typing import List

import torch.nn as nn

from modalities.nn.weight_init.weight_init_if import WeightInitializationIF


class WeightInitializerWrapper(WeightInitializationIF):
    def __init__(self, weight_initializers: List[WeightInitializationIF]):
        self.weight_initializers = weight_initializers

    def initialize_in_place(self, model: nn.Module):
        for weight_initializer in self.weight_initializers:
            weight_initializer.initialize_in_place(model)


class ModulewiseNormalInitialization(WeightInitializationIF):
    def __init__(self, mean: float, std: float):
        """Initializes the weights of a model by sampling from a normal distribution.
        NOTE: This class supports the initialization of nn.Linear and nn.Embedding layers.
        For other layer types, the initialization must be subclassed and extended

        Args:
            mean (float): mean of the normal distribution
            std (float): standard deviation of the normal distribution
        """
        self.mean = mean
        self.std = std

    def _init_weights_impl(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=self.mean, std=self.std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=self.mean, std=self.std)
        else:
            warnings.warn(
                f"Module {module.__class__} is not of type nn.Linear or nn.Embedding. "
                "Looking for weight and bias attributes to initialize."
            )
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight, mean=self.mean, std=self.std)

            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

            if not hasattr(module, "weight") and not hasattr(module, "bias"):
                raise NotImplementedError(f"ERROR! Initialization of {module.__class__} not implemented")

    def initialize_in_place(self, model: nn.Module):
        model.apply(self._init_weights_impl)


class NamedParameterwiseNormalInitialization(WeightInitializationIF):
    def __init__(self, mean: float, std: float, parameter_name_suffixes: List[str]):
        self.mean = mean
        self.std = std
        self.parameter_name_suffixes = parameter_name_suffixes

    def initialize_in_place(self, model: nn.Module):
        for pn, p in model.named_parameters():
            for parameter_name_suffix in self.parameter_name_suffixes:
                if pn.endswith(parameter_name_suffix):
                    nn.init.normal_(p, mean=self.mean, std=self.std)
