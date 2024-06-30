from abc import ABC, abstractmethod
from typing import List

import torch.nn as nn


class WeightInitializationIF(ABC):
    @abstractmethod
    def initialize_in_place(self, model: nn.Module):
        raise NotImplementedError


class WeightInitializer(WeightInitializationIF):
    def __init__(self, initializers: List[WeightInitializationIF]):
        self.initializers = initializers

    def initialize_weights(self, model: nn.Module):
        for initializer in self.initializers:
            initializer.initialize_in_place(model)


class ModulewiseNormaltInitialization(WeightInitializationIF):
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
