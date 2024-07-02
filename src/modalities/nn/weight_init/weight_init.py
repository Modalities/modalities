import re
from typing import List

import torch.nn as nn

from modalities.nn.weight_init.parameter_name_regex_filters import RegexFilter
from modalities.nn.weight_init.weight_init_if import WeightInitializationIF


class WeightInitializerWrapper(WeightInitializationIF):
    def __init__(self, weight_initializers: List[WeightInitializationIF]):
        self.weight_initializers = weight_initializers

    def initialize_in_place(self, model: nn.Module):
        for weight_initializer in self.weight_initializers:
            weight_initializer.initialize_in_place(model)


class NamedParameterwiseNormalInitialization(WeightInitializationIF):
    def __init__(self, mean: float, std: float, parameter_name_regexes: RegexFilter):
        self.mean = mean
        self.std = std
        self.parameter_name_regexes = parameter_name_regexes

    def initialize_in_place(self, model: nn.Module):
        for parameter_name, p in model.named_parameters():
            weight_regexes = self.parameter_name_regexes.weights
            for weight_regex in weight_regexes:
                if re.fullmatch(weight_regex, parameter_name):
                    nn.init.normal_(p, mean=self.mean, std=self.std)
            bias_regexes = self.parameter_name_regexes.biases
            for bias_regex in bias_regexes:
                if re.fullmatch(bias_regex, parameter_name):
                    nn.init.zeros_(p)
