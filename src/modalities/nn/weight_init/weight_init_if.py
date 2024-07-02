from abc import ABC, abstractmethod

import torch.nn as nn


class WeightInitializationIF(ABC):
    @abstractmethod
    def initialize_in_place(self, model: nn.Module):
        raise NotImplementedError
