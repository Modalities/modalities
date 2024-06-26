import math
from abc import abstractmethod
from enum import Enum
from functools import partial
from typing import Annotated, Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from modalities.batch import DatasetBatch, InferenceResultBatch

WeightDecayGroups = Dict[str, List[str]]


class ActivationType(str, Enum):
    GELU = "gelu"
    FUSED_SWIGLU = "fused_swiglu"


class WeightInitializationConfig(BaseModel):
    mean: Annotated[float, Field(strict=True, ge=0.0)]
    std: Annotated[float, Field(strict=True, ge=0.0)] | str  # can be float or "auto"
    type: str


class NNModel(nn.Module):
    def __init__(self, seed: int = None, weight_decay_groups: Optional[WeightDecayGroups] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self._weight_decay_groups = weight_decay_groups if weight_decay_groups is not None else {}
        super(NNModel, self).__init__()

    @property
    def weight_decay_groups(self) -> WeightDecayGroups:
        return self._weight_decay_groups

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}

    def _init_weights(self, module: nn.Module, weight_init: WeightInitializationConfig):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=weight_init.mean, std=weight_init.std)

    def initialize_weights(self, weight_init, number_of_layers: int, hidden_dim: Optional[int] = None):
        # auto: choose std automatically
        if weight_init.std == "auto":
            assert hidden_dim is not None, "ERROR! weight_init.std = 'auto' not implemented"
            weight_init.std = math.sqrt(2 / (5 * hidden_dim))

        # initialize weights
        self.apply(partial(self._init_weights, weight_init=weight_init))

        if weight_init.type == "scaled":
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=weight_init.mean, std=weight_init.std / math.sqrt(2 * number_of_layers)
                    )


def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    forward_result = model.forward(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
