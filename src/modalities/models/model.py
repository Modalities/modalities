from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from modalities.batch import DatasetBatch, InferenceResultBatch

WeightDecayGroups = Dict[str, List[str]]


class ActivationType(str, Enum):
    GELU = "gelu"
    SWIGLU = "swiglu"


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


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, bias: bool):
        super().__init__()

        hidden_dim = SwiGLU._get_hidden_dim(n_embd)

        self.c_fc = nn.Linear(
            in_features=n_embd,
            out_features=hidden_dim,
            bias=bias,
        )
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(
            in_features=n_embd,
            out_features=hidden_dim,
            bias=bias,
        )
        self.out_proj = nn.Linear(
            in_features=hidden_dim,
            out_features=n_embd,
            bias=bias,
        )

    @staticmethod
    def _get_hidden_dim(n_embd: int) -> int:
        # Best practice: 4 * n_embd (https://arxiv.org/pdf/1706.03762)
        # To ensure that the number of parameters in the SwiGLU module with its additional
        # linear layer are equivalent to the TransformerMLP, we need to adapt the SwiGLU hidden dimension as follows:
        # 2 * (n_embd * hidden_dim) == 3 * (n_embd * 2/3 * hidden_dim)
        # Besides, we ensure that hidden_dim is the smallest multiple of
        # 256 that is greater than or equal the provided hidden_dim
        return 256 * ((int(2 * 4 * n_embd / 3) + 256 - 1) // 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.silu(self.c_fc(x)) * self.c_proj(x))


def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    forward_result = model.forward(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
