from abc import abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn

from modalities.batch import DatasetBatch, InferenceResultBatch
from transformers import PreTrainedTokenizer


class NNModel(nn.Module):
    def __init__(self, seed: int = None):
        if seed is not None:
            torch.manual_seed(seed)
        super(NNModel, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}

class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, bias: bool):
        super().__init__()

        hidden_dim = self._get_hidden_dim(n_embd)

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

    def _get_hidden_dim(self, n_embd: int) -> int:
        # Best practice: 4 * n_embd (https://arxiv.org/pdf/1706.03762)
        # Because we add an additional linear layer, we need to adjust the hidden_dim to 2/3 of the original value
        # which is equivalent to the number of parameters in TransformerMLP, i.e.
        # 2 * (n_embd * hidden_dim) == 3 * (n_embd * 2/3 * hidden_dim)
        # Besides, we ensure that hidden_dim is the smallest multiple of 256 that is greater than or equal the provided hidden_dim 
        return 256 * ((int(2 * 4 * n_embd / 3) + 256 - 1) // 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.silu(self.c_fc(x)) * self.c_proj(x))

def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    forward_result = model.forward(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
