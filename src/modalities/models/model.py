from abc import abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from modalities.batch import DatasetBatch, InferenceResultBatch

WeightDecayGroups = dict[str, list[str]]


class ActivationType(str, Enum):
    """
    Enum class representing different activation types.

    Attributes:
        GELU (str): GELU activation type.
        SWIGLU (str): SWIGLU activation type.
    """

    GELU = "gelu"
    SWIGLU = "swiglu"


class NNModel(nn.Module):
    """NNModel class to define a base model."""

    def __init__(self, seed: int = None, weight_decay_groups: Optional[WeightDecayGroups] = None):
        """
        Initializes an NNModel object.

        Args:
            seed (int, optional): The seed value for random number generation. Defaults to None.
            weight_decay_groups (Optional[WeightDecayGroups], optional): The weight decay groups. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self._weight_decay_groups = weight_decay_groups if weight_decay_groups is not None else {}
        super(NNModel, self).__init__()

    @property
    def weight_decay_groups(self) -> WeightDecayGroups:
        """
        Returns the weight decay groups.

        Returns:
            WeightDecayGroups: The weight decay groups.
        """
        return self._weight_decay_groups

    @abstractmethod
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing output tensors.
        """
        raise NotImplementedError

    def get_parameters(self) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of the model's parameters.

        Returns:
            A dictionary where the keys are the parameter names and the values are the corresponding parameter tensors.
        """
        return {name: param for name, param in self.named_parameters()}


class SwiGLU(nn.Module):
    """SwiGLU class to define the SwiGLU activation function."""

    def __init__(
        self, n_embd: int, ffn_hidden: int, bias: bool, enforce_swiglu_hidden_dim_multiple_of: Optional[int] = None
    ):
        """
        Initializes the SwiGLU object.

        Args:
            n_embd (int): The number of embedding dimensions.
            ffn_hidden (int): The number of hidden dimensions in the feed-forward network.
            Best practice: 4 * n_embd (https://arxiv.org/pdf/1706.03762)
            bias (bool): Whether to include bias terms in the linear layers.
            enforce_swiglu_hidden_dim_multiple_of (int): The multiple of which the hidden dimension should be enforced.
                This is required for FSDP + TP as the combincation does not support uneven sharding (yet).
                Defaults to 256 if not provided.
        """

        super().__init__()
        if enforce_swiglu_hidden_dim_multiple_of is None:
            enforce_swiglu_hidden_dim_multiple_of = 256
        hidden_dim = SwiGLU._get_hidden_dim(
            ffn_hidden=ffn_hidden, enforce_swiglu_hidden_dim_multiple_of=enforce_swiglu_hidden_dim_multiple_of
        )

        self.W = nn.Linear(
            in_features=n_embd,
            out_features=hidden_dim,
            bias=bias,
        )
        self.silu = nn.SiLU()
        self.V = nn.Linear(
            in_features=n_embd,
            out_features=hidden_dim,
            bias=bias,
        )
        self.W_2 = nn.Linear(
            in_features=hidden_dim,
            out_features=n_embd,
            bias=bias,
        )

    @staticmethod
    def _get_hidden_dim(ffn_hidden: int, enforce_swiglu_hidden_dim_multiple_of: int) -> int:
        # Calculate the hidden dimension for the SwiGLU module based on the provided embedding dimension.

        # Best practice: 4 * n_embd (https://arxiv.org/pdf/1706.03762)
        # To ensure that the number of parameters in the SwiGLU module with its additional
        # linear layer are equivalent to the TransformerMLP, we need to adapt the SwiGLU hidden dimension as follows:
        # 2 * (n_embd * hidden_dim) == 3 * (n_embd * 2/3 * hidden_dim)
        # Besides, we ensure that hidden_dim is the smallest multiple of
        # `enforce_swiglu_hidden_dim_multiple_of` that is greater than or equal the provided hidden_dim.
        # In case of TP we must set this to be at least of world size as FSDP + TP does not uneven sharding.
        # FSDP itself without TP support it already however.
        return enforce_swiglu_hidden_dim_multiple_of * (
            (int(2 * ffn_hidden / 3) + enforce_swiglu_hidden_dim_multiple_of - 1)
            // enforce_swiglu_hidden_dim_multiple_of
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.W_2(self.silu(self.W(x)) * self.V(x))


def model_predict_batch(model: nn.Module, batch: DatasetBatch) -> InferenceResultBatch:
    """
    Predicts the output for a batch of samples using the given model.

    Args:
        model (nn.Module): The model used for prediction.
        batch (DatasetBatch): The batch of samples to be predicted.

    Returns:
        InferenceResultBatch: The batch of inference results containing the predicted targets and predictions.
    """
    forward_result = model(batch.samples)
    result_batch = InferenceResultBatch(targets=batch.targets, predictions=forward_result)
    return result_batch
