from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn
from torch import linalg as LA

from modalities.config.config import GradientClippingMode
from modalities.models.model import NNModel
from modalities.utils.gradient_clipping import build_gradient_clipper


@pytest.mark.parametrize(
    "gradient_clipping_mode", [mode for mode in GradientClippingMode if mode != GradientClippingMode.NONE]
)
def test_clipping_gradients_makes_them_smaller(gradient_clipping_mode: GradientClippingMode):
    grad_sum1, grad_sum2, grad_sum_clipper = _run_gradient_clipping_experiment(gradient_clipping_mode, threshold=0.1)
    assert grad_sum1 > grad_sum2
    assert grad_sum1 == grad_sum_clipper


def test_gradient_clipping_mode_none_does_not_change_gradients():
    grad_sum1, grad_sum2, grad_sum_clipper = _run_gradient_clipping_experiment(GradientClippingMode.NONE)
    assert grad_sum1 == grad_sum2
    assert grad_sum1 == grad_sum_clipper


class DummyModel(NNModel):
    def __init__(self):
        super().__init__()
        self._weights = nn.Linear(2, 3)
        self._weights.weight = nn.Parameter(torch.ones_like(self._weights.weight))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self._weights(**inputs)
        return {"output": output}

    def get_aggregated_grad_norm(self, gradient_clipping_mode: GradientClippingMode) -> torch.Tensor:
        gradient_vector = torch.cat([self._weights.weight.grad.flatten(), self._weights.bias.grad])
        if gradient_clipping_mode == GradientClippingMode.MAX_NORM:
            return LA.vector_norm(gradient_vector, ord=torch.inf)
        if gradient_clipping_mode == GradientClippingMode.P1_NORM:
            return LA.vector_norm(gradient_vector, ord=1)
        if gradient_clipping_mode == GradientClippingMode.P2_NORM:
            return LA.vector_norm(gradient_vector, ord=2)
        if gradient_clipping_mode == GradientClippingMode.NONE:
            return gradient_vector.abs().sum()
        if gradient_clipping_mode == GradientClippingMode.VALUE:
            return gradient_vector.abs().sum()


def _run_gradient_clipping_experiment(gradient_clipping_mode: GradientClippingMode, threshold: Optional[float] = None):
    model = DummyModel()
    inputs = {"input": torch.ones(2, 2)}
    output: torch.Tensor = model(inputs)["output"]
    loss = output.sum()
    loss.backward()
    grad_sum1 = model.get_aggregated_grad_norm(gradient_clipping_mode=gradient_clipping_mode)
    clipper = build_gradient_clipper(gradient_clipping_mode, threshold)
    grad_sum_clipper = clipper(model)
    grad_sum2 = model.get_aggregated_grad_norm(gradient_clipping_mode=gradient_clipping_mode)
    return grad_sum1, grad_sum2, grad_sum_clipper
