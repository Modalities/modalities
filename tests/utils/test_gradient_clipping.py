from typing import Dict

import pytest
import torch
import torch.nn as nn

from modalities.config.config import GradientClippingMode
from modalities.models.model import NNModel
from modalities.utils.gradient_clipping import build_gradient_clipper


@pytest.mark.parametrize(
    "gradient_clipping_mode", [mode for mode in GradientClippingMode if mode != GradientClippingMode.NONE]
)
def test_clipping_gradients_makes_them_smaller(gradient_clipping_mode: GradientClippingMode):
    grad_sum1, grad_sum2 = _run_gradient_clipping_experiment(gradient_clipping_mode)
    assert grad_sum1 > grad_sum2


def test_gradient_clipping_mode_none_does_not_change_gradients():
    grad_sum1, grad_sum2 = _run_gradient_clipping_experiment(GradientClippingMode.NONE)
    assert grad_sum1 == grad_sum2


class TestModel(NNModel):
    def __init__(self):
        super().__init__()
        self._weights = nn.Linear(2, 3)
        self._weights.weight = nn.Parameter(torch.ones_like(self._weights.weight))

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self._weights(**inputs)
        return {"output": output}

    def get_grad_sum(self) -> float:
        return self._weights.weight.grad.sum().item()


def _run_gradient_clipping_experiment(gradient_clipping_mode):
    model = TestModel()
    inputs = {"input": torch.rand(2, 2)}
    output: torch.Tensor = model(inputs)["output"]
    loss = output.sum()
    loss.backward()
    grad_sum1 = model.get_grad_sum()
    clipper = build_gradient_clipper(gradient_clipping_mode, 0.001)
    clipper(model)
    grad_sum2 = model.get_grad_sum()
    return grad_sum1, grad_sum2
