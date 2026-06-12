import pytest
import torch
import torch.nn as nn

from modalities.training.gradient_clipping.ep_gradient_clipper import EPGradientClipper
from modalities.training.gradient_clipping.fsdp_gradient_clipper import GradientClippingMode


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.param2 = nn.Parameter(torch.tensor([3.0, 4.0]))


def test_ep_gradient_clipper_clips_gradients():
    model = MockModel()
    model.param1.grad = torch.tensor([1.0, 1.0])
    model.param2.grad = torch.tensor([1.0, 1.0])

    clipper = EPGradientClipper(model_parts=model, max_norm=1.0, norm_type=GradientClippingMode.P2_NORM)
    total_norm = clipper.clip_gradients()

    assert torch.allclose(total_norm, torch.tensor(2.0))
    assert torch.allclose(model.param1.grad, torch.tensor([0.5, 0.5]), atol=1e-6)
    assert torch.allclose(model.param2.grad, torch.tensor([0.5, 0.5]), atol=1e-6)


def test_ep_gradient_clipper_returns_zero_for_no_gradients():
    model = MockModel()

    clipper = EPGradientClipper(model_parts=model, max_norm=1.0, norm_type=GradientClippingMode.P2_NORM)
    total_norm = clipper.clip_gradients()

    assert torch.allclose(total_norm.cpu(), torch.tensor(0.0))


def test_ep_gradient_clipper_raises_for_nonfinite_norm():
    model = MockModel()
    model.param1.grad = torch.tensor([float("nan"), 1.0])

    clipper = EPGradientClipper(
        model_parts=model,
        max_norm=1.0,
        norm_type=GradientClippingMode.P2_NORM,
        error_if_nonfinite=True,
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        clipper.clip_gradients()
