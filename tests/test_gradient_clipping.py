import types
from unittest.mock import MagicMock

import torch

from modalities.training.gradient_clipping.fsdp_gradient_clipper import (
    DummyGradientClipper,
    FSDP1GradientClipper,
    FSDP1LoggingOnlyGradientClipper,
    FSDP2GradientClipper,
    FSDP2LoggingOnlyGradientClipper,
    GradientClippingMode,
)


class MockFSDPModel:
    def __init__(self):
        self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.param2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.param1.grad = torch.tensor([1.0, 1.0])
        self.param2.grad = torch.tensor([1.0, 1.0])

    def parameters(self):
        return [self.param1, self.param2]


# Test for FSDP1 gradient clipper
def test_fsdp1_gradient_clipper():
    """
    Test FSDP1GradientClipper's ability to clip gradients correctly.
    Uses a mock model with a dynamically added clip_grad_norm_ method to verify norm calculation and gradient scaling.
    """
    mock_model = MockFSDPModel()
    max_norm = 1.0
    norm_type = GradientClippingMode.P2_NORM

    # Note: FSDPGradientClipper requires clip_grad_norm_, but user's model lacks it.
    # To use FSDPGradientClipper, we’d need to add this method, which deviates from the request.
    # For strict adherence, we could skip this test or raise an error, but let’s adapt.
    # Temporarily extend MockFSDPModel in this test (with a comment explaining).
    def clip_grad_norm_(self, max_norm, norm_type):
        params = [p for p in self.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in params]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                p.grad.data.mul_(clip_coef)
        return total_norm

    # Dynamically add the method for this test
    mock_model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, mock_model)

    clipper = FSDP1GradientClipper(wrapped_model=mock_model, max_norm=max_norm, norm_type=norm_type)
    norm = clipper.clip_gradients()

    # Expected norm before clipping: sqrt(1^2 + 1^2 + 1^2 + 1^2) = 2.0
    expected_norm = torch.tensor(2.0)
    assert torch.allclose(norm, expected_norm), f"Expected norm {expected_norm}, got {norm}"

    # Gradients should be scaled to max_norm / total_norm = 1.0 / 2.0 = 0.5
    expected_grad = torch.tensor([0.5, 0.5])
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, expected_grad), f"Expected grad {expected_grad}, got {param.grad}"


def test_fsdp1_logging_only_gradient_clipper():
    """
    Test that FSDP1LoggingOnlyGradientClipper calls clip_grad_norm_ with max_norm=torch.inf,
    ensuring no clipping occurs, and returns the gradient norm.
    """
    # Create a mock FSDP1 model
    mock_model = MagicMock()
    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP1LoggingOnlyGradientClipper(wrapped_model=mock_model, norm_type=norm_type)

    # Call clip_gradients
    clipper.clip_gradients()

    # Verify that clip_grad_norm_ was called with max_norm=torch.inf
    mock_model.clip_grad_norm_.assert_called_once_with(max_norm=torch.inf, norm_type=norm_type.value)


def test_fsdp2_clip_grad_norm():
    """
    Test the static clip_grad_norm_ method in FSDP2GradientClipper to ensure it correctly
    computes the gradient norm and clips gradients when necessary.
    """
    # Create parameters with gradients
    mock_model = MockFSDPModel()

    # Compute expected total norm (Euclidean norm, norm_type=2)
    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # sqrt(4) = 2.0

    # Test case 1: max_norm > total_norm (no clipping)
    max_norm = expected_norm + 1  # 3.0
    norm = FSDP2GradientClipper.clip_grad_norm_(parameters=mock_model.parameters(), max_norm=max_norm, norm_type=2.0)
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"
    assert torch.allclose(mock_model.param1.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"
    assert torch.allclose(mock_model.param2.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"

    # Test case 2: max_norm < total_norm (clipping occurs)
    max_norm = expected_norm / 2  # 1.0
    norm = FSDP2GradientClipper.clip_grad_norm_(parameters=mock_model.parameters(), max_norm=max_norm, norm_type=2.0)
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match pre-clipping total norm"
    scale = max_norm / expected_norm  # 1.0 / 2.0 = 0.5
    expected_grad = torch.tensor([1.0 * scale, 1.0 * scale])
    assert torch.allclose(mock_model.param1.grad, expected_grad), "Gradients should be clipped"
    assert torch.allclose(mock_model.param2.grad, expected_grad), "Gradients should be clipped"


def test_fsdp2_gradient_clipper():
    """
    Test that FSDP2GradientClipper correctly calls clip_grad_norm_ on the wrapped model's parameters.
    """
    # Create a mock FSDP2 model with parameters

    mock_model = MockFSDPModel()

    max_norm = 1.0
    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP2GradientClipper(wrapped_model=mock_model, max_norm=max_norm, norm_type=norm_type)

    # Call clip_gradients
    norm = clipper.clip_gradients()

    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # 2.0
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"

    scale = max_norm / expected_norm  # 0.5
    expected_grad = torch.tensor([1.0 * scale, 1.0 * scale])
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, expected_grad), "Gradients should be clipped"


def test_fsdp2_logging_only_gradient_clipper():
    """
    Test that FSDP2LoggingOnlyGradientClipper computes the gradient norm without clipping.
    """
    mock_model = MockFSDPModel()

    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP2LoggingOnlyGradientClipper(wrapped_model=mock_model, norm_type=norm_type)

    # Call clip_gradients
    norm = clipper.clip_gradients()

    # Verify the norm and that gradients are unchanged
    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # 2.0
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"
    for param in mock_model.parameters():
        assert torch.allclose(param.grad, torch.tensor([1.0, 1.0])), "Gradients should not be modified"


def test_dummy_gradient_clipper():
    """
    Test that DummyGradientClipper returns a tensor with -1.0 and does not affect gradients.
    """
    clipper = DummyGradientClipper()
    norm = clipper.clip_gradients()
    assert torch.allclose(norm, torch.tensor([-1.0])), "Norm should be -1.0 indicating no clipping"
