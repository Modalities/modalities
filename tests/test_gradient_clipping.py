from modalities.training.gradient_clipping.fsdp_gradient_clipper import DummyGradientClipper, FSDP1GradientClipper, FSDP1LoggingOnlyGradientClipper, FSDP2GradientClipper, FSDP2LoggingOnlyGradientClipper, GradientClippingMode
import torch
from unittest.mock import MagicMock

class MockFSDP2Model:
    def __init__(self):
        self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.param2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.param1.grad = torch.tensor([1.0, 1.0])
        self.param2.grad = torch.tensor([1.0, 1.0])

    def parameters(self):
        return [self.param1, self.param2]
# Note: Replace 'your_module' above with the correct module path where the gradient clipping classes are defined.

def test_fsdp1_gradient_clipper():
    """
    Test that FSDP1GradientClipper correctly calls the wrapped model's clip_grad_norm_ method
    with the specified max_norm and norm_type.
    """
    # Create a mock FSDP1 model
    mock_model = MagicMock()
    max_norm = 1.0
    norm_type = GradientClippingMode.P2_NORM
    clipper = FSDP1GradientClipper(wrapped_model=mock_model, max_norm=max_norm, norm_type=norm_type)

    # Call clip_gradients
    norm = clipper.clip_gradients()

    # Verify that clip_grad_norm_ was called with the correct arguments
    mock_model.clip_grad_norm_.assert_called_once_with(max_norm=max_norm, norm_type=norm_type.value)
    # Note: The actual norm returned depends on the mock's return value, which isn't tested here


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
    norm = clipper.clip_gradients()

    # Verify that clip_grad_norm_ was called with max_norm=torch.inf
    mock_model.clip_grad_norm_.assert_called_once_with(max_norm=torch.inf, norm_type=norm_type.value)


def test_fsdp2_clip_grad_norm():
    """
    Test the static clip_grad_norm_ method in FSDP2GradientClipper to ensure it correctly
    computes the gradient norm and clips gradients when necessary.
    """
    # Create parameters with gradients
    param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    param2 = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    param1.grad = torch.tensor([1.0, 1.0])
    param2.grad = torch.tensor([1.0, 1.0])
    parameters = [param1, param2]

    # Compute expected total norm (Euclidean norm, norm_type=2)
    expected_norm = (1**2 + 1**2 + 1**2 + 1**2) ** 0.5  # sqrt(4) = 2.0

    # Test case 1: max_norm > total_norm (no clipping)
    max_norm = expected_norm + 1  # 3.0
    norm = FSDP2GradientClipper.clip_grad_norm_(
        parameters=parameters, max_norm=max_norm, norm_type=2.0
    )
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match expected total norm"
    assert torch.allclose(param1.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"
    assert torch.allclose(param2.grad, torch.tensor([1.0, 1.0])), "Gradients should not be clipped"

    # Test case 2: max_norm < total_norm (clipping occurs)
    max_norm = expected_norm / 2  # 1.0
    norm = FSDP2GradientClipper.clip_grad_norm_(
        parameters=parameters, max_norm=max_norm, norm_type=2.0
    )
    assert torch.allclose(norm, torch.tensor(expected_norm)), "Norm should match pre-clipping total norm"
    scale = max_norm / expected_norm  # 1.0 / 2.0 = 0.5
    expected_grad = torch.tensor([1.0 * scale, 1.0 * scale])
    assert torch.allclose(param1.grad, expected_grad), "Gradients should be clipped"
    assert torch.allclose(param2.grad, expected_grad), "Gradients should be clipped"


def test_fsdp2_gradient_clipper():
    """
    Test that FSDP2GradientClipper correctly calls clip_grad_norm_ on the wrapped model's parameters.
    """
    # Create a mock FSDP2 model with parameters

        
    mock_model = MockFSDP2Model()
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
    mock_model = MockFSDP2Model()
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