import torch

from modalities.models.model import NNModel
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF, GradientClippingMode



class TorchGradientClipper(GradientClipperIF):
    """The TorchGradientClipper class that is responsible for clipping the gradients of a pytorch model."""

    def __init__(self, model: NNModel, max_norm: float, norm_type=GradientClippingMode) -> None:
        """
        Initialize the TorchGradientClipper object.

        Args:
            model (NNModel): The pytorch model.
            max_norm (float): The maximum norm value for gradient clipping.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.model = model
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        """
        Clips the gradients of the wrapped model using the specified maximum norm and norm type.

        Returns:
            torch.Tensor: The gradient norm after clipping.
        """
        gradient_norm_score = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type.value
        )
        return gradient_norm_score


class TorchLoggingOnlyGradientClipper(GradientClipperIF):
    """The TorchLoggingOnlyGradientClipper class that is responsible for logging the gradient
    norms without actually clipping the gradients."""

    def __init__(self, model: NNModel, norm_type=GradientClippingMode) -> None:
        """
        Initialize the TorchLoggingOnlyGradientClipper.

        Args:
            model (NNModel): The pytorch model.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.model = model
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        """
        Returns the gradient norm, but does not apply clipping since max_norm is set to inifinity.

        Returns:
            torch.Tensor: The gradient norms.
        """
        gradient_norm_score = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.torch.inf, norm_type=self.norm_type.value
        )
        return gradient_norm_score
