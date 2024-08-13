import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from modalities.config.lookup_enum import LookupEnum
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF


class GradientClippingMode(LookupEnum):
    """
    Enum class representing different modes of gradient clipping.

    Attributes:
        P1_NORM (int): Mode for Manhattan norm based clipping.
        P2_NORM (int): Mode for Euclidean norm based clipping.
        MAX_NORM (str): Mode for maximum norm based clipping.
    """

    P1_NORM = 1
    P2_NORM = 2
    MAX_NORM = "inf"


class FSDPGradientClipper(GradientClipperIF):
    """The FSDPGradientClipper class that is responsible for clipping the gradients of a model wrapped with FSDP."""

    def __init__(self, wrapped_model: FSDP, max_norm: float, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDPGradientClipper object.

        Args:
            wrapped_model (FSDP): The wrapped model.
            max_norm (float): The maximum norm value for gradient clipping.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        """
        Clips the gradients of the wrapped model using the specified maximum norm and norm type.

        Returns:
            torch.Tensor: The gradient norm after clipping.
        """
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=self.max_norm, norm_type=self.norm_type.value)
        return gradient_norm_score


class FSDPLoggingOnlyGradientClipper(GradientClipperIF):
    """The FSDPLoggingOnlyGradientClipper class that is responsible for logging the gradient
    norms without actually clipping the gradients."""

    def __init__(self, wrapped_model: FSDP, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDPLoggingOnlyGradientClipper.

        Args:
            wrapped_model (FSDP): The wrapped FSDP model.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        """
        Returns the gradient norm, but does not apply clipping since max_norm is set to inifinity.

        Returns:
            torch.Tensor: The gradient norms.
        """
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=torch.inf, norm_type=self.norm_type.value)
        return gradient_norm_score


class DummyGradientClipper(GradientClipperIF):
    """The DummyGradientClipper class that does not apply gradient clipping."""

    def __init__(self) -> None:
        pass

    def clip_gradients(self) -> torch.Tensor:
        """
        Returns a tensor with value -1.0 indicating that DummyGradientClipper does not actually apply gradient clipping.

        Returns:
            torch.Tensor: Tensor with value -1.0
        """
        gradient_norm_score = torch.Tensor([-1.0])
        return gradient_norm_score
