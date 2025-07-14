from typing import Iterable, Optional

import torch
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.tensor import DTensor

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


class FSDP1GradientClipper(GradientClipperIF):
    """The FSDP1GradientClipper class that is responsible for clipping the gradients of a model wrapped with FSDP.
    Follows the documentation from
    https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
    """

    def __init__(self, wrapped_model: FSDP1, max_norm: float, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDP1GradientClipper object.

        Args:
            wrapped_model (FSDP1): The wrapped model.
            max_norm (float): The maximum norm value for gradient clipping.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.max_norm = max_norm
        self.norm_type = norm_type

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Clips the gradients of the wrapped model using the specified maximum norm and norm type.

        Returns:
            torch.Tensor: The gradient norm after clipping.
        """
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=self.max_norm, norm_type=self.norm_type.value)
        return gradient_norm_score


class FSDP1LoggingOnlyGradientClipper(GradientClipperIF):
    """The FSDP1LoggingOnlyGradientClipper class that is responsible for logging the gradient
    norms without actually clipping the gradients."""

    def __init__(self, wrapped_model: FSDP1, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDP1LoggingOnlyGradientClipper.

        Args:
            wrapped_model (FSDP1): The wrapped FSDP1 model.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.norm_type = norm_type

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Returns the gradient norm, but does not apply clipping since max_norm is set to inifinity.

        Returns:
            torch.Tensor: The gradient norms.
        """
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=torch.inf, norm_type=self.norm_type.value)
        return gradient_norm_score


class FSDP2GradientClipper(GradientClipperIF):
    """The FSDP2GradientClipper class that is responsible for clipping the gradients of a model wrapped with FSDP."""

    def __init__(self, wrapped_model: FSDP2, max_norm: float, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDP2GradientClipper object.

        Args:
            wrapped_model (FSDP2): The wrapped model.
            max_norm (float): The maximum norm value for gradient clipping.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.max_norm = max_norm
        self.norm_type = norm_type

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Clips the gradients of the wrapped model using the specified maximum norm and norm type.

        Returns:
            torch.Tensor: The gradient norm after clipping.
        """
        gradient_norm_score = FSDP2GradientClipper.clip_grad_norm_(
            parameters=self.wrapped_model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type.value,
            error_if_nonfinite=True,
            foreach=True,
        )
        return gradient_norm_score

    @staticmethod
    def clip_grad_norm_(
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Clip the gradient norm of an iterable of parameters.

        Gradient norm clipping requires computing the gradient norm over the entire model.
        `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.

        TODO: for pipeline parallelism, we need to implement it like here:
        https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/distributed/utils.py#L245
        I removed all the code w.r.t. pipeline parallelism for now.

        Args:
            parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).

        """
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            tensors=grads, norm_type=norm_type, error_if_nonfinite=error_if_nonfinite, foreach=foreach
        )

        # Inspired by torch titan
        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has the purpose to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(
            parameters=parameters, max_norm=max_norm, total_norm=total_norm, foreach=foreach
        )
        return total_norm


class FSDP2LoggingOnlyGradientClipper(GradientClipperIF):
    """The FSDP2LoggingOnlyGradientClipper class that is responsible for logging the gradient
    norms without actually clipping the gradients."""

    def __init__(self, wrapped_model: FSDP2, norm_type=GradientClippingMode) -> None:
        """
        Initialize the FSDP2LoggingOnlyGradientClipper.

        Args:
            wrapped_model (FSDP2): The wrapped FSDP2 model.
            norm_type (GradientClippingMode, optional): The type of gradient clipping. Defaults to GradientClippingMode.

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.norm_type = norm_type

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Returns the gradient norm, but does not apply clipping since max_norm is set to inifinity.

        Returns:
            torch.Tensor: The gradient norms.
        """
        grads = [p.grad for p in self.wrapped_model.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            tensors=grads, norm_type=self.norm_type.value, error_if_nonfinite=False, foreach=True
        )
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()
        return total_norm


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
