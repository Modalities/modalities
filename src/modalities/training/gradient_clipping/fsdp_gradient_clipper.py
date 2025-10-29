import math
from typing import Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.tensor import DTensor

from modalities.config.lookup_enum import LookupEnum
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_mesh_for_parallelism_method
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


class FSDP2LoggingOnlyGradientClipper(GradientClipperIF):
    """The FSDP2LoggingOnlyGradientClipper class that is responsible for logging the gradient
    norms without actually clipping the gradients."""

    def __init__(
        self,
        wrapped_model: FSDP2,
        norm_type: GradientClippingMode,
        device_mesh: Optional[DeviceMesh] = None,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> None:
        """
        Initialize the FSDP2LoggingOnlyGradientClipper.

        Args:
            wrapped_model (FSDP2): The wrapped FSDP2 model.
            norm_type (GradientClippingMode): The type of gradient clipping.
            device_mesh (DeviceMesh, optional): The device mesh used for distributed training. Defaults to None.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            None
        """
        self.wrapped_model = wrapped_model
        self.norm_type = norm_type
        self.device_mesh = device_mesh
        self.error_if_nonfinite = error_if_nonfinite
        self.foreach = foreach

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Returns the gradient norm, but does not apply clipping since max_norm is set to inifinity.

        Returns:
            torch.Tensor: The gradient norms.
        """
        grads = [p.grad for p in self.wrapped_model.parameters() if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            tensors=grads,
            norm_type=self.norm_type.value,
            error_if_nonfinite=self.error_if_nonfinite,
            foreach=self.foreach,
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

        pp_mesh = get_mesh_for_parallelism_method(
            device_mesh=self.device_mesh, parallelism_method=ParallelismDegrees.PP
        )
        if pp_mesh is not None:
            if math.isinf(self.norm_type.value):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= self.norm_type.value
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / self.norm_type.value
        return total_norm


class FSDP2GradientClipper(FSDP2LoggingOnlyGradientClipper):
    """The FSDP2GradientClipper class that is responsible for clipping the gradients of a model wrapped with FSDP."""

    def __init__(
        self,
        wrapped_model: FSDP2,
        max_norm: float,
        norm_type: GradientClippingMode,
        device_mesh: Optional[DeviceMesh] = None,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> None:
        """
        Initialize the FSDP2GradientClipper object.

        Args:
            wrapped_model (FSDP2): The wrapped FSDP2 model.
            max_norm (float): The maximum norm value for gradient clipping.
            norm_type (GradientClippingMode): The type of gradient clipping.
            device_mesh (DeviceMesh, optional): The device mesh used for distributed training. Defaults to None.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            None
        """
        super().__init__(
            wrapped_model=wrapped_model,
            norm_type=norm_type,
            device_mesh=device_mesh,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
        self.max_norm = max_norm

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        """
        Clips the gradients of the wrapped model using the specified maximum norm and norm type.

        Returns:
            torch.Tensor: The gradient norm after clipping.
        """
        total_norm = super().clip_gradients()
        torch.nn.utils.clip_grads_with_norm_(
            parameters=self.wrapped_model.parameters(),
            max_norm=self.max_norm,
            total_norm=total_norm,
            foreach=self.foreach,
        )
        return total_norm
