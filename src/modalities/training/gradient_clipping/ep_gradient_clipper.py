import math
from typing import Optional

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.tensor import DTensor

from modalities.running_env.fsdp.device_mesh import (
    ParallelismDegrees,
    get_mesh_for_parallelism_method,
    has_parallelism_method,
)
from modalities.training.gradient_clipping.fsdp_gradient_clipper import FSDP2GradientClipper, GradientClippingMode


class EPGradientClipper(FSDP2GradientClipper):
    """FSDP2 clipper wrapper that handles EP DTensor gradients safely."""

    def __init__(
        self,
        model_parts: FSDP2 | list[FSDP2],
        max_norm: float,
        norm_type: GradientClippingMode,
        device_mesh: Optional[DeviceMesh] = None,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> None:
        super().__init__(
            model_parts=model_parts,
            max_norm=max_norm,
            norm_type=norm_type,
            device_mesh=device_mesh,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    @torch.no_grad()
    def clip_gradients(self) -> torch.Tensor:
        grads = [p.grad for model in self.models for p in model.parameters() if p.grad is not None]

        if len(grads) == 0:
            device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
            total_norm = torch.tensor(0.0, device=device)
        else:
            norm_type_val = self.norm_type.value
            first_grad = grads[0]
            first_device = first_grad.to_local().device if isinstance(first_grad, DTensor) else first_grad.device
            norm_scalars: list[torch.Tensor] = []

            for grad in grads:
                grad_norm = torch.linalg.vector_norm(grad, ord=norm_type_val)
                if isinstance(grad_norm, DTensor):
                    grad_norm = grad_norm.full_tensor()
                norm_scalars.append(grad_norm.to(first_device))

            if math.isinf(norm_type_val):
                total_norm = torch.max(torch.stack(norm_scalars))
            else:
                total_norm = torch.linalg.vector_norm(torch.stack(norm_scalars), ord=norm_type_val)

            if self.error_if_nonfinite and (torch.isnan(total_norm) or torch.isinf(total_norm)):
                raise RuntimeError(
                    f"The total norm of order {norm_type_val} for gradients is non-finite: {total_norm.item()}"
                )

        if has_parallelism_method(self.device_mesh, ParallelismDegrees.PP):
            pp_mesh = get_mesh_for_parallelism_method(
                device_mesh=self.device_mesh, parallelism_method=ParallelismDegrees.PP
            )
            if math.isinf(self.norm_type.value):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= self.norm_type.value
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / self.norm_type.value

        clip_coef = self.max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

        for grad in grads:
            grad_device = grad.to_local().device if isinstance(grad, DTensor) else grad.device
            grad.mul_(clip_coef_clamped.to(grad_device))
        return total_norm
