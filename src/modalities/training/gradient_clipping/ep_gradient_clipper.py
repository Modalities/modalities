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
    """FSDP2 gradient clipper for models whose parameters live on different device meshes.

    With expert parallelism, the dense parameters are DTensors on the data parallel mesh, while the expert
    parameters are DTensors on the (expert FSDP x EP) mesh. Foreach ops cannot mix DTensors of different
    meshes, so the gradients are grouped by mesh (as in TorchTitan's EP-aware clip_grad_norm_).
    The norm of each group is reduced with a single collective, and the group norms are combined locally.
    """

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
        params_by_mesh: dict[Optional[DeviceMesh], list[torch.nn.Parameter]] = {}
        for model in self.models:
            for p in model.parameters():
                if p.grad is None:
                    continue
                mesh = p.grad.device_mesh if isinstance(p.grad, DTensor) else None
                params_by_mesh.setdefault(mesh, []).append(p)

        norm_type = self.norm_type.value
        group_norms: list[torch.Tensor] = []
        for params in params_by_mesh.values():
            group_norm = torch.nn.utils.get_total_norm(
                tensors=[p.grad for p in params], norm_type=norm_type, foreach=self.foreach
            )
            if isinstance(group_norm, DTensor):
                # one collective per mesh (reduces the partial norm over all shards)
                group_norm = group_norm.full_tensor()
            group_norms.append(group_norm)

        if len(group_norms) == 0:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
            total_norm = torch.tensor(0.0, device=device)
        elif math.isinf(norm_type):
            total_norm = torch.max(torch.stack(group_norms))
        else:
            total_norm = torch.linalg.vector_norm(torch.stack(group_norms), ord=norm_type)

        if has_parallelism_method(self.device_mesh, ParallelismDegrees.PP):
            pp_mesh = get_mesh_for_parallelism_method(
                device_mesh=self.device_mesh, parallelism_method=ParallelismDegrees.PP
            )
            if math.isinf(norm_type):
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
            else:
                total_norm **= norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
                total_norm **= 1.0 / norm_type

        if self.error_if_nonfinite and not torch.isfinite(total_norm):
            raise RuntimeError(f"The total norm of order {norm_type} for gradients is non-finite: {total_norm.item()}")

        for params in params_by_mesh.values():
            torch.nn.utils.clip_grads_with_norm_(
                parameters=params, max_norm=self.max_norm, total_norm=total_norm, foreach=self.foreach
            )
        return total_norm
