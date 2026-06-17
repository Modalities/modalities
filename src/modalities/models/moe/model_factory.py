import warnings

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from modalities.models.parallelism.expert_parallelism import ExpertParallel
from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_mesh_for_parallelism_method
from modalities.util import get_module_class_from_name


def get_ep_wrapped_model(
    model,
    block_names: list[str],
    device_mesh: DeviceMesh,
    mixed_precision_settings: FSDP2MixedPrecisionSettings,
) -> nn.Module:
    block_types = []
    missing_block_names = []
    for name in block_names:
        block_type = get_module_class_from_name(model, name)
        if block_type is None:
            missing_block_names.append(name)
        else:
            block_types.append(block_type)

    if len(missing_block_names) > 0 and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        warnings.warn(
            f"Could not resolve some requested MoE block names and they will be ignored: {missing_block_names}",
            stacklevel=2,
        )

    block_types = tuple(block_types)

    if len(block_types) == 0:
        raise ValueError(f"None of the requested MoE block names were found: {block_names}")

    ep_mesh = get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.EP)
    target_dtype = mixed_precision_settings.param_dtype.value

    wrapped_blocks = 0
    for module in model.modules():
        if isinstance(module, block_types):
            if hasattr(module, "experts"):
                ep_target = module
            elif (ffn := getattr(module, "ffn", None)) is not None and hasattr(ffn, "experts"):
                ep_target = ffn
            else:
                raise ValueError(
                    f"Module {type(module).__name__} has no EP-compatible experts location. "
                    "Expected `experts` or `ffn.experts`."
                )

            if getattr(ep_target, "_ep_enabled", False):
                continue

            experts = ep_target.experts
            missing = [a for a in ("w1", "w2") if not hasattr(experts, a)]
            if missing:
                raise ValueError(
                    f"Module {type(ep_target).__name__}.experts is not grouped-experts compatible. Missing: {missing}"
                )
            if experts.w1.ndim != 3 or experts.w2.ndim != 3:
                raise ValueError(
                    f"Expected grouped expert parameters with ndim=3. Got w1.ndim={experts.w1.ndim}, "
                    f"w2.ndim={experts.w2.ndim}"
                )

            ep_target._ep_mesh = ep_mesh
            ep_target._ep_group = ep_mesh.get_group()
            ep_target._ep_size = ep_mesh.size()
            ep_target._ep_rank = ep_mesh.get_local_rank()

            ep_target.experts = ExpertParallel()._apply(ep_target.experts, ep_mesh)
            ep_target.experts._ep_enabled = True

            for pname, p in list(ep_target.experts._parameters.items()):
                if isinstance(p, DTensor) and p.dtype != target_dtype:
                    local = p.to_local().to(target_dtype)
                    ep_target.experts._parameters[pname] = nn.Parameter(
                        DTensor.from_local(local, p.device_mesh, p.placements, run_check=False),
                        requires_grad=p.requires_grad,
                    )

            wrapped_blocks += 1

    if wrapped_blocks == 0:
        raise ValueError(f"No blocks matched the requested types: {[t.__name__ for t in block_types]}")

    model._ep_wrapped = True
    model._ep_mesh = ep_mesh
    model._ep_num_wrapped_blocks = wrapped_blocks

    return model
