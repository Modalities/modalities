import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh

from modalities.models.parallelism.expert_parallelism import ExpertParallel
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_mesh_for_parallelism_method
from modalities.util import get_module_class_from_name


def _validate_moe_block_for_ep(module) -> None:
    if not hasattr(module, "experts"):
        raise ValueError(f"Module {type(module).__name__} has no 'experts' attribute")

    experts = module.experts
    required_attrs = ["w1", "w2"]
    missing = [attr for attr in required_attrs if not hasattr(experts, attr)]
    if missing:
        raise ValueError(
            f"Module {type(module).__name__}.experts is not grouped-experts compatible. Missing: {missing}"
        )

    if experts.w1.ndim != 3 or experts.w2.ndim != 3:
        raise ValueError(
            f"Expected grouped expert parameters with ndim=3. Got w1.ndim={experts.w1.ndim}, "
            f"w2.ndim={experts.w2.ndim}"
        )


def _get_ep_target_module(module):
    if hasattr(module, "experts"):
        return module

    ffn = getattr(module, "ffn", None)
    if ffn is not None and hasattr(ffn, "experts"):
        return ffn

    return None


def _attach_ep_metadata(module, ep_mesh) -> None:
    setattr(module, "_ep_mesh", ep_mesh)
    setattr(module, "_ep_group", ep_mesh.get_group())
    setattr(module, "_ep_size", ep_mesh.size())
    setattr(module, "_ep_rank", ep_mesh.get_local_rank())


def get_ep_wrapped_model(
    model,
    block_names: list[str],
    device_mesh: DeviceMesh,
    mp_param_dtype=torch.bfloat16,
    mp_reduce_dtype=torch.bfloat16,
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
    MixedPrecisionPolicy(param_dtype=mp_param_dtype, reduce_dtype=mp_reduce_dtype)

    wrapped_blocks = 0
    for module in model.modules():
        if isinstance(module, block_types):
            ep_target_module = _get_ep_target_module(module)
            if ep_target_module is None:
                raise ValueError(
                    f"Module {type(module).__name__} has no EP-compatible experts location. "
                    "Expected `experts` or `ffn.experts`."
                )

            if getattr(ep_target_module, "_ep_enabled", False):
                continue

            _validate_moe_block_for_ep(ep_target_module)
            _attach_ep_metadata(ep_target_module, ep_mesh)

            ep_target_module.experts = ExpertParallel()._apply(ep_target_module.experts, ep_mesh)
            setattr(ep_target_module.experts, "_ep_enabled", True)

            wrapped_blocks += 1

    if wrapped_blocks == 0:
        raise ValueError(f"No blocks matched the requested types: {[t.__name__ for t in block_types]}")

    setattr(model, "_ep_wrapped", True)
    setattr(model, "_ep_mesh", ep_mesh)
    setattr(model, "_ep_num_wrapped_blocks", wrapped_blocks)

    return model
