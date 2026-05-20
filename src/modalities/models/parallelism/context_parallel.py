from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from modalities.models.gpt2.gpt2_model import AttentionImplementation, CausalSelfAttention


# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.
def apply_cp_to_sdpa_attention_forward(attention_modules: Sequence[nn.Module], cp_mesh: DeviceMesh) -> None:
    """Patch CausalSelfAttention.execute_attention to route SDPA through DTensor CP dispatch.

    The patch is class-level (not per-instance). `attention_modules` is used only as a
    guard: if the list is empty no patching happens, since the model has no SDPA layers to
    wrap. The individual module objects are not modified.

    It must run before tensor-parallel wrappers so CP logic executes inside local
    tensor regions.
    """
    if len(attention_modules) == 0:
        return

    # Detect re-entry. A second call with the same mesh is a no-op; a different mesh
    # would silently use the wrong mesh because cp_mesh is captured by closure.
    existing_mesh = getattr(CausalSelfAttention, "_cp_mesh", None)
    if getattr(CausalSelfAttention, "_cp_execute_attention_wrapped", False):
        if existing_mesh is not cp_mesh:
            raise RuntimeError(
                "apply_cp_to_sdpa_attention_forward has already patched CausalSelfAttention "
                "with a different cp_mesh. Re-patching with a new mesh is not supported."
            )
        return

    try:
        from torch.distributed.tensor import DTensor, Shard
        from torch.distributed.tensor.experimental._attention import _enable_context_parallel_dispatcher
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Context parallelism requires PyTorch experimental DTensor attention APIs. "
            "Install a PyTorch build that provides "
            "torch.distributed.tensor.experimental._attention."
        ) from exc

    _enable_context_parallel_dispatcher()

    original_execute_attention = CausalSelfAttention.execute_attention

    def cp_execute_attention(cls, q, k, v, dropout, attention_impl):
        if attention_impl != AttentionImplementation.PYTORCH_FLASH:
            return original_execute_attention(q, k, v, dropout, attention_impl)

        placement = [Shard(2)]
        if not isinstance(q, DTensor):
            q = DTensor.from_local(q, cp_mesh, placement, run_check=False)
        if not isinstance(k, DTensor):
            k = DTensor.from_local(k, cp_mesh, placement, run_check=False)
        if not isinstance(v, DTensor):
            v = DTensor.from_local(v, cp_mesh, placement, run_check=False)

        output = original_execute_attention(q, k, v, dropout, attention_impl)
        return output.to_local() if isinstance(output, DTensor) else output

    CausalSelfAttention.execute_attention = classmethod(cp_execute_attention)
    setattr(CausalSelfAttention, "_cp_execute_attention_wrapped", True)
    setattr(CausalSelfAttention, "_cp_mesh", cp_mesh)


def shard_tensor_buffers_for_context_parallel(
    cp_mesh: DeviceMesh,
    buffers: tuple[torch.Tensor, ...],
    seq_dims: tuple[int, ...],
    load_balancer_type: str | None = "headtail",
    shard_impl=None,
) -> tuple[torch.Tensor, ...]:
    """Shard tensor buffers across CP ranks along sequence dimensions.

    This mirrors TorchTitan's input sharding pattern while keeping the current
    codebase focused on plain tensor inputs/targets (no attention mask sharding yet).
    """
    if len(buffers) != len(seq_dims):
        raise ValueError(f"Expected len(buffers) == len(seq_dims), got {len(buffers)} and {len(seq_dims)}.")
    if len(buffers) == 0:
        return tuple()

    if shard_impl is None:
        try:
            from torch.distributed.tensor.experimental._attention import _context_parallel_shard, _HeadTailLoadBalancer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Context parallel input sharding requires PyTorch experimental DTensor attention APIs."
            ) from exc
        shard_impl = _context_parallel_shard

        if load_balancer_type == "headtail":
            seq_len = buffers[0].size(seq_dims[0])
            cp_world_size = cp_mesh.size()
            load_balancer = _HeadTailLoadBalancer(seq_len, cp_world_size, cp_mesh.device_type)
        elif load_balancer_type is None:
            load_balancer = None
        elif load_balancer_type == "ptrr":
            raise ValueError(
                "PTRR load balancing is not supported for plain tensor input/target sharding without block masks."
            )
        else:
            raise ValueError(
                f"Invalid load_balancer_type '{load_balancer_type}'. Must be one of: 'headtail', 'ptrr', or None"
            )
    else:
        # Tests can inject shard_impl and bypass private PyTorch imports.
        load_balancer = None

    sharded = shard_impl(
        mesh=cp_mesh,
        buffers=buffers,
        seq_dims=seq_dims,
        load_balancer=load_balancer,
    )
    return tuple(sharded)
