from collections.abc import Sequence

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from modalities.models.gpt2.gpt2_model import AttentionImplementation, CausalSelfAttention


# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.
def apply_cp_to_sdpa_attention_forward(attention_modules: Sequence[nn.Module], cp_mesh: DeviceMesh) -> None:
    """Wrap SDPA attention forward methods with context parallel DTensor dispatch.

    This wrapper is intentionally minimal and only targets the QKV->SDPA path.
    It must run before tensor-parallel wrappers so CP logic executes inside local
    tensor regions.
    """
    if len(attention_modules) == 0:
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

    if getattr(CausalSelfAttention, "_cp_execute_attention_wrapped", False):
        return

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
    CausalSelfAttention._cp_execute_attention_wrapped = True
