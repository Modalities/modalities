# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single, all_to_all_single_autograd
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor


def _permute_tokens(
    x: Tensor,
    num_tokens_per_expert_group: Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[tuple, Tensor, Tensor, Tensor]:
    """
    Reorder tokens from the post-all-to-all layout to per-local-expert contiguous layout.

    After the all-to-all, received tokens are ordered as:
      [e0_from_rank0 tokens, e1_from_rank0 tokens, ..., e0_from_rank1 tokens, ...]

    We reorder to:
      [all tokens for local_expert_0, all tokens for local_expert_1, ...]

    Returns (original_shape, x_permuted, permuted_indices, new_num_tokens_per_expert).
    """
    counts = num_tokens_per_expert_group.view(ep_degree, num_local_experts)  # (ep_degree, num_local_experts)

    flat_counts = counts.flatten()  # length = ep_degree * num_local_experts
    
    offsets = flat_counts.cumsum(0) - flat_counts

    # build permuted_indices
    indices_per_expert: list[Tensor] = []
    for e in range(num_local_experts):
        for r in range(ep_degree):
            count = int(counts[r, e].item())
            if count > 0:
                start = int(offsets[r * num_local_experts + e].item())
                indices_per_expert.append(
                    torch.arange(start, start + count, device=x.device, dtype=torch.long)
                )

    if indices_per_expert:
        permuted_indices = torch.cat(indices_per_expert)
    else:
        permuted_indices = torch.zeros(0, dtype=torch.long, device=x.device)

    new_num_tokens_per_expert = counts.sum(dim=0)  # (num_local_experts,)
    original_shape = x.shape
    x_permuted = x[permuted_indices] if permuted_indices.numel() > 0 else x.new_zeros((0, x.shape[-1]))
    return original_shape, x_permuted, permuted_indices, new_num_tokens_per_expert


def _unpermute_tokens(out: Tensor, original_shape: tuple, permuted_indices: Tensor) -> Tensor:
    """
    Inverse of _permute_tokens: scatter expert outputs back to the all-to-all layout.
    """
    out_unpermuted = out.new_zeros(original_shape)
    if permuted_indices.numel() > 0:
        out_unpermuted[permuted_indices] = out
    return out_unpermuted


class ExpertParallel:
    """
    Expert Parallelism for grouped-expert MoE layers.

    Shards GroupedExperts parameters on the expert dimension (Shard(0)) across EP ranks,
    and wraps forward() with all-to-all token dispatch/combine collectives.

    Usage:
        module.experts = ExpertParallel()._apply(module.experts, ep_mesh)
    """

    def __init__(self) -> None:
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.original_shape: tuple | None = None
        self.permuted_indices: Tensor | None = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert, None, None, group=device_mesh.get_group()
            )
            
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        self.original_shape, routed_input, self.permuted_indices, num_tokens_per_expert_group = (
            _permute_tokens(routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts)
        )
        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute_tokens(routed_output, self.original_shape, self.permuted_indices)
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
