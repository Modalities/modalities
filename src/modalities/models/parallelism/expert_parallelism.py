# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import all_to_all_single, all_to_all_single_autograd
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor


def expert_major_permutation(counts: Tensor, num_sources: int, num_local_experts: int, num_rows: int) -> Tensor:
    """
    Computes, without any host synchronization, the gather index that reorders rows from the
    source-major layout produced by the all-to-all
      [src0: e0 | e1 | ... , src1: e0 | e1 | ..., ...]
    into the expert-major layout expected by the experts
      [e0: src0 | src1 | ..., e1: src0 | src1 | ..., ...].

    Args:
        counts (Tensor): Number of rows per (source, local expert), flattened source-major.
        num_sources (int): Number of sources (EP degree).
        num_local_experts (int): Number of experts on this rank.
        num_rows (int): Total number of rows, i.e. counts.sum(). Must be known on the host.

    Returns:
        Tensor: Index such that x[index] is in expert-major order.
    """
    counts = counts.view(num_sources, num_local_experts)
    flat_counts = counts.reshape(-1)
    src_start = (torch.cumsum(flat_counts, dim=0) - flat_counts).view(num_sources, num_local_experts)
    seg_counts = counts.t().reshape(-1)
    seg_src_start = src_start.t().reshape(-1)
    seg_dst_start = torch.cumsum(seg_counts, dim=0) - seg_counts
    shift = torch.repeat_interleave(seg_src_start - seg_dst_start, seg_counts, output_size=num_rows)
    return shift + torch.arange(num_rows, device=counts.device, dtype=shift.dtype)


class ExpertParallel:
    """
    Expert Parallelism for grouped-expert MoE layers.

    Shards GroupedExperts parameters on the expert dimension (Shard(0)) across EP ranks,
    and wraps forward() with all-to-all token dispatch/combine collectives.

    The only host synchronization per forward is a single device-to-host copy of the
    (ep_degree x num_local_experts) token count matrix, which is required to issue the
    variable-sized token all-to-all. All index computations stay on the device.

    Usage:
        module.experts = ExpertParallel()._apply(module.experts, ep_mesh)
    """

    def __init__(self) -> None:
        # Per-call state, written in _token_dispatch and consumed in _token_combine of the same forward call.
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.permuted_indices: Tensor | None = None
        self.num_dropped_rows: int = 0

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )

    def _token_dispatch(self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh) -> tuple:
        routed_input, num_tokens_per_expert = inputs[0], inputs[1]
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree
        group = device_mesh.get_group()

        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(num_tokens_per_expert, None, None, group=group)
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(num_tokens_per_expert_group)
            # The single host synchronization of this layer: both count matrices in one copy.
            counts_cpu = (
                torch.stack([num_tokens_per_expert, num_tokens_per_expert_group])
                .to("cpu")
                .view(2, ep_degree, num_local_experts)
            )
            self.input_splits = counts_cpu[0].sum(dim=1).tolist()
            self.output_splits = counts_cpu[1].sum(dim=1).tolist()
            num_tokens_per_local_expert_cpu = counts_cpu[1].sum(dim=0).tolist()

        # Rows beyond sum(input_splits) are tokens dropped by a capacity limit; they are not dispatched.
        num_valid_rows = sum(self.input_splits)
        self.num_dropped_rows = routed_input.shape[0] - num_valid_rows
        if self.num_dropped_rows > 0:
            routed_input = routed_input[:num_valid_rows]

        routed_input = all_to_all_single_autograd(routed_input, self.output_splits, self.input_splits, group)

        num_received_rows = sum(self.output_splits)
        if num_local_experts > 1:
            with torch.no_grad():
                self.permuted_indices = expert_major_permutation(
                    num_tokens_per_expert_group, ep_degree, num_local_experts, num_received_rows
                )
            routed_input = routed_input.index_select(0, self.permuted_indices)
        else:
            self.permuted_indices = None

        num_tokens_per_local_expert = num_tokens_per_expert_group.view(ep_degree, num_local_experts).sum(dim=0)
        return routed_input, num_tokens_per_local_expert, num_tokens_per_local_expert_cpu

    def _token_combine(self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh) -> Tensor:
        if self.permuted_indices is not None:
            routed_output = torch.index_copy(torch.empty_like(routed_output), 0, self.permuted_indices, routed_output)
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        if self.num_dropped_rows > 0:
            routed_output = torch.cat(
                [routed_output, routed_output.new_zeros(self.num_dropped_rows, routed_output.shape[-1])]
            )
        self.permuted_indices = None
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
