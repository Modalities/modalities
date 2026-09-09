"""Expert parallelism for mixture-of-experts layers.

Under expert parallelism the routed experts of a layer are partitioned across the ``ep`` mesh
dimension, so each rank stores and evaluates only ``num_experts / ep_degree`` of them. Because every
rank still routes its *own* tokens, tokens have to travel to the rank that owns their expert and
their results have to travel back. That is the dispatch/combine pair of all-to-all collectives this
module inserts around :class:`~modalities.models.components.moe.experts.GroupedExperts`.

The alternative -- what modalities does without expert parallelism -- is to let FSDP2 all-gather the
full expert stack on every rank. For Nemotron-3 Nano 30B-A3B that means moving 2.55 GiB of bf16
expert weights per MoE layer per forward, versus roughly a sixth of that in tokens here, which is
what makes expert parallelism worthwhile at scale.

Design notes
------------
The dispatch/combine structure follows Meta's open-source project TorchTitan
(``torchtitan/models/common/moe.py``, ``torchtitan/distributed/expert_parallel.py``), licensed under
the BSD 3-Clause License.

Two things are done deliberately differently from a straightforward implementation:

1. **The post-all-to-all permutation is built entirely on device.** Received tokens arrive grouped by
   *sender* and have to be regrouped by *local expert* before a grouped matmul can consume them. The
   obvious implementation loops over ``(sender, local_expert)`` pairs and calls ``.item()`` to slice
   each range, which costs ``2 * ep_degree * num_local_experts`` device-to-host synchronizations per
   MoE layer per forward pass -- several thousand per step for a 23-MoE-layer model. Here the index
   tensor is constructed with vectorized ops instead (see :func:`_build_permute_indices`).
2. **There is exactly one device-to-host synchronization per dispatch**, for the split lists that
   ``all_to_all_single`` requires on the host. Both split vectors are copied in a single transfer.

Autograd flows through both all-to-all calls via ``all_to_all_single_autograd`` (the adjoint of an
all-to-all is an all-to-all with the splits swapped) and through the gather/scatter permutation.
"""

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import all_to_all_single, all_to_all_single_autograd
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from modalities.models.components.moe.experts import GroupedExperts


def _build_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
    total_tokens: int,
) -> torch.Tensor:
    """
    Builds the index tensor that regroups received tokens from sender-major to expert-major order.

    After the token all-to-all, the receive buffer is ordered by sender and, within a sender, by
    local expert::

        [r0e0, r0e1, ..., r0e(L-1)] [r1e0, r1e1, ...] ... [r(P-1)e0, ...]

    A grouped matmul needs all tokens of one expert to be contiguous, i.e. expert-major order::

        [e0r0, e0r1, ..., e0r(P-1)] [e1r0, ...] ... [e(L-1)r0, ...]

    Both orders are concatenations of the same ``P * L`` variable-length blocks, so the permutation
    is fully determined by the block lengths. This function turns those lengths into the flat index
    tensor without a single host synchronization.

    Args:
        tokens_per_expert_group (torch.Tensor): Received token counts of shape
            ``(ep_degree * num_local_experts,)``, ordered by sender then by local expert.
        ep_degree (int): The expert parallel degree, i.e. the number of senders ``P``.
        num_local_experts (int): The number of experts owned by this rank, ``L``.
        total_tokens (int): The sum of ``tokens_per_expert_group``. Passed in as a Python int because
            the caller already has it on the host from the all-to-all split lists; recomputing it
            here would reintroduce a synchronization.

    Returns:
        torch.Tensor: Int64 gather indices of shape ``(total_tokens,)`` such that
            ``received[indices]`` is in expert-major order.
    """
    device = tokens_per_expert_group.device
    counts = tokens_per_expert_group.view(ep_degree, num_local_experts)
    # Exclusive cumulative sum over the receive buffer gives each block's start offset.
    flat_counts = counts.reshape(-1)
    block_starts = flat_counts.cumsum(0) - flat_counts

    # Transpose both into expert-major (expert, sender) order, which is the output order.
    counts_out = counts.t().reshape(-1)
    starts_out = block_starts.view(ep_degree, num_local_experts).t().reshape(-1)

    if total_tokens == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    # For every output position, find which block it belongs to and its offset within that block.
    # repeat_interleave needs output_size to stay synchronization-free.
    block_of_position = torch.repeat_interleave(
        torch.arange(counts_out.numel(), device=device),
        counts_out,
        output_size=total_tokens,
    )
    offset_in_block = torch.arange(total_tokens, device=device) - (counts_out.cumsum(0) - counts_out)[block_of_position]
    return starts_out[block_of_position] + offset_in_block


class ExpertParallelGroupedExperts(nn.Module):
    """
    Wraps a :class:`GroupedExperts` stack whose experts are partitioned across the ``ep`` mesh dim.

    Drop-in replacement for the wrapped module: the forward signature
    ``(tokens_sorted_by_expert, tokens_per_expert)`` and the returned shape are unchanged, so
    :class:`~modalities.models.components.moe.moe.MoE` needs no knowledge of expert parallelism. The
    ``tokens_per_expert`` it passes in are *global* counts over all ``num_experts``; the wrapped
    module receives the local counts for the experts this rank owns.
    """

    def __init__(self, experts: GroupedExperts, ep_mesh: DeviceMesh):
        """
        Initializes the expert-parallel wrapper.

        Args:
            experts (GroupedExperts): The expert stack. Its ``w1``/``w2`` are expected to already be
                sharded over ``ep_mesh`` along the expert dimension (see
                :func:`shard_experts_over_ep_mesh`).
            ep_mesh (DeviceMesh): The one-dimensional ``ep`` sub-mesh whose process group carries the
                dispatch and combine all-to-all collectives.

        Raises:
            ValueError: If the mesh is not one-dimensional, or if the number of experts is not
                divisible by the expert parallel degree.
        """
        super().__init__()
        if ep_mesh.ndim != 1:
            raise ValueError(f"ep_mesh must be one-dimensional, got ndim={ep_mesh.ndim}.")
        ep_degree = ep_mesh.size()
        if experts.num_experts % ep_degree != 0:
            raise ValueError(
                f"num_experts ({experts.num_experts}) must be divisible by the expert parallel "
                f"degree ({ep_degree})."
            )

        self.experts = experts
        self.ep_mesh = ep_mesh
        self.ep_degree = ep_degree
        self.num_local_experts = experts.num_experts // ep_degree

    @property
    def num_experts(self) -> int:
        """The *global* number of experts, which is what the router is validated against."""
        return self.experts.num_experts

    def _dispatch(
        self, x_sorted: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
        """
        Sends every token to the rank owning its expert and regroups the receive buffer by expert.

        Args:
            x_sorted (torch.Tensor): Locally routed tokens sorted by global expert index, of shape
                ``(num_routed, n_embd)``.
            tokens_per_expert (torch.Tensor): Local token counts per *global* expert, of shape
                ``(num_experts,)``.

        Returns:
            tuple: ``(x_received, local_tokens_per_expert, permute_indices, input_splits,
            output_splits)``, where ``x_received`` is in expert-major order and
            ``local_tokens_per_expert`` has shape ``(num_local_experts,)``.
        """
        group = self.ep_mesh.get_group()

        with torch.no_grad():
            # Experts are assigned to ranks in contiguous blocks, matching Shard(0) on the expert
            # dimension, so an even-split all-to-all sends each rank exactly the counts of the
            # experts it owns.
            tokens_per_expert_group = all_to_all_single(tokens_per_expert, None, None, group=group)
            tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(tokens_per_expert_group)
            # all_to_all_single takes its split lists on the host. Stacking both vectors keeps this
            # to a single device-to-host copy, which is the only synchronization in the dispatch.
            splits = torch.stack(
                (
                    tokens_per_expert.view(self.ep_degree, -1).sum(dim=1),
                    tokens_per_expert_group.view(self.ep_degree, -1).sum(dim=1),
                )
            ).to(device="cpu", non_blocking=False)
        input_splits: list[int] = splits[0].tolist()
        output_splits: list[int] = splits[1].tolist()

        x_received = all_to_all_single_autograd(x_sorted, output_splits, input_splits, group)
        permute_indices = _build_permute_indices(
            tokens_per_expert_group=tokens_per_expert_group,
            ep_degree=self.ep_degree,
            num_local_experts=self.num_local_experts,
            total_tokens=sum(output_splits),
        )
        local_tokens_per_expert = tokens_per_expert_group.view(self.ep_degree, self.num_local_experts).sum(dim=0)
        return (
            x_received.index_select(0, permute_indices),
            local_tokens_per_expert,
            permute_indices,
            input_splits,
            output_splits,
        )

    def _combine(
        self,
        expert_out: torch.Tensor,
        permute_indices: torch.Tensor,
        num_received: int,
        input_splits: list[int],
        output_splits: list[int],
    ) -> torch.Tensor:
        """
        Undoes the expert-major regrouping and returns each token's result to its owning rank.

        Args:
            expert_out (torch.Tensor): Expert outputs in expert-major order.
            permute_indices (torch.Tensor): The indices produced by :meth:`_dispatch`.
            num_received (int): Row count of the dispatch receive buffer.
            input_splits (list[int]): The dispatch input splits; the combine output splits.
            output_splits (list[int]): The dispatch output splits; the combine input splits.

        Returns:
            torch.Tensor: Results in the caller's original expert-sorted order.
        """
        # Scatter back to sender-major order. index_copy is the differentiable inverse of the
        # index_select used in the dispatch.
        unpermuted = expert_out.new_zeros((num_received, expert_out.shape[-1])).index_copy(
            0, permute_indices, expert_out
        )
        # Splits swap relative to the dispatch: what we received, we now send back.
        return all_to_all_single_autograd(unpermuted, input_splits, output_splits, self.ep_mesh.get_group())

    def forward(self, x_sorted: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the globally distributed expert stack on locally routed tokens.

        Args:
            x_sorted (torch.Tensor): Tokens sorted by global expert index, shape
                ``(num_routed, n_embd)``.
            tokens_per_expert (torch.Tensor): Token counts per global expert, shape
                ``(num_experts,)``.

        Returns:
            torch.Tensor: Expert outputs of shape ``(num_routed, n_embd)``, in the same order as
                ``x_sorted``.
        """
        x_received, local_tokens_per_expert, permute_indices, input_splits, output_splits = self._dispatch(
            x_sorted, tokens_per_expert
        )
        expert_out = self.experts(x_received, local_tokens_per_expert)
        return self._combine(
            expert_out=expert_out,
            permute_indices=permute_indices,
            num_received=x_received.shape[0],
            input_splits=input_splits,
            output_splits=output_splits,
        )


def shard_experts_over_ep_mesh(experts: GroupedExperts, ep_mesh: DeviceMesh) -> None:
    """
    Replaces an expert stack's weights in place with DTensors sharded over the expert dimension.

    The parameters are re-created rather than redistributed: this runs on a meta-device model whose
    weights are materialized and initialized later, so there is no data to preserve. Keeping them as
    DTensors (rather than plain per-rank slices) means checkpoints still see the full global expert
    stack, and lets FSDP2 shard what remains of the data-parallel dimension on top.

    Args:
        experts (GroupedExperts): The expert stack to shard in place.
        ep_mesh (DeviceMesh): The one-dimensional ``ep`` sub-mesh to shard over.

    Raises:
        ValueError: If a weight's expert dimension is not divisible by the expert parallel degree.
    """
    ep_degree = ep_mesh.size()
    for name in ("w1", "w2"):
        param = getattr(experts, name)
        if isinstance(param, DTensor):
            continue
        num_experts = param.shape[0]
        if num_experts % ep_degree != 0:
            raise ValueError(
                f"Expert dimension of {name} ({num_experts}) is not divisible by the expert "
                f"parallel degree ({ep_degree})."
            )
        local_shape = (num_experts // ep_degree, *param.shape[1:])
        local = torch.empty(local_shape, dtype=param.dtype, device=param.device)
        sharded = DTensor.from_local(local, ep_mesh, [Shard(0)], run_check=False)
        experts.register_parameter(name, nn.Parameter(sharded, requires_grad=param.requires_grad))
