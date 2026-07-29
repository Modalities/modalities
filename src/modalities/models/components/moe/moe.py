# Portions of this file are adapted from NVIDIA's Megatron-LM
# (megatron/core/transformer/moe/moe_utils.py::switch_load_balancing_loss_func): the
# load-balancing loss formula E * sum_i(f_i * P_i) and its sequence-level variant.
# Copyright (c) 2025, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.
#
# The dispatch/combine structure and the expert-bias / token-count buffers are inspired by Meta's
# open-source project TorchTitan (torchtitan/models/common/moe.py::MoE), licensed under the
# BSD 3-Clause License.

"""Mixture-of-experts feed-forward layer.

The layer combines a :class:`TopKRouter`, a :class:`GroupedExperts` stack and an optional always-on
shared expert. It implements the two load-balancing mechanisms that Nemotron-3 Nano uses together:

1. **Auxiliary-loss-free balancing** (https://arxiv.org/abs/2408.15664): the layer counts how many
   tokens each expert received; an optimizer hook nudges a per-expert selection bias towards the
   mean load. This is the primary mechanism and requires no gradient plumbing.
2. **The classic load-balancing loss** (Switch Transformer / Lepikhin et al.): a small
   differentiable penalty, computed per sequence, that discourages degenerate routing. Exposed via
   :attr:`last_aux_loss` for the training loss to pick up.
"""

import torch
import torch.nn as nn

from modalities.models.components.moe.experts import GroupedExperts
from modalities.models.components.moe.router import TopKRouter


class MoE(nn.Module):
    """
    A sparse mixture-of-experts feed-forward block.

    Tokens are routed to their top-k experts, sorted by expert so that each expert operates on a
    contiguous block, evaluated with grouped matmuls, and scattered back with their routing
    weights. A shared expert, if configured, processes every token densely and its output is added.
    """

    def __init__(
        self,
        router: TopKRouter,
        experts: GroupedExperts,
        shared_experts: nn.Module | None = None,
        aux_loss_coeff: float = 0.0,
    ):
        """
        Initializes the MoE layer.

        Args:
            router (TopKRouter): The routing module.
            experts (GroupedExperts): The routed expert stack.
            shared_experts (nn.Module | None): An optional dense module applied to every token.
                Nemotron-3 Nano uses two shared experts, realized as one MLP of twice the expert
                hidden dimension.
            aux_loss_coeff (float): Coefficient of the sequence-level load-balancing loss. Zero
                disables the loss (and its computation) entirely.

        Raises:
            ValueError: If ``aux_loss_coeff`` is negative or the router and experts disagree on
                the number of experts.
        """
        super().__init__()
        if aux_loss_coeff < 0:
            raise ValueError(f"aux_loss_coeff must be non-negative, got {aux_loss_coeff}.")
        if router.num_experts != experts.num_experts:
            raise ValueError(
                f"Router and experts disagree on the number of experts: "
                f"{router.num_experts} vs {experts.num_experts}."
            )

        self.router = router
        self.experts = experts
        self.shared_experts = shared_experts
        self.aux_loss_coeff = aux_loss_coeff
        # Overwritten on every forward pass. Deliberately not a buffer: it must stay in the
        # autograd graph and must not be checkpointed.
        self.last_aux_loss: torch.Tensor | None = None

    def _compute_aux_loss(self, scores: torch.Tensor, top_indices: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Computes the sequence-level load-balancing loss.

        Follows ``switch_load_balancing_loss_func`` of the reference implementation::

            loss = E * sum_i (f_i * P_i)

        with ``f_i`` the fraction of routed slots that went to expert ``i`` and ``P_i`` the mean
        router probability of expert ``i``. Computing it per sequence and averaging over the batch
        (rather than pooling the whole batch) is the ``seq_aux_loss`` variant: it prevents one
        sequence's routing from being balanced out by another's.

        Args:
            scores (torch.Tensor): Dense router scores of shape ``(num_tokens, num_experts)``.
            top_indices (torch.Tensor): Selected experts of shape ``(num_tokens, top_k)``.
            batch_size (int): The number of sequences in the batch.

        Returns:
            torch.Tensor: The scalar auxiliary loss, already multiplied by ``aux_loss_coeff``.
        """
        num_experts = self.router.num_experts
        top_k = self.router.top_k
        # Normalize over all experts so that P_i sums to one per token, matching the reference.
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        probs = probs.view(batch_size, -1, num_experts)

        routing_map = torch.zeros_like(scores).scatter_(-1, top_indices, 1.0)
        routing_map = routing_map.view(batch_size, -1, num_experts)

        tokens_per_seq = probs.shape[1]
        # f_i: fraction of the sequence's routed slots that landed on expert i.
        fraction_per_expert = routing_map.sum(dim=1) / (tokens_per_seq * top_k)
        # P_i: mean router probability assigned to expert i within the sequence.
        prob_per_expert = probs.mean(dim=1)
        per_sequence_loss = num_experts * (fraction_per_expert * prob_per_expert).sum(dim=-1)
        return self.aux_loss_coeff * per_sequence_loss.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: Output of shape ``(B, L, n_embd)``.
        """
        batch_size, seq_len, n_embd = x.shape
        x_flat = x.reshape(-1, n_embd)

        top_weights, top_indices, scores = self.router(x_flat)

        self.last_aux_loss = (
            self._compute_aux_loss(scores=scores, top_indices=top_indices, batch_size=batch_size)
            if self.aux_loss_coeff > 0
            else None
        )

        # Flatten the (token, slot) pairs and sort them by expert so that every expert's tokens
        # form one contiguous block, which is what the grouped matmul requires.
        flat_expert_indices = top_indices.reshape(-1)
        sort_order = torch.argsort(flat_expert_indices, stable=True)
        sorted_expert_indices = flat_expert_indices[sort_order]
        # Integer division recovers the token each routed slot belongs to.
        sorted_token_indices = sort_order // self.router.top_k

        tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=self.router.num_experts)

        # Track expert load for the auxiliary-loss-free bias update. Under activation
        # checkpointing the forward pass runs twice, so counts are inflated by a constant factor.
        # The bias update only uses the sign of the deviation from the mean, so this is harmless.
        with torch.no_grad():
            self.router.tokens_per_expert += tokens_per_expert.to(self.router.tokens_per_expert.dtype)

        x_sorted = x_flat[sorted_token_indices]
        expert_out = self.experts(x_sorted, tokens_per_expert)

        # Weight each routed slot by its routing probability and accumulate per token.
        sorted_weights = top_weights.reshape(-1)[sort_order].unsqueeze(-1)
        expert_out = expert_out * sorted_weights.to(expert_out.dtype)

        out_flat = torch.zeros_like(x_flat)
        out_flat.index_add_(0, sorted_token_indices, expert_out.to(out_flat.dtype))
        out = out_flat.view(batch_size, seq_len, n_embd)

        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out
