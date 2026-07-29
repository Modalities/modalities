# Portions of this file are adapted from NVIDIA's Megatron-LM
# (megatron/core/transformer/moe/moe_utils.py::topk_routing_with_score_function): the sigmoid
# scoring path, the selection-only expert bias, and the top-k renormalization.
# Copyright (c) 2025, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0.
#
# The router interface is additionally inspired by Meta's open-source project TorchTitan
# (torchtitan/models/common/moe.py::TokenChoiceTopKRouter), licensed under the BSD 3-Clause License.

"""Top-k mixture-of-experts router.

Implements the routing scheme used by Nemotron-3 Nano: sigmoid gating with an additive
load-balancing bias that influences expert *selection* only, followed by a renormalization over
the selected experts and a constant rescaling. This matches the ``sigmoid`` branch of
``megatron.core.transformer.moe.moe_utils.topk_routing_with_score_function``.
"""

from enum import Enum

import torch
import torch.nn as nn


class RouterScoreFunction(str, Enum):
    """
    Enum of the supported router score functions.

    Attributes:
        SIGMOID (str): Independent per-expert sigmoid scores, renormalized over the selected
            experts. Used by Nemotron-3 Nano and DeepSeek-style MoEs.
        SOFTMAX (str): A softmax over the selected experts' logits.
    """

    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


class TopKRouter(nn.Module):
    """
    Routes each token to its top-k experts.

    The router is a single bias-free linear layer ("learnt MLP router"). Its scores are computed
    in float32 regardless of the activation dtype, because top-k selection over 128 nearly-tied
    scores is sensitive to rounding and any instability there shows up as expert flapping.
    """

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int,
        score_function: RouterScoreFunction = RouterScoreFunction.SIGMOID,
        route_scale: float = 1.0,
        use_expert_bias: bool = True,
        router_dtype: torch.dtype = torch.float32,
    ):
        """
        Initializes the TopKRouter.

        Args:
            n_embd (int): The model dimension.
            num_experts (int): The number of routable experts.
            top_k (int): How many experts each token is routed to.
            score_function (RouterScoreFunction): Which score function to use.
            route_scale (float): Constant factor applied to the final routing weights. Nemotron-3
                Nano uses 2.5 to compensate for the renormalization shrinking the weights.
            use_expert_bias (bool): Whether to maintain an additive per-expert selection bias for
                auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664).
            router_dtype (torch.dtype): Dtype in which scores and top-k selection are computed.

        Raises:
            ValueError: If ``top_k`` is not in ``[1, num_experts]``.
        """
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}.")

        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_function = RouterScoreFunction(score_function)
        self.route_scale = route_scale
        self.router_dtype = router_dtype

        self.gate = nn.Linear(in_features=n_embd, out_features=num_experts, bias=False)

        if use_expert_bias:
            # Persistent: the bias is part of the model state and must survive checkpointing.
            self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32), persistent=True)
        else:
            self.expert_bias = None
        # Non-persistent: a per-step counter that is reduced and reset by the load-balancing hook.
        self.register_buffer("tokens_per_expert", torch.zeros(num_experts, dtype=torch.float32), persistent=False)

    def reset_parameters(self) -> None:
        """Zeroes the load-balancing state. The gate weight is initialized by the model initializer."""
        with torch.no_grad():
            if self.expert_bias is not None and self.expert_bias.device.type != "meta":
                self.expert_bias.zero_()
            if self.tokens_per_expert.device.type != "meta":
                self.tokens_per_expert.zero_()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the routing decision for every token.

        Args:
            x (torch.Tensor): Input of shape ``(num_tokens, n_embd)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ``top_weights`` of shape ``(num_tokens, top_k)``: the routing weights, in the
                  dtype of ``x``.
                - ``top_indices`` of shape ``(num_tokens, top_k)``: the selected expert indices.
                - ``scores`` of shape ``(num_tokens, num_experts)``: the dense scores, kept for
                  the auxiliary load-balancing loss.
        """
        logits = self.gate(x).to(self.router_dtype)

        if self.score_function == RouterScoreFunction.SIGMOID:
            scores = torch.sigmoid(logits)
            # The bias steers *selection* only; the returned weights come from the unbiased
            # scores. Otherwise the balancing bias would leak into the forward signal.
            selection_scores = scores if self.expert_bias is None else scores + self.expert_bias.to(scores.dtype)
            _, top_indices = torch.topk(selection_scores, k=self.top_k, dim=-1)
            top_weights = torch.gather(scores, dim=-1, index=top_indices)
            if self.top_k > 1:
                top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            scores = torch.softmax(logits, dim=-1)
            selection_scores = scores if self.expert_bias is None else scores + self.expert_bias.to(scores.dtype)
            _, top_indices = torch.topk(selection_scores, k=self.top_k, dim=-1)
            top_logits = torch.gather(logits, dim=-1, index=top_indices)
            top_weights = torch.softmax(top_logits, dim=-1)

        if self.route_scale != 1.0:
            top_weights = top_weights * self.route_scale

        return top_weights.to(x.dtype), top_indices, scores
