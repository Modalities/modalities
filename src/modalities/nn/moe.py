from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel


# MoE implementation inspired from Dbrx https://github.com/databricks/dbrx/blob/main/model/modeling_dbrx.py
class MoEFFNConfig(BaseModel):
    moe_num_experts: int
    moe_top_k: int
    moe_normalize_expert_weights: float
    uniform_expert_assignment: bool
    ffn_hidden_size: int
    act_fn: Callable[[], nn.Module] = nn.SiLU
    moe_jitter_eps: Optional[float]


class MoERouter(nn.Module):
    def __init__(self, hidden_size: int, moe_config: MoEFFNConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_num_experts = moe_config.moe_num_experts
        self.moe_top_k = moe_config.moe_top_k
        self.moe_normalize_expert_weights = moe_config.moe_normalize_expert_weights
        self.uniform_expert_assignment = moe_config.uniform_expert_assignment
        self.moe_jitter_eps = moe_config.moe_jitter_eps

        self.layer = nn.Linear(self.hidden_size, self.moe_num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.training and self.moe_jitter_eps is not None:
            x = x * self._jitter(x)

        weights = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1, dtype=torch.float32)
        top_weights, top_experts = torch.topk(weights, self.moe_top_k, dim=-1)

        if self.moe_normalize_expert_weights:
            top_weights = top_weights / torch.norm(
                top_weights, p=self.moe_normalize_expert_weights, dim=-1, keepdim=True
            )

        if self.uniform_expert_assignment:
            with torch.no_grad():
                uniform_tensor = (
                    torch.arange(0, top_experts.numel(), device=top_experts.device, dtype=top_experts.dtype)
                    % self.moe_num_experts
                )
                top_experts = uniform_tensor.reshape(top_experts.shape)
                # Note, weights and top_weights are not changed

        top_weights = top_weights.to(x.dtype)
        return top_weights, top_experts

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.moe_jitter_eps is None:
            raise RuntimeError("The router does not have moe_jitter_eps set.")
        low = 1.0 - self.moe_jitter_eps
        high = 1.0 + self.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)


class MoEExpertGLU(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, act_fn: Callable[[], nn.Module] = nn.GELU
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_num_experts = moe_num_experts
        self.activation_fn = act_fn()

        self.w1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.v1 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(moe_num_experts * ffn_hidden_size, hidden_size))

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        expert_w1 = self.w1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
        expert_v1 = self.v1.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]
        expert_w2 = self.w2.view(self.moe_num_experts, self.ffn_hidden_size, self.hidden_size)[expert_idx]

        x1 = x.matmul(expert_w1.t())
        x2 = x.matmul(expert_v1.t())
        x1 = self.activation_fn(x1)
        x1 = x1 * x2
        x1 = x1.matmul(expert_w2)
        return x1


class MoEExperts(nn.Module):
    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, moe_num_experts: int, act_fn: Callable[[], nn.Module] = nn.GELU
    ):
        super().__init__()
        self.moe_num_experts = moe_num_experts
        self.mlp = MoEExpertGLU(
            hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size, moe_num_experts=moe_num_experts, act_fn=act_fn
        )

    def forward(self, x: torch.Tensor, top_weights: torch.Tensor, top_experts: torch.LongTensor) -> torch.Tensor:
        bsz, q_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)
        out = torch.zeros_like(x)

        expert_mask = nn.functional.one_hot(top_experts, num_classes=self.moe_num_experts).permute(2, 1, 0)
        for expert_idx in range(0, self.moe_num_experts):
            topk_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.shape[0] == 0:
                continue

            token_list = token_idx.tolist()
            topk_list = topk_idx.tolist()

            expert_tokens = x[None, token_list].reshape(-1, hidden_size)
            expert_out = self.mlp(expert_tokens, expert_idx) * top_weights[token_list, topk_list, None]

            out.index_add_(0, token_idx, expert_out)

        out = out.reshape(bsz, q_len, hidden_size)
        return out


class MoEFFN(nn.Module):
    def __init__(self, hidden_router_size: int, config: MoEFFNConfig):
        super().__init__()
        self.config = config

        self.router = MoERouter(hidden_router_size, config)

        self.experts = MoEExperts(
            hidden_size=hidden_router_size,
            ffn_hidden_size=self.config.ffn_hidden_size,
            moe_num_experts=self.config.moe_num_experts,
            act_fn=self.config.act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_weights, top_experts = self.router(x)
        out = self.experts(x, top_weights, top_experts)
        return out
