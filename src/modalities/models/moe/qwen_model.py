import math
from typing import Literal, Optional, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from modalities.models.model import NNModel

try:
    from torch.distributed.tensor import DTensor
except Exception:
    DTensor = None


class QwenModelConfig(BaseModel):
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    num_layers: int
    d_ff: int
    sample_key: str = "input_ids"
    prediction_key: str = "logits"
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    tie_embeddings: bool = False
    norm_eps: float = 1e-6
    rope_base: float = 1000000.0

    moe_num_experts: int = 128
    moe_top_k: int = 8
    moe_d_ff: int = 768
    moe_capacity_factor: float = 1.25
    moe_min_capacity: int = 4
    moe_overflow_policy: Literal["drop", "residual"] = "residual"
    moe_router_noise_std: float = 0.0
    moe_router_temperature: float = 1.0
    moe_router_dropout: float = 0.0
    moe_aux_loss_coef: float = 0.001
    moe_z_loss_coef: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _compute_cache(self, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        t = torch.arange(self.max_seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x: torch.Tensor, seq_len: int):
        if self.cos_cached is None:
            self._compute_cache(x.device)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len, rope_base, norm_eps, attn_dropout):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, base=rope_base)
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rope(q, seq_len=T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.n_rep > 1:
            k = (
                k.unsqueeze(2)
                .expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
                .reshape(B, self.n_heads, T, self.head_dim)
            )
            v = (
                v.unsqueeze(2)
                .expand(B, self.n_kv_heads, self.n_rep, T, self.head_dim)
                .reshape(B, self.n_heads, T, self.head_dim)
            )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class GroupedExperts(nn.Module):
    def __init__(
        self,
        num_experts,
        d_model,
        d_ff,
        ffn_dropout,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity()

        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.d_ff))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))

    def _forward_local(self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.to_local() if DTensor is not None and isinstance(self.w1, DTensor) else self.w1
        w2 = self.w2.to_local() if DTensor is not None and isinstance(self.w2, DTensor) else self.w2
        w3 = self.w3.to_local() if DTensor is not None and isinstance(self.w3, DTensor) else self.w3
        # F.linear requires matching dtypes between inputs and weights. Under mixed precision,
        # routed_input can be BF16 while local expert weights remain FP32.
        if routed_input.dtype != w1.dtype:
            w1 = w1.to(dtype=routed_input.dtype)
            w2 = w2.to(dtype=routed_input.dtype)
            w3 = w3.to(dtype=routed_input.dtype)
        local_num_tokens = (
            num_tokens_per_expert.to_local()
            if DTensor is not None and isinstance(num_tokens_per_expert, DTensor)
            else num_tokens_per_expert
        )

        outputs: list[torch.Tensor] = []
        start = 0
        total_rows = routed_input.shape[0]

        for expert_idx, num_tokens in enumerate(local_num_tokens.tolist()):
            requested_tokens = int(num_tokens)
            end = start + requested_tokens
            local_end = min(end, total_rows)
            expert_input = routed_input[start:local_end]
            real_tokens = int(expert_input.shape[0])

            out_parts: list[torch.Tensor] = []
            if real_tokens > 0:
                x1 = F.linear(expert_input, w1[expert_idx])
                x2 = F.linear(expert_input, w2[expert_idx])
                out_parts.append(self.dropout(F.linear(F.silu(x1) * x2, w3[expert_idx])))

            pad = requested_tokens - real_tokens
            if pad > 0:
                out_parts.append(routed_input.new_zeros((pad, self.d_model)))

            if out_parts:
                outputs.append(torch.cat(out_parts, dim=0) if len(out_parts) > 1 else out_parts[0])

            start = end

        if not outputs:
            return routed_input.new_zeros((0, self.d_model))

        out = torch.cat(outputs, dim=0)
        if out.shape[0] < total_rows:
            out = torch.cat([out, routed_input.new_zeros((total_rows - out.shape[0], self.d_model))], dim=0)
        elif out.shape[0] > total_rows:
            out = out[:total_rows]
        return out

    def forward(self, routed_input: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        return self._forward_local(routed_input, num_tokens_per_expert)


class MoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        moe_d_ff: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_capacity_factor: float,
        moe_min_capacity: int,
        moe_overflow_policy: str,
        moe_router_noise_std: float,
        moe_router_temperature: float,
        moe_router_dropout: float,
        moe_aux_loss_coef: float,
        moe_z_loss_coef: float,
        ffn_dropout: float,
    ):
        super().__init__()
        self.num_experts = moe_num_experts
        self.top_k = moe_top_k
        self.capacity_factor = moe_capacity_factor
        self.min_capacity = moe_min_capacity
        self.overflow_policy = moe_overflow_policy
        self.router_noise_std = moe_router_noise_std
        self.router_dropout = nn.Dropout(moe_router_dropout) if moe_router_dropout > 0 else nn.Identity()
        self.router_temperature = moe_router_temperature
        self.aux_loss_coef = moe_aux_loss_coef
        self.z_loss_coef = moe_z_loss_coef

        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.experts = GroupedExperts(
            num_experts=moe_num_experts, d_model=d_model, d_ff=moe_d_ff, ffn_dropout=ffn_dropout
        )
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        E = self.num_experts
        K = self.top_k
        N = B * T
        x_flat = x.view(N, D)

        logits = self.router(self.router_dropout(x_flat).to(self.router.weight.dtype)).float()
        if self.router_noise_std > 0 and self.training:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        logits = logits / self.router_temperature

        probs = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(probs, k=K, dim=-1)
        topk_w = (topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)).to(x_flat.dtype)

        capacity = max(math.ceil(self.capacity_factor * N / E), self.min_capacity)

        dispatch_mask = F.one_hot(topk_idx, num_classes=E).to(x_flat.dtype)
        positions = torch.cumsum(dispatch_mask.sum(dim=1), dim=0)
        capacity_mask = (positions <= capacity).to(x_flat.dtype)
        final_mask = dispatch_mask * capacity_mask.unsqueeze(1)

        load = final_mask.sum(dim=[0, 1])
        importance = probs.sum(dim=0)

        flat_valid = capacity_mask.gather(1, topk_idx).bool().reshape(-1)
        flat_token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)[flat_valid]
        flat_expert_ids = topk_idx.reshape(-1)[flat_valid]
        flat_weights = topk_w.reshape(-1)[flat_valid]

        if flat_expert_ids.numel() > 0:
            sort_idx = torch.argsort(flat_expert_ids)
            token_ids_sorted = flat_token_ids[sort_idx]
            expert_ids_sorted = flat_expert_ids[sort_idx]
            weights_sorted = flat_weights[sort_idx]

            routed_output = self.experts(x_flat[token_ids_sorted], torch.bincount(expert_ids_sorted, minlength=E))
            weighted_output = routed_output * weights_sorted.unsqueeze(-1)

            out = x_flat.new_zeros((N, D))
            out.index_add_(0, token_ids_sorted, weighted_output)
            assigned = x_flat.new_zeros((N,))
            assigned.index_add_(0, token_ids_sorted, weights_sorted)
        else:
            out = x_flat.new_zeros((N, D))
            assigned = x_flat.new_zeros((N,))

        not_assigned = assigned < 1e-6
        if not_assigned.any() and self.overflow_policy == "residual":
            out[not_assigned] = x_flat[not_assigned]

        aux = None
        if self.aux_loss_coef > 0:
            imp = importance / (importance.sum() + 1e-9)
            ld = load / (load.sum() + 1e-9)
            aux = self.aux_loss_coef * E * torch.sum(imp * ld)
        if self.z_loss_coef > 0:
            z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
            aux = (aux if aux is not None else torch.tensor(0.0, device=x.device)) + self.z_loss_coef * z_loss

        self.last_aux_loss = aux
        return out.view(B, T, D)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        rope_base: float,
        norm_eps: float,
        attn_dropout: float,
        ffn_dropout: float,
        moe_d_ff: int = 768,
        moe_num_experts: int = 128,
        moe_top_k: int = 8,
        moe_capacity_factor: float = 1.25,
        moe_min_capacity: int = 4,
        moe_overflow_policy: str = "residual",
        moe_router_noise_std: float = 0.0,
        moe_router_temperature: float = 1.0,
        moe_router_dropout: float = 0.0,
        moe_aux_loss_coef: float = 0.001,
        moe_z_loss_coef: float = 0.0,
    ):
        super().__init__()
        self.pre_attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            norm_eps=norm_eps,
            attn_dropout=attn_dropout,
        )
        self.pre_ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = MoEBlock(
            d_model=d_model,
            moe_d_ff=moe_d_ff,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_capacity_factor=moe_capacity_factor,
            moe_min_capacity=moe_min_capacity,
            moe_overflow_policy=moe_overflow_policy,
            moe_router_noise_std=moe_router_noise_std,
            moe_router_temperature=moe_router_temperature,
            moe_router_dropout=moe_router_dropout,
            moe_aux_loss_coef=moe_aux_loss_coef,
            moe_z_loss_coef=moe_z_loss_coef,
            ffn_dropout=ffn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.pre_attn_norm(x))
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return getattr(self.ffn, "last_aux_loss", None)


class QwenModel(NNModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        num_layers: int,
        moe_d_ff: int = 768,
        sample_key: str = "input_ids",
        prediction_key: str = "logits",
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        tie_embeddings: bool = False,
        norm_eps: float = 1e-6,
        rope_base: float = 1000000.0,
        moe_num_experts: int = 128,
        moe_top_k: int = 8,
        moe_capacity_factor: float = 1.25,
        moe_min_capacity: int = 4,
        moe_overflow_policy: str = "residual",
        moe_router_noise_std: float = 0.0,
        moe_router_temperature: float = 1.0,
        moe_router_dropout: float = 0.0,
        moe_aux_loss_coef: float = 0.001,
        moe_z_loss_coef: float = 0.0,
    ):
        weight_decay_groups = {
            "linear": ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head", "router", "w1", "w2", "w3"],
            "embedding": ["token_emb"],
            "layernorm": ["pre_attn_norm", "pre_ffn_norm", "final_norm", "q_norm", "k_norm"],
        }
        super().__init__(weight_decay_groups=weight_decay_groups)
        self.sample_key = sample_key
        self.prediction_key = prediction_key

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleDict(
            {
                str(i): TransformerBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=max_seq_len,
                    rope_base=rope_base,
                    norm_eps=norm_eps,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    moe_d_ff=moe_d_ff,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_capacity_factor=moe_capacity_factor,
                    moe_min_capacity=moe_min_capacity,
                    moe_overflow_policy=moe_overflow_policy,
                    moe_router_noise_std=moe_router_noise_std,
                    moe_router_temperature=moe_router_temperature,
                    moe_router_dropout=moe_router_dropout,
                    moe_aux_loss_coef=moe_aux_loss_coef,
                    moe_z_loss_coef=moe_z_loss_coef,
                )
                for i in range(num_layers)
            }
        )

        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        if isinstance(inputs, dict):
            return {self.prediction_key: self.forward_impl(inputs[self.sample_key])}
        return self.forward_impl(inputs)

    def forward_impl(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for layer in self.layers.values():
            x = layer(x)
        return self.lm_head(self.final_norm(x))
