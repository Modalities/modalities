import math
from typing import Literal, Optional, overload

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from torch.optim.optimizer import register_optimizer_step_post_hook

from modalities.models.components.rotary_embedding import (
    apply_rotary_pos_emb,
    compute_default_inv_freq,
    update_cos_sin_tables,
)
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
    # None means dropless routing (default). If set, each expert processes at most
    # max(ceil(moe_capacity_factor * num_tokens * moe_top_k / moe_num_experts), moe_min_capacity) tokens.
    moe_capacity_factor: Optional[float] = None
    moe_min_capacity: int = 4
    moe_overflow_policy: Literal["drop", "residual"] = "residual"
    moe_expert_backend: Literal["auto", "grouped_mm", "loop"] = "auto"
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
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._seq_len_cached: Optional[int] = None

    def _compute_cache(self, device):
        self.inv_freq = compute_default_inv_freq(dim_model=self.head_dim, base_freq=self.base, device=device)
        self._seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor, seq_len: int):
        if self.inv_freq is None:
            self._compute_cache(x.device)
        self._seq_len_cached, self.cos_cached, self.sin_cached = update_cos_sin_tables(
            x=x,
            inv_freq=self.inv_freq,
            attention_scaling=1.0,
            seq_length_dim=-2,
            seq_len_cached=self._seq_len_cached,
            cos_cached=self.cos_cached,
            sin_cached=self.sin_cached,
        )
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def apply_rotary_emb(q, k, cos, sin):
    return (
        apply_rotary_pos_emb(x=q, cos=cos, sin=sin, seq_length_dim=-2),
        apply_rotary_pos_emb(x=k, cos=cos, sin=sin, seq_length_dim=-2),
    )


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


def _local_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if DTensor is not None and isinstance(t, DTensor) else t


class GroupedExperts(nn.Module):
    # torch._grouped_mm needs every token group to start at a 16-byte aligned offset along the
    # token dimension (including in the backward 2d x 2d grouped GEMM). 16 rows covers bf16 and fp8.
    GROUPED_MM_ALIGNMENT = 16

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        ffn_dropout: float,
        backend: Literal["auto", "grouped_mm", "loop"] = "auto",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_ff = d_ff
        self.backend = backend
        self.init_std = init_std
        self.dropout = nn.Dropout(ffn_dropout) if ffn_dropout > 0 else nn.Identity()

        # w1: gate projection, w2: up projection, w3: down projection
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.d_ff))

    def reset_parameters(self):
        # NOTE: kaiming init on a 3D tensor would use fan_in = d_ff * d_model instead of the per-expert fan_in.
        for w in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(w, mean=0.0, std=self.init_std, a=-3 * self.init_std, b=3 * self.init_std)

    def _use_grouped_mm(self, x: torch.Tensor) -> bool:
        if self.backend == "grouped_mm":
            return True
        if self.backend == "loop":
            return False
        # auto: grouped_mm on CUDA. On sm80 (A100) torch 2.10 uses a fallback path that syncs on the group
        # offsets, but it is still 2-4x faster than the per-expert loop (scripts/benchmark_moe_experts.py).
        return x.is_cuda and hasattr(torch, "_grouped_mm")

    def forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        num_tokens_per_expert_cpu: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Runs the (local) experts on expert-major sorted tokens.

        Args:
            routed_input (torch.Tensor): Tokens sorted by expert, shape (num_rows, d_model). Rows beyond
                sum(num_tokens_per_expert) are tokens dropped by a capacity limit; their output is zero.
            num_tokens_per_expert (torch.Tensor): Number of tokens per local expert (on device).
            num_tokens_per_expert_cpu (Optional[list[int]]): The same counts on the host, if already available.
                Providing them avoids a host synchronization in the loop backend.
        """
        w1, w2, w3 = _local_tensor(self.w1), _local_tensor(self.w2), _local_tensor(self.w3)
        if routed_input.dtype != w1.dtype:
            # without FSDP mixed precision, weights may be fp32 while activations are bf16
            w1, w2, w3 = (w.to(dtype=routed_input.dtype) for w in (w1, w2, w3))
        num_tokens_per_expert = _local_tensor(num_tokens_per_expert)

        use_grouped_mm = self._use_grouped_mm(routed_input)
        if num_tokens_per_expert_cpu is None and not use_grouped_mm:
            num_tokens_per_expert_cpu = num_tokens_per_expert.tolist()  # host sync (non-EP loop path only)

        num_rows = routed_input.shape[0]
        num_valid_rows = sum(num_tokens_per_expert_cpu) if num_tokens_per_expert_cpu is not None else num_rows
        x = routed_input[:num_valid_rows] if num_valid_rows < num_rows else routed_input

        if use_grouped_mm:
            out = self._run_grouped_mm(x, num_tokens_per_expert, w1, w2, w3)
        else:
            out = self._run_loop(x, num_tokens_per_expert_cpu, w1, w2, w3)

        if num_valid_rows < num_rows:
            out = torch.cat([out, out.new_zeros(num_rows - num_valid_rows, self.d_model)])
        return out

    def _run_loop(
        self,
        x: torch.Tensor,
        num_tokens_per_expert_cpu: list[int],
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for expert_idx, expert_input in enumerate(torch.split(x, num_tokens_per_expert_cpu, dim=0)):
            if expert_input.shape[0] == 0:
                continue
            h = F.silu(F.linear(expert_input, w1[expert_idx])) * F.linear(expert_input, w2[expert_idx])
            outputs.append(self.dropout(F.linear(h, w3[expert_idx])))
        if len(outputs) == 0:
            # Keep the expert weights in the autograd graph so that every rank produces gradients
            # (required for collectives on gradients, e.g. FSDP reduce-scatter).
            h = F.silu(F.linear(x, w1[0])) * F.linear(x, w2[0])
            return F.linear(h, w3[0])
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _run_grouped_mm(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> torch.Tensor:
        # Pad every expert group to a multiple of GROUPED_MM_ALIGNMENT rows. The padded length is bounded
        # statically, so this needs no host synchronization. Padding rows are zeros and never read back.
        align = self.GROUPED_MM_ALIGNMENT
        num_rows = x.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0]
        with torch.no_grad():
            counts = num_tokens_per_expert.to(torch.int64)
            padded_counts = (counts + align - 1) // align * align
            padded_start = torch.cumsum(padded_counts, dim=0) - padded_counts
            start = torch.cumsum(counts, dim=0) - counts
            dest = torch.repeat_interleave(padded_start - start, counts, output_size=num_rows) + torch.arange(
                num_rows, device=x.device
            )
            offs = torch.cumsum(padded_counts, dim=0).to(torch.int32)
        padded_len = num_rows + num_local_experts * (align - 1)
        x_padded = torch.index_copy(x.new_zeros(padded_len, x.shape[-1]), 0, dest, x)

        h = F.silu(torch._grouped_mm(x_padded, w1.transpose(-2, -1), offs=offs)) * torch._grouped_mm(
            x_padded, w2.transpose(-2, -1), offs=offs
        )
        out = self.dropout(torch._grouped_mm(h, w3.transpose(-2, -1), offs=offs))
        return out.index_select(0, dest)


class MoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        moe_d_ff: int,
        moe_num_experts: int,
        moe_top_k: int,
        moe_capacity_factor: Optional[float],
        moe_min_capacity: int,
        moe_overflow_policy: str,
        moe_router_noise_std: float,
        moe_router_temperature: float,
        moe_router_dropout: float,
        moe_aux_loss_coef: float,
        moe_z_loss_coef: float,
        ffn_dropout: float,
        moe_expert_backend: Literal["auto", "grouped_mm", "loop"] = "auto",
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
            num_experts=moe_num_experts,
            d_model=d_model,
            d_ff=moe_d_ff,
            ffn_dropout=ffn_dropout,
            backend=moe_expert_backend,
        )
        self.last_tokens_per_expert: Optional[torch.Tensor] = None
        self.last_router_prob_mean: Optional[torch.Tensor] = None
        self.last_z_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        E = self.num_experts
        K = self.top_k
        N = B * T
        x_flat = x.reshape(N, D)

        # Router in fp32 for numerically stable routing decisions.
        logits = F.linear(self.router_dropout(x_flat).float(), self.router.weight.float())
        if self.router_noise_std > 0 and self.training:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        logits = logits / self.router_temperature

        scores = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(scores, k=K, dim=-1)
        topk_w = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)

        # Everything below is free of host synchronization (no boolean masking, no bincount, no .item()).
        with torch.no_grad():
            flat_expert_ids = topk_idx.reshape(-1)
            num_tokens_per_expert = torch.zeros(E, dtype=torch.int64, device=x.device).scatter_add_(
                0, flat_expert_ids, torch.ones_like(flat_expert_ids)
            )
            # stable sort keeps token order within each expert (earlier tokens win under a capacity limit)
            sort_idx = torch.argsort(flat_expert_ids, stable=True)
            num_tokens_for_experts = num_tokens_per_expert
            token_kept = None
            if self.capacity_factor is not None:
                sort_idx, num_tokens_for_experts, token_kept = self._apply_capacity(
                    sort_idx, flat_expert_ids, num_tokens_per_expert, N
                )
            token_ids_sorted = sort_idx // K

        routed_input = x_flat.index_select(0, token_ids_sorted)
        num_tokens_cpu = None
        if self.capacity_factor is not None and not getattr(self.experts, "_ep_enabled", False):
            num_tokens_cpu = num_tokens_for_experts.tolist()  # host sync; EP dispatch gets these counts for free
        if num_tokens_cpu is None:
            routed_output = self.experts(routed_input, num_tokens_for_experts)
        else:
            routed_output = self.experts(routed_input, num_tokens_for_experts, num_tokens_cpu)

        # Un-sort to (token, k) order and combine the top-k expert outputs with the routing weights.
        routed_output = torch.index_copy(torch.empty_like(routed_output), 0, sort_idx, routed_output)
        out = torch.bmm(topk_w.to(routed_output.dtype).unsqueeze(1), routed_output.view(N, K, D)).squeeze(1)

        if token_kept is not None and self.overflow_policy == "residual":
            # tokens that were dropped by all of their top-k experts pass through unchanged
            all_dropped = token_kept.view(N, K).sum(dim=-1) == 0
            out = torch.where(all_dropped.unsqueeze(-1), x_flat, out)

        # Router statistics of this micro-batch for the global-batch load balancing loss, which is computed
        # in the loss function (see GlobalBatchLoadBalancingLoss): local token counts per expert (f_i is
        # aggregated across data parallel ranks and micro-batches there) and the mean routing probability P_i.
        self.last_tokens_per_expert = num_tokens_per_expert
        self.last_router_prob_mean = scores.mean(dim=0)
        self.last_z_loss = (
            self.z_loss_coef * torch.mean(torch.logsumexp(logits, dim=-1) ** 2) if self.z_loss_coef > 0 else None
        )
        return out.view(B, T, D)

    def _apply_capacity(
        self,
        sort_idx: torch.Tensor,
        flat_expert_ids: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drops assignments beyond the expert capacity without host synchronization.

        Returns the new sort index (kept assignments grouped by expert first, dropped ones at the end),
        the number of kept tokens per expert, and a per-assignment (token-major) kept indicator.
        """
        E = self.num_experts
        capacity = max(math.ceil(self.capacity_factor * num_tokens * self.top_k / E), self.min_capacity)
        expert_ids_sorted = flat_expert_ids[sort_idx]
        expert_start = torch.cumsum(num_tokens_per_expert, dim=0) - num_tokens_per_expert
        position_in_expert = torch.arange(sort_idx.shape[0], device=sort_idx.device) - expert_start[expert_ids_sorted]
        kept_sorted = position_in_expert < capacity
        reorder = torch.argsort(expert_ids_sorted + E * (~kept_sorted).to(expert_ids_sorted.dtype), stable=True)
        token_kept = torch.empty_like(kept_sorted, dtype=torch.int32).scatter_(0, sort_idx, kept_sorted.to(torch.int32))
        return sort_idx[reorder], num_tokens_per_expert.clamp(max=capacity), token_kept


class GlobalBatchLoadBalancingLoss:
    """Global-batch load balancing loss (Qiu et al., 2025, https://arxiv.org/abs/2501.11873), as used by Qwen3,
    plus the optional router z-loss.

    For every MoE layer: L_lb = coef * E * sum_i f_i * P_i, where
      - P_i is the mean routing probability of expert i over the local micro-batch (carries the gradient),
      - f_i is the fraction of routed tokens assigned to expert i over the global batch: the token counts
        are summed over all data parallel ranks and over the micro-batches of the current optimizer step
        (gradient accumulation). Within a step, f_i is the running estimate over the micro-batches seen so far.
    As data parallel gradients are averaged, the optimized objective is E * sum_i f_i^global * P_i^global,
    i.e. load balance is encouraged over the global batch instead of within every micro-batch.

    The accumulated counts are reset after every optimizer step (global optimizer step post-hook).
    Without gradients (evaluation), the local micro-batch statistics are used without communication.
    """

    def __init__(self, model: nn.Module, process_groups: Optional[list[dist.ProcessGroup]] = None):
        """
        Args:
            model (nn.Module): The model containing the MoE blocks.
            process_groups (Optional[list[dist.ProcessGroup]]): Groups over which the token counts are summed,
                i.e. the data parallel groups. If None, the default (world) group is used when distributed
                training is initialized.
        """
        self.model = model
        self.process_groups = process_groups
        self._accumulated_tokens_per_expert: Optional[torch.Tensor] = None
        self._reset_hook = register_optimizer_step_post_hook(lambda optimizer, args, kwargs: self.reset())

    def reset(self) -> None:
        self._accumulated_tokens_per_expert = None

    def _all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        for group in self.process_groups if self.process_groups is not None else [None]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    def __call__(self) -> Optional[torch.Tensor]:
        blocks = [m for m in self.model.modules() if isinstance(m, MoEBlock) and m.last_tokens_per_expert is not None]
        if len(blocks) == 0:
            return None
        tokens_per_expert = torch.stack([b.last_tokens_per_expert for b in blocks])  # (num_layers, num_experts)
        if torch.is_grad_enabled():
            # one collective per micro-batch for all layers
            tokens_per_expert = self._all_reduce(tokens_per_expert.clone())
            if self._accumulated_tokens_per_expert is not None:
                tokens_per_expert = tokens_per_expert + self._accumulated_tokens_per_expert
            self._accumulated_tokens_per_expert = tokens_per_expert
        tokens_fraction = tokens_per_expert.float() / tokens_per_expert.sum(dim=-1, keepdim=True).clamp_min(1)

        loss = torch.zeros((), device=tokens_fraction.device)
        for block, f in zip(blocks, tokens_fraction):
            if block.aux_loss_coef > 0:
                loss = loss + block.aux_loss_coef * block.num_experts * torch.sum(f * block.last_router_prob_mean)
            if block.last_z_loss is not None:
                loss = loss + block.last_z_loss
        return loss


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
        moe_capacity_factor: Optional[float] = None,
        moe_min_capacity: int = 4,
        moe_overflow_policy: str = "residual",
        moe_expert_backend: str = "auto",
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
            moe_expert_backend=moe_expert_backend,
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
        moe_capacity_factor: Optional[float] = None,
        moe_min_capacity: int = 4,
        moe_overflow_policy: str = "residual",
        moe_expert_backend: str = "auto",
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
                    moe_expert_backend=moe_expert_backend,
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
