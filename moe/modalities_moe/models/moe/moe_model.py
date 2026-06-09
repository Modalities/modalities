import math
from dataclasses import dataclass
from typing import Literal, Optional, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

# TODO reolve this import
try:
    from torch.distributed.tensor import DTensor
except Exception:
    DTensor = None


class MoEModelConfig(BaseModel):
    # model config
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
    moe_every_n_layers: int = 1
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.0
    moe_router_noise_std: float = 0.0


@dataclass
class MoEArguments:
    # Model hyperparameters
    d_model: int
    d_ff: int

    # MoE hyperparameters
    num_experts: int
    top_k: int
    capacity_factor: float = 1.25
    min_capacity: int = 4
    overflow_policy: Literal["drop", "residual"] = "residual"

    # Router configuration
    router_noise_std: float = 0.0
    router_temperature: float = 1.0
    router_dropout: float = 0.0

    # Auxiliary loss coefficients
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 0.0

    # Training configuration
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(Expert, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = torch.nn.functional.silu(x1) * x2
        x = self.w3(x)
        return self.dropout(x)


class GroupedExperts(nn.Module):
    """Grouped experts for torchtitan compatibility."""

    def __init__(self, config: MoEArguments):
        super().__init__()
        self.num_experts = config.num_experts
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.b1 = nn.Parameter(torch.empty(self.num_experts, self.d_ff))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.d_ff, self.d_model))
        self.b2 = nn.Parameter(torch.empty(self.num_experts, self.d_ff))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.d_model, self.d_ff))
        self.b3 = nn.Parameter(torch.empty(self.num_experts, self.d_model))

        self.initialize()

    def initialize(self):
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        bound_w1 = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.b1, -bound_w1, bound_w1)

        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        bound_w2 = 1 / math.sqrt(self.d_model)
        nn.init.uniform_(self.b2, -bound_w2, bound_w2)

        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        bound_w3 = 1 / math.sqrt(self.d_ff)
        nn.init.uniform_(self.b3, -bound_w3, bound_w3)

    def _forward_local(self, routed_input, num_tokens_per_expert) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        start = 0

        # ExpertParallel may convert parameters to DTensor. Local expert compute
        # expects plain tensors, so we materialize local shards when needed.
        w1 = self.w1.to_local() if DTensor is not None and isinstance(self.w1, DTensor) else self.w1
        b1 = self.b1.to_local() if DTensor is not None and isinstance(self.b1, DTensor) else self.b1
        w2 = self.w2.to_local() if DTensor is not None and isinstance(self.w2, DTensor) else self.w2
        b2 = self.b2.to_local() if DTensor is not None and isinstance(self.b2, DTensor) else self.b2
        w3 = self.w3.to_local() if DTensor is not None and isinstance(self.w3, DTensor) else self.w3
        b3 = self.b3.to_local() if DTensor is not None and isinstance(self.b3, DTensor) else self.b3

        local_num_tokens = (
            num_tokens_per_expert.to_local()
            if DTensor is not None and isinstance(num_tokens_per_expert, DTensor)
            else num_tokens_per_expert
        )

        total_rows = routed_input.shape[0]
        for expert_idx, num_tokens in enumerate(local_num_tokens.tolist()):
            requested_tokens = int(num_tokens)
            end = start + requested_tokens

            # EP alignment can request padded tokens; only a subset may exist in routed_input.
            local_end = min(end, total_rows)
            expert_input = routed_input[start:local_end]
            real_tokens = int(expert_input.shape[0])

            out_parts: list[torch.Tensor] = []
            if real_tokens > 0:
                x1 = torch.nn.functional.linear(expert_input, w1[expert_idx], b1[expert_idx])
                x2 = torch.nn.functional.linear(expert_input, w2[expert_idx], b2[expert_idx])
                hidden = torch.nn.functional.silu(x1) * x2
                out_real = torch.nn.functional.linear(hidden, w3[expert_idx], b3[expert_idx])
                out_parts.append(self.dropout(out_real))

            pad_tokens = requested_tokens - real_tokens
            if pad_tokens > 0:
                out_parts.append(routed_input.new_zeros((pad_tokens, self.d_model)))

            if len(out_parts) > 0:
                outputs.append(torch.cat(out_parts, dim=0) if len(out_parts) > 1 else out_parts[0])

            start = end

        if len(outputs) == 0:
            return routed_input.new_zeros((0, self.d_model))

        out = torch.cat(outputs, dim=0)

        # EP permute may append extra global padding slots beyond per-expert aligned sizes.
        # output_fn(_unpermute) expects the same row count as routed_input.
        if out.shape[0] < total_rows:
            out = torch.cat(
                [out, routed_input.new_zeros((total_rows - out.shape[0], self.d_model))],
                dim=0,
            )
        elif out.shape[0] > total_rows:
            out = out[:total_rows]

        return out

    def forward(self, routed_input, num_tokens_per_expert) -> torch.Tensor:
        # routed_input: (M, D), sorted/grouped by expert id
        # num_tokens_per_expert: (E_local,) for local compute, or global counts before EP input_fn
        return self._forward_local(routed_input, num_tokens_per_expert)


class MoEBlock(nn.Module):
    def __init__(self, config: MoEArguments):
        super(MoEBlock, self).__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.router = nn.Linear(config.d_model, self.num_experts)
        self.router_dropout = nn.Dropout(config.router_dropout) if config.router_dropout > 0 else nn.Identity()
        self.experts = GroupedExperts(config)

        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x):
        B, T, D = x.size()
        E = self.config.num_experts
        K = self.config.top_k
        N = B * T

        x_flat = x.view(N, D)

        # Router logits
        logits = self.router(self.router_dropout(x_flat))  # (N, E)
        if self.config.router_noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self.config.router_noise_std
            logits = logits + noise
        logits = logits / self.config.router_temperature
        probs = torch.softmax(logits, dim=-1)  # (N, E)

        # top-k
        topk_val, topk_idx = torch.topk(probs, k=K, dim=-1)  # (N, K)
        topk_w = topk_val / (topk_val.sum(dim=-1, keepdim=True) + 1e-9)  # (N, K)

        # capacity per expert
        capacity = math.ceil(self.config.capacity_factor * N / E)
        capacity = max(capacity, self.config.min_capacity)

        # dispatch mask - preserve dtype of input
        dispatch_mask = torch.nn.functional.one_hot(topk_idx, num_classes=E).to(x_flat.dtype)  # (N, K, E)

        # token assignment
        expert_mask = dispatch_mask.sum(dim=1)  # (N, E)
        positions = torch.cumsum(expert_mask, dim=0)  # (N, E)
        capacity_mask = (positions <= capacity).to(x_flat.dtype)  # (N, E)
        final_mask = dispatch_mask * capacity_mask.unsqueeze(1)  # (N, K, E)
        combine_weights = final_mask * topk_w.unsqueeze(-1)  # (N, K, E)

        combine_weights.sum(dim=1)  # (N, E)

        # count actual assignments per expert
        load = final_mask.sum(dim=[0, 1])  # (E,)
        importance = probs.sum(dim=0)  # (E,)

        # Build routed token stream.
        valid_mask = capacity_mask.gather(1, topk_idx).bool()  # (N, K)
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K)

        flat_valid = valid_mask.reshape(-1)
        flat_token_ids = token_ids.reshape(-1)[flat_valid]
        flat_expert_ids = topk_idx.reshape(-1)[flat_valid]
        flat_weights = topk_w.reshape(-1)[flat_valid]

        if flat_expert_ids.numel() > 0:
            sort_idx = torch.argsort(flat_expert_ids)
            token_ids_sorted = flat_token_ids[sort_idx]
            expert_ids_sorted = flat_expert_ids[sort_idx]
            weights_sorted = flat_weights[sort_idx]

            routed_input = x_flat[token_ids_sorted]
            num_tokens_per_expert = torch.bincount(expert_ids_sorted, minlength=E)

            routed_output = self.experts(routed_input, num_tokens_per_expert)
            weighted_output = routed_output * weights_sorted.unsqueeze(-1)

            out = x_flat.new_zeros((N, D))
            out.index_add_(0, token_ids_sorted, weighted_output)

            assigned = x_flat.new_zeros((N,))
            assigned.index_add_(0, token_ids_sorted, weights_sorted)
        else:
            out = x_flat.new_zeros((N, D))
            assigned = x_flat.new_zeros((N,))

        # Overflow handling: tokens not assigned to any expert
        not_assigned = assigned < 1e-6

        if not_assigned.any():
            if self.config.overflow_policy == "residual":
                out[not_assigned] = x_flat[not_assigned]
            # if 'drop', out is already zero for those positions

        # auxiliary loss
        aux = None
        if self.config.aux_loss_coef > 0:
            imp = importance / (importance.sum() + 1e-9)
            ld = load / (load.sum() + 1e-9)
            lb = E * torch.sum(imp * ld)
            aux = self.config.aux_loss_coef * lb

        if self.config.z_loss_coef > 0:
            z = torch.logsumexp(logits, dim=-1)
            z_loss = torch.mean(z**2)
            aux = (aux if aux is not None else 0.0) + self.config.z_loss_coef * z_loss

        self.last_aux_loss = aux
        return out.view(B, T, D)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super(GroupedQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.q_proj(query).view(query.size(0), -1, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(key.size(0), -1, self.n_kv_heads, self.head_dim)
        V = self.v_proj(value).view(value.size(0), -1, self.n_kv_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        # Compute attention scores
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", Q, K) / (self.head_dim**0.5)
        if mask is not None:
            attn_scores += mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, V)
        attn_output = attn_output.contiguous().view(query.size(0), -1, self.n_heads * self.head_dim)
        return self.out_proj(attn_output), None


class TransformerBlock(nn.Module):
    """Transformer block with MoE"""

    def __init__(self, d_model, d_ff, num_heads, num_kv_heads, moe_config: MoEArguments):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = num_heads
        self.n_kv_heads = num_kv_heads
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.pre_attn_norm = RMSNorm(d_model)
        self.pre_ffn_norm = RMSNorm(d_model)

        if moe_config is not None:
            self.ffn = MoEBlock(moe_config)
            self.is_moe = True
        else:
            self.ffn = Expert(d_model, d_ff)
            self.is_moe = False

    def forward(self, x):
        x_norm = self.pre_attn_norm(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Pre-MoE norm
        x_norm = self.pre_ffn_norm(x)
        moe_output = self.ffn(x_norm)
        x = x + moe_output

        return x

    @property  # TODO: AUX LOSS IN FORWARD
    def aux_loss(self):
        if self.is_moe and hasattr(self.ffn, "last_aux_loss"):
            return self.ffn.last_aux_loss
        return None


class MoEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        num_layers: int,
        sample_key: str = "input_ids",
        prediction_key: str = "logits",
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        tie_embeddings: bool = True,
        moe_every_n_layers: int = 1,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_capacity_factor: float = 1.25,
        moe_aux_loss_coef: float = 0.01,
        moe_z_loss_coef: float = 0.0,
        moe_router_noise_std: float = 0.0,
    ):
        super(MoEModel, self).__init__()
        self.sample_key = sample_key
        self.prediction_key = prediction_key
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.tie_embeddings = tie_embeddings
        self.moe_every_n_layers = moe_every_n_layers
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_aux_loss_coef = moe_aux_loss_coef
        self.moe_z_loss_coef = moe_z_loss_coef
        self.moe_router_noise_std = moe_router_noise_std

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)

        moe_config = MoEArguments(
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            capacity_factor=self.moe_capacity_factor,
            aux_loss_coef=self.moe_aux_loss_coef,
            z_loss_coef=self.moe_z_loss_coef,
            router_noise_std=self.moe_router_noise_std,
            dropout=self.ffn_dropout,
        )

        self.layers = nn.ModuleDict()
        for i in range(self.num_layers):
            if i % self.moe_every_n_layers == 0:
                block = TransformerBlock(self.d_model, self.d_ff, self.n_heads, self.n_kv_heads, moe_config)
            else:
                block = TransformerBlock(self.d_model, self.d_ff, self.n_heads, self.n_kv_heads, None)  # No MoE
            self.layers[str(i)] = block
        self.final_norm = RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if self.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    @property
    def weight_decay_groups(self):
        return {
            "linear": ["attention", "router", "w1", "w2", "w3", "b1", "b2", "b3", "lm_head"],
            "embedding": ["token_emb", "pos_emb"],
            "layernorm": ["pre_attn_norm", "pre_ffn_norm", "final_norm"],
        }

    @overload
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the MoE module.

        Args:
            inputs (dict[str, torch.Tensor]): A dictionary containing input tensors.
                - sample_key (str): Key for the input tensor containing token ids.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing output tensors.
                - prediction_key (str): Key for the output tensor containing logits.
        """
        ...

    @overload
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            inputs (torch.Tensor): A tensor containing input token ids.

        Returns:
            torch.Tensor: A tensor containing output logits.
        """
        ...

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Forward pass of the module.

        Args:
            inputs (dict[str, torch.Tensor] | torch.Tensor): Input data.

        Returns:
            dict[str, torch.Tensor] | torch.Tensor: Model output.
        """
        if isinstance(inputs, dict):
            return {self.prediction_key: self.forward_impl(inputs[self.sample_key])}
        else:
            return self.forward_impl(inputs)

    def forward_impl(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds model's max_seq_len {self.max_seq_len}"
        device = input_ids.device

        # Token and position embeddings
        token_embeddings = self.token_emb(input_ids)  # (B, T, D)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_embeddings = self.pos_emb(positions)  # (B, T, D)
        x = token_embeddings + pos_embeddings  # (B, T, D)

        # Transformer blocks
        for i, layer in enumerate(self.layers.values()):
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits


if __name__ == "__main__":  # sanity test
    torch.manual_seed(0)

    model = MoEModel(
        vocab_size=32064,
        max_seq_len=32768,
        d_model=4096,
        n_heads=32,
        n_kv_heads=8,
        num_layers=32,
        d_ff=14336,
        moe_every_n_layers=1,
        moe_num_experts=8,
        moe_top_k=2,
    )

    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    x = torch.randint(0, model.vocab_size, (2, 64))
    logits = model(x)

    print("logits:", logits.shape)
    loss = logits.mean()
    loss.backward()
    print("backward OK")
