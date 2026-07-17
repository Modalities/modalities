import math
from typing import Optional

import torch


def compute_default_inv_freq(dim_model: int, base_freq: float, device: Optional[torch.device] = None) -> torch.Tensor:
    return 1.0 / (base_freq ** (torch.arange(0, dim_model, 2, device=device).float() / dim_model))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seq_length_dim: int) -> torch.Tensor:
    cos = cos[:, :, : x.shape[seq_length_dim], :]
    sin = sin[:, :, : x.shape[seq_length_dim], :]
    return (x * cos) + (rotate_half(x) * sin)


def update_cos_sin_tables(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
    seq_length_dim: int,
    seq_len_cached: Optional[int],
    cos_cached: Optional[torch.Tensor],
    sin_cached: Optional[torch.Tensor],
) -> tuple[int, torch.Tensor, torch.Tensor]:
    seq_len = x.shape[seq_length_dim]

    if (
        seq_len != seq_len_cached
        or cos_cached is None
        or sin_cached is None
        or cos_cached.device != x.device
        or cos_cached.dtype != x.dtype
    ):
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq.to(x.dtype))
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos_cached = (emb.cos() * attention_scaling)[None, None, :, :].to(x.dtype)
        sin_cached = (emb.sin() * attention_scaling)[None, None, :, :].to(x.dtype)
        seq_len_cached = seq_len

    return seq_len_cached, cos_cached, sin_cached


def compute_yarn_inv_freq_and_attention_scaling(
    dim_model: int,
    base_freq: float,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    factor: Optional[float],
    attention_factor: Optional[float],
    mscale: Optional[float],
    mscale_all_dim: Optional[float],
    beta_fast: float,
    beta_slow: float,
    truncate: bool,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    factor_float = (
        float(factor) if factor is not None else float(max_position_embeddings / original_max_position_embeddings)
    )

    def get_mscale(scale: float, mscale_value: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_value * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale is not None and mscale_all_dim is not None:
            attention_factor = float(
                get_mscale(factor_float, float(mscale)) / get_mscale(factor_float, float(mscale_all_dim))
            )
        else:
            attention_factor = get_mscale(factor_float)

    def find_correction_dim(num_rotations: float, dim: int, base: float, max_pos_emb: int) -> float:
        return (dim * math.log(max_pos_emb / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        max_pos_emb: int,
        do_truncate: bool,
    ) -> tuple[float, float]:
        low = find_correction_dim(low_rot, dim, base, max_pos_emb)
        high = find_correction_dim(high_rot, dim, base, max_pos_emb)
        if do_truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value: float, max_value: float, dim: int) -> torch.Tensor:
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_value) / (max_value - min_value)
        return torch.clamp(linear_func, 0, 1)

    pos_freqs = base_freq ** (torch.arange(0, dim_model, 2, device=device, dtype=torch.float) / dim_model)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor_float * pos_freqs)

    low, high = find_correction_range(
        beta_fast,
        beta_slow,
        dim_model,
        base_freq,
        original_max_position_embeddings,
        bool(truncate),
    )

    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim_model // 2).to(
        device=device, dtype=torch.float
    )
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )

    return inv_freq, float(attention_factor)
