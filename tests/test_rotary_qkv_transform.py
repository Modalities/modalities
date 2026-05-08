import math

import torch

from modalities.models.gpt2.gpt2_model import AttentionConfig, RotaryTransform


def _compute_yarn_parameters_reference(
    *,
    base_freq: float,
    dim: int,
    max_position_embeddings: int,
    rope_scaling: dict,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    factor = float(rope_scaling["factor"])
    attention_factor = rope_scaling.get("attention_factor")
    mscale = rope_scaling.get("mscale")
    mscale_all_dim = rope_scaling.get("mscale_all_dim")
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings") or max_position_embeddings
    beta_fast = rope_scaling.get("beta_fast") or 32
    beta_slow = rope_scaling.get("beta_slow") or 1
    truncate = rope_scaling.get("truncate", True)

    def get_mscale(scale: float, mscale_value: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_value * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, float(mscale)) / get_mscale(factor, float(mscale_all_dim)))
        else:
            attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations: float, dim_value: int, base_value: float, max_pos_emb: int) -> float:
        return (dim_value * math.log(max_pos_emb / (num_rotations * 2 * math.pi))) / (2 * math.log(base_value))

    def find_correction_range(
        low_rot: float,
        high_rot: float,
        dim_value: int,
        base_value: float,
        max_pos_emb: int,
        truncate_bounds: bool,
    ) -> tuple[float, float]:
        low = find_correction_dim(low_rot, dim_value, base_value, max_pos_emb)
        high = find_correction_dim(high_rot, dim_value, base_value, max_pos_emb)
        if truncate_bounds:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim_value - 1)

    def linear_ramp_factor(min_value: float, max_value: float, dim_value: int) -> torch.Tensor:
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim_value, dtype=torch.float32, device=device) - min_value) / (
            max_value - min_value
        )
        return torch.clamp(linear_func, 0, 1)

    pos_freqs = float(base_freq) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range(
        float(beta_fast),
        float(beta_slow),
        dim,
        float(base_freq),
        int(original_max_position_embeddings),
        bool(truncate),
    )
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, float(attention_factor)


def test_rotary_transform():
    bs = 1
    n_heads = 2
    embedding_dim = 8
    seq_lenght = 2
    head_dim = embedding_dim // n_heads

    q = torch.ones(bs, n_heads, seq_lenght, head_dim) + 1
    q[:, :, :, head_dim // 2 :] = q[:, :, :, head_dim // 2 :] + 1
    k = torch.ones(bs, n_heads, seq_lenght, head_dim) + 2
    k[:, :, :, head_dim // 2 :] = k[:, :, :, head_dim // 2 :] + 1
    v = torch.ones(bs, n_heads, seq_lenght, head_dim)

    rotary_transform = RotaryTransform(n_embd=embedding_dim, n_head=n_heads)

    q_rot, k_rot, v_rot = rotary_transform(q=q, k=k, v=v)

    assert torch.equal(v, v_rot)
    assert v.shape == v_rot.shape

    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    m = torch.tensor([0, 1]).view(2, 1)
    theta_0 = theta[0]
    theta_1 = theta[1]
    theta = torch.tensor([theta_0, theta_1, theta_0, theta_1]).view(1, 4)
    m_theta = m * theta

    cos_m_theta = m_theta.cos()
    sin_m_theta = m_theta.sin()

    for comp, comp_rot in zip([q, k], [q_rot, k_rot]):
        assert not torch.equal(comp, comp_rot)
        assert comp.shape == comp_rot.shape
        comp_h_1, comp_h_2 = comp.chunk(2, dim=-1)
        comp_rot_h = torch.cat([-comp_h_2, comp_h_1], dim=-1)
        comp_rot_expected = comp * cos_m_theta + comp_rot_h * sin_m_theta
        assert torch.equal(comp_rot_expected, comp_rot)


def test_rotary_transform_yarn_matches_hf_rope_utils():
    bs = 2
    n_heads = 4
    embedding_dim = 16
    seq_len = 8
    head_dim = embedding_dim // n_heads

    q = torch.randn(bs, n_heads, seq_len, head_dim)
    k = torch.randn(bs, n_heads, seq_len, head_dim)
    v = torch.randn(bs, n_heads, seq_len, head_dim)

    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 4,
    }

    rotary_transform = RotaryTransform(
        n_embd=embedding_dim,
        n_head=n_heads,
        base_freq=10000,
        max_position_embeddings=seq_len,
        rope_scaling=rope_scaling,
    )

    q_rot, k_rot, v_rot = rotary_transform(q=q, k=k, v=v)

    inv_freq, attention_scaling = _compute_yarn_parameters_reference(
        base_freq=10000.0,
        dim=head_dim,
        max_position_embeddings=seq_len,
        rope_scaling=rope_scaling,
        device=q.device,
    )
    t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq.to(q.dtype))
    emb = torch.cat((freqs, freqs), dim=-1).to(q.device)
    cos = (emb.cos() * attention_scaling)[None, None, :, :].to(q.dtype)
    sin = (emb.sin() * attention_scaling)[None, None, :, :].to(q.dtype)

    q_expected = (q * cos) + (rotary_transform.rotate_half(q) * sin)
    k_expected = (k * cos) + (rotary_transform.rotate_half(k) * sin)

    assert torch.allclose(q_rot, q_expected)
    assert torch.allclose(k_rot, k_expected)
    assert torch.equal(v, v_rot)


def test_rotary_config_accepts_type_alias_for_rope_scaling():
    config = AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
        n_embd=128,
        n_head=8,
        seq_length_dim=-2,
        base_freq=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 128,
        },
    )

    assert config.rope_scaling is not None
    assert config.rope_scaling["rope_type"] == "yarn"
