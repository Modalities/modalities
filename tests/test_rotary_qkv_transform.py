from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from modalities.models.gpt2.gpt2_model import AttentionConfig, RotaryTransform


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


def test_rotary_transform_yarn_matches_hf_reference():
    torch.manual_seed(42)
    bs = 2
    n_heads = 2
    embedding_dim = 8
    seq_length = 4
    head_dim = embedding_dim // n_heads

    q = torch.randn(bs, n_heads, seq_length, head_dim)
    k = torch.randn(bs, n_heads, seq_length, head_dim)
    v = torch.randn(bs, n_heads, seq_length, head_dim)

    rope_scaling = {
        "rope_type": "yarn",
        "factor": 8.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 128,
    }

    rotary_transform = RotaryTransform(
        n_embd=embedding_dim,
        n_head=n_heads,
        base_freq=10000,
        rope_scaling=rope_scaling,
        max_position_embeddings=1024,
    )

    q_rot, k_rot, v_rot = rotary_transform(q=q, k=k, v=v)

    hf_config = SimpleNamespace(
        rope_theta=10000.0,
        rope_scaling=rope_scaling,
        head_dim=head_dim,
        hidden_size=head_dim,
        num_attention_heads=1,
        max_position_embeddings=1024,
        partial_rotary_factor=1.0,
    )
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["yarn"](hf_config, device=q.device)

    positions = torch.arange(seq_length, dtype=torch.float32, device=q.device)
    freqs = torch.einsum("i,j->ij", positions, inv_freq.to(q.dtype))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * attention_scaling)[None, None, :, :].to(q.dtype)
    sin = (emb.sin() * attention_scaling)[None, None, :, :].to(q.dtype)

    q_expected = (q * cos) + (rotary_transform.rotate_half(q) * sin)
    k_expected = (k * cos) + (rotary_transform.rotate_half(k) * sin)

    assert torch.allclose(q_rot, q_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(k_rot, k_expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(v, v_rot)


def test_rotary_transform_config_accepts_type_alias_for_yarn():
    config = AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
        n_embd=8,
        n_head=2,
        seq_length_dim=-2,
        base_freq=10000,
        rope_scaling={"type": "yarn", "factor": 2.0},
    )
    assert config.rope_scaling["rope_type"] == "yarn"


def test_rotary_transform_config_rejects_missing_yarn_factor():
    with pytest.raises(ValueError, match="rope_scaling.factor"):
        AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
            n_embd=8,
            n_head=2,
            seq_length_dim=-2,
            base_freq=10000,
            rope_scaling={"rope_type": "yarn"},
        )
