import pytest
import torch

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


def _apply_rotary(x: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor) -> torch.Tensor:
    cos_local = cos_cached[:, :, : x.shape[-2], :]
    sin_local = sin_cached[:, :, : x.shape[-2], :]
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_local) + (x_rot * sin_local)


def _assert_yarn_outputs_match_reference(
    rotary_transform: RotaryTransform,
    q: torch.Tensor,
    k: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    seq_length: int,
) -> None:
    t = torch.arange(seq_length, device=q.device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, rotary_transform.inv_freq.to(q.dtype))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * rotary_transform.attention_scaling)[None, None, :, :].to(q.dtype)
    sin = (emb.sin() * rotary_transform.attention_scaling)[None, None, :, :].to(q.dtype)

    q_expected = _apply_rotary(q, cos, sin)
    k_expected = _apply_rotary(k, cos, sin)

    assert torch.allclose(q_rot, q_expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_rot, k_expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {
            "rope_type": "yarn",
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4,
        },
        {
            "rope_type": "yarn",
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4,
        },
    ],
)
def test_rotary_transform_yarn_matches_reference(rope_scaling: dict):
    bs = 1
    n_heads = 2
    embedding_dim = 8
    seq_length = 8
    head_dim = embedding_dim // n_heads

    q = torch.randn(bs, n_heads, seq_length, head_dim)
    k = torch.randn(bs, n_heads, seq_length, head_dim)
    v = torch.randn(bs, n_heads, seq_length, head_dim)

    rotary_transform = RotaryTransform(
        n_embd=embedding_dim,
        n_head=n_heads,
        base_freq=10000,
        max_position_embeddings=seq_length,
        rope_scaling=rope_scaling,
    )

    q_rot, k_rot, v_rot = rotary_transform(q=q, k=k, v=v)
    assert torch.equal(v, v_rot)

    _assert_yarn_outputs_match_reference(
        rotary_transform=rotary_transform,
        q=q,
        k=k,
        q_rot=q_rot,
        k_rot=k_rot,
        seq_length=seq_length,
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("beta_fast", "32"),
        ("beta_slow", torch.tensor(1.0)),
        ("beta_fast", True),
    ],
)
def test_rotary_transform_yarn_rejects_invalid_beta_values(key: str, value: object):
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 4,
        key: value,
    }

    with pytest.raises(ValueError, match=rf"rope_scaling\.{key} must be a float"):
        RotaryTransform(
            n_embd=8,
            n_head=2,
            base_freq=10000,
            max_position_embeddings=8,
            rope_scaling=rope_scaling,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("beta_fast", "32"),
        ("beta_slow", torch.tensor(1.0)),
        ("beta_slow", False),
    ],
)
def test_rotary_transform_config_yarn_rejects_invalid_beta_values(key: str, value: object):
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 4,
        key: value,
    }

    with pytest.raises(ValueError, match=rf"rope_scaling\.{key} must be a float"):
        AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
            n_embd=8,
            n_head=2,
            seq_length_dim=-2,
            base_freq=10000,
            max_position_embeddings=8,
            rope_scaling=rope_scaling,
        )


@pytest.mark.parametrize(
    ("rope_scaling", "match"),
    [
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": "1.0",
                "mscale_all_dim": 1.0,
            },
            r"rope_scaling\.mscale must be a float",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": 1.0,
                "mscale_all_dim": torch.tensor(1.0),
            },
            r"rope_scaling\.mscale_all_dim must be a float",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": True,
                "mscale_all_dim": 1.0,
            },
            r"rope_scaling\.mscale must be a float",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": 1.0,
            },
            r"rope_scaling\.mscale and rope_scaling\.mscale_all_dim must be provided together",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale_all_dim": 1.0,
            },
            r"rope_scaling\.mscale and rope_scaling\.mscale_all_dim must be provided together",
        ),
    ],
)
def test_rotary_transform_yarn_rejects_invalid_mscale_values(rope_scaling: dict, match: str):
    with pytest.raises(ValueError, match=match):
        RotaryTransform(
            n_embd=8,
            n_head=2,
            base_freq=10000,
            max_position_embeddings=8,
            rope_scaling=rope_scaling,
        )


@pytest.mark.parametrize(
    ("rope_scaling", "match"),
    [
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": "1.0",
                "mscale_all_dim": 1.0,
            },
            r"rope_scaling\.mscale must be a float",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": 1.0,
                "mscale_all_dim": torch.tensor(1.0),
            },
            r"rope_scaling\.mscale_all_dim must be a float",
        ),
        (
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4,
                "mscale": 1.0,
            },
            r"rope_scaling\.mscale and rope_scaling\.mscale_all_dim must be provided together",
        ),
    ],
)
def test_rotary_transform_config_yarn_rejects_invalid_mscale_values(rope_scaling: dict, match: str):
    with pytest.raises(ValueError, match=match):
        AttentionConfig.QueryKeyValueTransformConfig.RotaryTransformConfig(
            n_embd=8,
            n_head=2,
            seq_length_dim=-2,
            base_freq=10000,
            max_position_embeddings=8,
            rope_scaling=rope_scaling,
        )
