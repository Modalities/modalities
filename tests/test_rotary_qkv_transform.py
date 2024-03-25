import torch

from modalities.models.gpt2.gpt2_model import RotaryTransform


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
