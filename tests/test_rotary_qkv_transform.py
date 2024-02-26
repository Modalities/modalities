from modalities.models.gpt2.gpt2_model import RotaryTransform
import torch 

def test_rotary_transform():
    bs = 10
    n_heads = 4
    embedding_dim = 128
    seq_lenght = 12

    q = torch.ones(bs,n_heads,seq_lenght, embedding_dim//n_heads) + 1
    k = torch.ones(bs,n_heads,seq_lenght, embedding_dim//n_heads) + 2 
    v = torch.ones(bs,n_heads,seq_lenght, embedding_dim//n_heads)

    rotary_transform = RotaryTransform(n_embd=embedding_dim, n_head=n_heads)

    q_rot, k_rot, v_rot = rotary_transform(q=q, k=k,v=v)

    assert not torch.equal(q, q_rot)
    assert q.shape == q_rot.shape
    assert not torch.equal(k, k_rot)
    assert k.shape == k_rot.shape
    assert torch.equal(v, v_rot)
    assert v.shape == v_rot.shape
