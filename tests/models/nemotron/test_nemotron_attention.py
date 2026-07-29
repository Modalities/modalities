import pytest
import torch

from modalities.models.nemotron.nemotron_attention import NemotronAttentionImplementation, NemotronSelfAttention

# head_dim is deliberately not n_embd // n_head_q: that decoupling is the reason this module
# exists instead of reusing the GPT2 attention.
ATTENTION_KWARGS = dict(n_embd=24, n_head_q=4, n_head_kv=2, head_dim=8)


def _make_attention(**overrides) -> NemotronSelfAttention:
    torch.manual_seed(0)
    return NemotronSelfAttention(**{**ATTENTION_KWARGS, **overrides})


def test_projection_shapes_decouple_head_dim_from_model_dim():
    attn = _make_attention()
    assert attn.q_attn.weight.shape == (4 * 8, 24)
    assert attn.k_attn.weight.shape == (2 * 8, 24)
    assert attn.v_attn.weight.shape == (2 * 8, 24)
    assert attn.c_proj.weight.shape == (24, 4 * 8)
    assert attn.q_attn.bias is None
    assert attn.n_rep == 2


def test_nemotron_3_nano_attention_dimensions():
    # Model report Table 1: 32 Q heads, 2 KV heads, head dim 128, model dim 2688.
    attn = NemotronSelfAttention(n_embd=2688, n_head_q=32, n_head_kv=2, head_dim=128)
    assert attn.q_attn.out_features == 4096
    assert attn.k_attn.out_features == 256
    assert attn.v_attn.out_features == 256
    assert attn.c_proj.in_features == 4096
    assert attn.n_rep == 16


def test_forward_preserves_shape():
    attn = _make_attention()
    x = torch.randn(2, 7, 24)
    out = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_manual_and_pytorch_flash_implementations_agree():
    manual = _make_attention(attention_implementation=NemotronAttentionImplementation.MANUAL).eval()
    flash = _make_attention(attention_implementation=NemotronAttentionImplementation.PYTORCH_FLASH).eval()
    flash.load_state_dict(manual.state_dict())

    x = torch.randn(2, 12, 24)
    with torch.no_grad():
        torch.testing.assert_close(flash(x), manual(x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "implementation",
    [NemotronAttentionImplementation.MANUAL, NemotronAttentionImplementation.PYTORCH_FLASH],
)
def test_attention_is_causal(implementation):
    attn = _make_attention(attention_implementation=implementation).eval()
    x = torch.randn(1, 10, 24)
    with torch.no_grad():
        baseline = attn(x)
        perturbed = x.clone()
        perturbed[:, 6] += 5.0
        outputs = attn(perturbed)

    torch.testing.assert_close(outputs[:, :6], baseline[:, :6], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(outputs[:, 6], baseline[:, 6])


def test_grouped_query_attention_matches_explicitly_repeated_kv_heads():
    # A GQA model with n_head_kv == n_head_q must behave exactly like multi-head attention;
    # this pins the head-repetition logic against a configuration where it is a no-op.
    gqa = _make_attention(n_head_q=4, n_head_kv=4).eval()
    x = torch.randn(2, 6, 24)
    with torch.no_grad():
        out = gqa(x)
    assert gqa.n_rep == 1
    assert out.shape == x.shape


def test_repeat_kv_duplicates_each_head_contiguously():
    x = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32).reshape(2, 2, 3, 2)
    repeated = NemotronSelfAttention._repeat_kv(x, n_rep=3)
    assert repeated.shape == (2, 6, 3, 2)
    # Heads 0,1,2 are all copies of source head 0; heads 3,4,5 of source head 1.
    for head in range(3):
        torch.testing.assert_close(repeated[:, head], x[:, 0])
    for head in range(3, 6):
        torch.testing.assert_close(repeated[:, head], x[:, 1])


def test_repeat_kv_is_a_noop_for_n_rep_one():
    x = torch.randn(1, 2, 3, 4)
    assert NemotronSelfAttention._repeat_kv(x, n_rep=1) is x


def test_batch_elements_are_independent():
    attn = _make_attention().eval()
    x = torch.randn(3, 8, 24)
    with torch.no_grad():
        batched = attn(x)
        individually = torch.cat([attn(x[i : i + 1]) for i in range(3)], dim=0)
    torch.testing.assert_close(batched, individually, rtol=1e-4, atol=1e-5)


def test_attention_is_differentiable():
    attn = _make_attention()
    x = torch.randn(2, 6, 24, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_dropout_is_disabled_in_eval_mode():
    attn = _make_attention(dropout=0.9).eval()
    x = torch.randn(2, 6, 24)
    with torch.no_grad():
        torch.testing.assert_close(attn(x), attn(x))


def test_attention_rejects_indivisible_head_counts():
    with pytest.raises(ValueError, match="must be divisible by n_head_kv"):
        _make_attention(n_head_q=3, n_head_kv=2)


def test_dao_flash_raises_when_not_installed_or_matches_reference():
    try:
        import flash_attn  # noqa: F401
    except ModuleNotFoundError:
        attn = _make_attention(attention_implementation=NemotronAttentionImplementation.DAO_FLASH)
        with pytest.raises(NotImplementedError, match="flash-attn is not installed"):
            attn(torch.randn(1, 4, 24))
    else:
        if not torch.cuda.is_available():
            pytest.skip("flash-attn requires CUDA")
        # head_dim must be a multiple of 8 for the Dao kernel; use a realistic configuration.
        manual = (
            NemotronSelfAttention(
                n_embd=64,
                n_head_q=4,
                n_head_kv=2,
                head_dim=32,
                attention_implementation=NemotronAttentionImplementation.MANUAL,
            )
            .cuda()
            .bfloat16()
            .eval()
        )
        dao = (
            NemotronSelfAttention(
                n_embd=64,
                n_head_q=4,
                n_head_kv=2,
                head_dim=32,
                attention_implementation=NemotronAttentionImplementation.DAO_FLASH,
            )
            .cuda()
            .bfloat16()
            .eval()
        )
        dao.load_state_dict(manual.state_dict())
        x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(dao(x), manual(x), rtol=2e-2, atol=2e-2)
