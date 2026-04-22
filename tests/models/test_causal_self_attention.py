"""
Note: test_attention_types_approximate_equality can print the output of different attention implementations.
      To do so, turn on verbose and run 'pytest tests/models/test_causal_self_attention.py -s'
"""

import os
import subprocess
import sys
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from modalities.models.gpt2.gpt2_model import (
    AttentionConfig,
    CausalSelfAttention,
    LayerNorms,
    LayerNormWrapperConfig,
    PytorchRMSLayerNormConfig,
    is_flash_attn_v4_available,
)

torch.manual_seed(0)

FLASH_ATTN_V4_AVAILABLE = is_flash_attn_v4_available()
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _get_random_input_seq(embedding_shape):
    flash_attn_supported_dtype = torch.bfloat16
    return torch.rand(size=embedding_shape, dtype=flash_attn_supported_dtype).cuda()


def _get_random_attention_layer(n_head_q, n_head_kv, n_embd, attention_impl, attention_config):
    self_attention_layer = CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        bias=False,
        dropout=0.0,
        attention_config=attention_config,
        attention_impl=attention_impl,
    ).cuda()
    self_attention_layer.q_attn = self_attention_layer.q_attn.bfloat16()
    self_attention_layer.k_attn = self_attention_layer.k_attn.bfloat16()
    self_attention_layer.v_attn = self_attention_layer.v_attn.bfloat16()
    self_attention_layer.c_proj = self_attention_layer.c_proj.bfloat16()
    return self_attention_layer


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd",
    [
        (4, 4, 32),  # MHA (multi head attention)
        (32, 32, 768),  # MHA (multi head attention)
        (4, 2, 32),  # GQA (group query attention)
        (8, 2, 32),  # GQA
        (32, 4, 768),  # GQA
    ],
)
def test_repeat_kv_heads(n_head_q, n_head_kv, n_embd):
    batch_size = 2
    block_size = 10
    head_dim = n_embd // n_head_q
    AttentionConfig(qkv_transforms=[])

    q = torch.rand(batch_size, n_head_q, block_size - 1, head_dim, dtype=torch.bfloat16).cuda()
    k_in = torch.rand(batch_size, n_head_kv, block_size - 1, head_dim, dtype=torch.bfloat16).cuda()
    v_in = torch.rand(batch_size, n_head_kv, block_size - 1, head_dim, dtype=torch.bfloat16).cuda()

    k_out, v_out = CausalSelfAttention.repeat_kv_heads(q, k_in, v_in)

    # assert that shapes are correct: (batch_size, num_heads, seq_length, head_dim)
    assert k_out.shape == q.shape
    assert v_out.shape == q.shape

    # assert that repetitions are correct
    if n_head_q != n_head_kv:
        # e.g. n_head_q = 6, n_head_kv = 2
        for i in range(0, n_head_q, n_head_q // n_head_kv):  # e.g. i = 0,3
            for j in range(1, n_head_q // n_head_kv):  # e.g. j = 1,2
                torch.testing.assert_close(
                    k_out[:, i, :, :], k_out[:, i + j, :, :]
                )  # compares i=0 vs. i+j=1,2 | i=3 vs. i+j=4,5
                torch.testing.assert_close(
                    v_out[:, i, :, :], v_out[:, i + j, :, :]
                )  # compares i=0 vs. i+j=1,2 | i=3 vs. i+j=4,5


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_impl, successful",
    [
        # manual
        (4, 4, 32, "manual", True),  # MHA
        (8, 2, 32, "manual", True),  # GQA
        (9, 8, 32, "manual", False),
        (8, 3, 32, "manual", False),
        (6, 6, 32, "manual", False),
        # pytorch_flash
        (4, 4, 32, "pytorch_flash", True),  # MHA
        (8, 2, 32, "pytorch_flash", True),  # GQA
        (9, 8, 32, "pytorch_flash", False),
        (8, 3, 32, "pytorch_flash", False),
        (6, 6, 32, "pytorch_flash", False),
        # dao_flash
        (4, 4, 32, "dao_flash", True),  # MHA
        (8, 2, 32, "dao_flash", True),  # GQA
        (9, 8, 32, "dao_flash", False),
        (8, 3, 32, "dao_flash", False),
        (6, 6, 32, "dao_flash", False),
    ],
)
def test_forward_pass_success(n_head_q, n_head_kv, n_embd, attention_impl, successful):
    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size - 1, n_embd)
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "attention_config": attention_config,
        "attention_impl": attention_impl,
    }

    if not successful:
        with pytest.raises(Exception):
            _get_random_attention_layer(**attention_layer_args)
    else:
        attention_layer = _get_random_attention_layer(**attention_layer_args).cuda()
        embedded_input_seq = _get_random_input_seq(embedding_shape).cuda()
        output_tensor = attention_layer(embedded_input_seq)
        assert output_tensor.shape == embedding_shape


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
@pytest.mark.parametrize(
    "seq_length, n_head_q, n_head_kv, head_dim, attention_impl",
    [
        # manual
        (12, 4, 4, 32, "manual"),  # MHA
        (12, 8, 2, 32, "manual"),  # GQA
        (16, 8, 8, 16, "manual"),  # MHA
        # pytorch_flash
        (12, 4, 4, 32, "pytorch_flash"),  # MHA
        (12, 8, 2, 32, "pytorch_flash"),  # GQA
        (16, 8, 8, 16, "pytorch_flash"),  # MHA
        # dao_flash
        (12, 4, 4, 32, "dao_flash"),  # MHA
        (12, 8, 2, 32, "dao_flash"),  # GQA
        (16, 8, 8, 16, "dao_flash"),  # MHA
    ],
)
def test_forward_pass_shapes(seq_length, n_head_q, n_head_kv, head_dim, attention_impl):
    # Source: https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a
    batch_size = 2

    # shapes: (batch_size, num_heads, seq_length, head_dim)
    query_orig = torch.rand(batch_size, n_head_q, seq_length, head_dim, dtype=torch.bfloat16).cuda()
    key_orig = torch.rand(batch_size, n_head_kv, seq_length, head_dim, dtype=torch.bfloat16).cuda()
    value_orig = torch.rand(batch_size, n_head_kv, seq_length, head_dim, dtype=torch.bfloat16).cuda()

    out = CausalSelfAttention.execute_attention(
        query_orig,
        key_orig,
        value_orig,
        dropout=0.0,
        attention_impl=attention_impl,
    )

    # shape: (batch_size, seq_length, num_heads, head_dim)
    assert out.shape == (batch_size, seq_length, n_head_q, head_dim)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_impl_1, attention_impl_2, verbose",
    [
        # manual vs. pytorch_flash
        (4, 4, 4, "manual", "pytorch_flash", False),  # MHA
        (4, 4, 32, "manual", "pytorch_flash", False),
        (4, 4, 768, "manual", "pytorch_flash", False),
        (8, 8, 2048, "manual", "pytorch_flash", False),
        (8, 2, 2048, "manual", "pytorch_flash", False),  # GQA
        # manual vs. dao_flash
        (4, 4, 4, "manual", "dao_flash", False),  # MQA
        (4, 4, 32, "manual", "dao_flash", False),
        (4, 4, 768, "manual", "dao_flash", False),
        (8, 8, 2048, "manual", "dao_flash", False),
        (8, 2, 2048, "manual", "dao_flash", False),  # GQA
        # pytorch_flash vs. dao_flash
        (4, 4, 4, "pytorch_flash", "dao_flash", False),
        (4, 4, 32, "pytorch_flash", "dao_flash", False),
        (4, 4, 768, "pytorch_flash", "dao_flash", False),
        (8, 8, 2048, "pytorch_flash", "dao_flash", False),
        (8, 2, 2048, "pytorch_flash", "dao_flash", False),  # GQA
    ],
)
def test_attention_implementation_approximate_equality(
    n_head_q, n_head_kv, n_embd, attention_impl_1, attention_impl_2, verbose
):
    # flash attention is non-deterministic,
    # see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # and https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    # as well as https://github.com/pytorch/pytorch/issues/119188#issuecomment-2043157422

    batch_size = 2
    block_size = 10
    embedding_shape = (batch_size, block_size - 1, n_embd)
    embedded_input_seq = _get_random_input_seq(embedding_shape)

    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer_args = {
        "n_head_q": n_head_q,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "attention_config": attention_config,
        "attention_impl": attention_impl_1,
    }

    attention_layer = {}
    attention_layer[attention_impl_1] = _get_random_attention_layer(**attention_layer_args)
    attention_layer[attention_impl_2] = deepcopy(attention_layer[attention_impl_1])
    attention_layer[attention_impl_2].attention_impl = attention_impl_2

    output_tensor = {}
    output_tensor[attention_impl_1] = attention_layer[attention_impl_1](embedded_input_seq)
    output_tensor[attention_impl_2] = attention_layer[attention_impl_2](embedded_input_seq)
    if verbose:
        print(f"{attention_impl_1} vs. {attention_impl_2}: \n{output_tensor}")
    torch.testing.assert_close(
        output_tensor[attention_impl_1],
        output_tensor[attention_impl_2],
        atol=2.5e-3,  # default for bfloat16: 1e-5
        rtol=0.016,  # default for bfloat16: 0.016
    )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.parametrize(
    "n_head_q, n_head_kv, n_embd, attention_impl",
    [
        (4, 4, 32, "manual"),
        (8, 2, 32, "manual"),
        (4, 4, 32, "pytorch_flash"),
        (8, 2, 32, "pytorch_flash"),
        (4, 4, 32, "dao_flash"),
        (8, 2, 32, "dao_flash"),
    ],
)
def test_qk_norm(n_head_q, n_head_kv, n_embd, attention_impl):
    batch_size = 2
    block_size = 10
    head_dim = n_embd // n_head_q
    embedding_shape = (batch_size, block_size - 1, n_embd)
    embedded_input_seq = _get_random_input_seq(embedding_shape)

    attention_config_no_norm = AttentionConfig(qkv_transforms=[], use_qk_norm=False)
    attention_config_with_norm = AttentionConfig(
        qkv_transforms=[],
        use_qk_norm=True,
        qk_norm_config=LayerNormWrapperConfig(
            norm_type=LayerNorms.pytorch_rms_norm, config=PytorchRMSLayerNormConfig(normalized_shape=head_dim)
        ),
    )

    # Create two separate layers with same initial weights
    torch.manual_seed(0)
    layer_no_norm = _get_random_attention_layer(n_head_q, n_head_kv, n_embd, attention_impl, attention_config_no_norm)

    torch.manual_seed(0)
    layer_with_norm = _get_random_attention_layer(
        n_head_q, n_head_kv, n_embd, attention_impl, attention_config_with_norm
    )

    output_no_norm = layer_no_norm(embedded_input_seq)
    output_with_norm = layer_with_norm(embedded_input_seq)

    assert output_no_norm.shape == output_with_norm.shape == embedding_shape
    assert not torch.allclose(output_no_norm, output_with_norm, atol=1e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(not FLASH_ATTN_V4_AVAILABLE, reason="FA4 not installed")
def test_dao_flash_v4_forward_mha_subprocess():
    result = _run_fa4_subprocess(
        """
        import torch
        from modalities.models.gpt2.gpt2_model import CausalSelfAttention

        q = torch.rand(2, 4, 12, 32, dtype=torch.bfloat16, device='cuda')
        k = torch.rand(2, 4, 12, 32, dtype=torch.bfloat16, device='cuda')
        v = torch.rand(2, 4, 12, 32, dtype=torch.bfloat16, device='cuda')
        out = CausalSelfAttention.execute_attention(q, k, v, dropout=0.0, attention_impl='dao_flash_v4')
        torch.cuda.synchronize()
        assert tuple(out.shape) == (2, 12, 4, 32)
        print('ok')
        """
    )
    assert result.stdout.strip().endswith("ok")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(not FLASH_ATTN_V4_AVAILABLE, reason="FA4 not installed")
def test_dao_flash_v4_forward_gqa_subprocess():
    result = _run_fa4_subprocess(
        """
        import torch
        from modalities.models.gpt2.gpt2_model import CausalSelfAttention

        q = torch.rand(2, 8, 12, 32, dtype=torch.bfloat16, device='cuda')
        k = torch.rand(2, 2, 12, 32, dtype=torch.bfloat16, device='cuda')
        v = torch.rand(2, 2, 12, 32, dtype=torch.bfloat16, device='cuda')
        out = CausalSelfAttention.execute_attention(q, k, v, dropout=0.0, attention_impl='dao_flash_v4')
        torch.cuda.synchronize()
        assert tuple(out.shape) == (2, 12, 8, 32)
        print('ok')
        """
    )
    assert result.stdout.strip().endswith("ok")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(not FLASH_ATTN_V4_AVAILABLE, reason="FA4 not installed")
def test_dao_flash_v4_qk_norm_subprocess():
    result = _run_fa4_subprocess(
        """
        import torch
        from modalities.models.gpt2.gpt2_model import (
            AttentionConfig,
            CausalSelfAttention,
            LayerNorms,
            LayerNormWrapperConfig,
            PytorchRMSLayerNormConfig,
        )

        torch.manual_seed(0)
        attention_config_no_norm = AttentionConfig(qkv_transforms=[])
        attention_config_with_norm = AttentionConfig(
            qkv_transforms=[],
            qk_norm_config=LayerNormWrapperConfig(
                norm_type=LayerNorms.pytorch_rms_norm,
                config=PytorchRMSLayerNormConfig(normalized_shape=8),
            ),
        )

        torch.manual_seed(0)
        layer_no_norm = CausalSelfAttention(
            4, 4, 32, attention_config_no_norm, 'dao_flash_v4', False, 0.0
        ).cuda().bfloat16()
        torch.manual_seed(0)
        layer_with_norm = CausalSelfAttention(
            4, 4, 32, attention_config_with_norm, 'dao_flash_v4', False, 0.0
        ).cuda().bfloat16()
        x = torch.rand((2, 9, 32), dtype=torch.bfloat16, device='cuda')
        out_no_norm = layer_no_norm(x)
        out_with_norm = layer_with_norm(x)
        torch.cuda.synchronize()
        assert out_no_norm.shape == out_with_norm.shape == (2, 9, 32)
        assert not torch.allclose(out_no_norm, out_with_norm, atol=1e-6)
        print('ok')
        """
    )
    assert result.stdout.strip().endswith("ok")


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(not FLASH_ATTN_V4_AVAILABLE, reason="FA4 not installed")
def test_dao_flash_v4_backward_approximate_equality_subprocess():
    result = _run_fa4_subprocess(
        """
        import torch
        from modalities.models.gpt2.gpt2_model import CausalSelfAttention

        query_ref = torch.rand((2, 8, 12, 64), dtype=torch.bfloat16, device='cuda', requires_grad=True)
        key_ref = torch.rand((2, 2, 12, 64), dtype=torch.bfloat16, device='cuda', requires_grad=True)
        value_ref = torch.rand((2, 2, 12, 64), dtype=torch.bfloat16, device='cuda', requires_grad=True)

        query_fa4 = query_ref.detach().clone().requires_grad_(True)
        key_fa4 = key_ref.detach().clone().requires_grad_(True)
        value_fa4 = value_ref.detach().clone().requires_grad_(True)

        output_ref = CausalSelfAttention.execute_attention(
            query_ref, key_ref, value_ref, dropout=0.0, attention_impl='pytorch_flash'
        )
        output_fa4 = CausalSelfAttention.execute_attention(
            query_fa4, key_fa4, value_fa4, dropout=0.0, attention_impl='dao_flash_v4'
        )
        torch.testing.assert_close(output_ref, output_fa4, atol=2.5e-3, rtol=0.016)

        output_ref.float().sum().backward()
        output_fa4.float().sum().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(query_ref.grad, query_fa4.grad, atol=5e-3, rtol=0.02)
        torch.testing.assert_close(key_ref.grad, key_fa4.grad, atol=5e-3, rtol=0.02)
        torch.testing.assert_close(value_ref.grad, value_fa4.grad, atol=5e-3, rtol=0.02)
        print('ok')
        """
    )
    assert result.stdout.strip().endswith("ok")


def _run_fa4_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run flash attention 4 related code in a subprocess to isolate FA4's CUDA context
    and avoid conflicts with other tests.
    The code should print 'ok' if it runs successfully.
    The function returns the CompletedProcess object,
    which contains stdout and stderr for further inspection if needed.
    TODO: This might be an A100 / SM80-specific issue, so we can consider removing this subprocess isolation
          if we confirm that FA4 works well on newer architectures without it.
    """
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(SRC_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
