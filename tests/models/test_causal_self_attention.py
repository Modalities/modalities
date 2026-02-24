"""
Note: test_attention_types_approximate_equality can print the output of different attention implementations.
      To do so, turn on verbose and run 'pytest tests/models/test_causal_self_attention.py -s'
"""

from copy import deepcopy

import pytest
import torch

import modalities.models.gpt2.gpt2_model as gpt2_model
from modalities.models.gpt2.gpt2_model import (
    AttentionConfig,
    CausalSelfAttention,
    LayerNorms,
    LayerNormWrapperConfig,
    PytorchRMSLayerNormConfig,
    flash_attn_varlen_func,
)

torch.manual_seed(0)  # FIXME remove or do within tests?


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


def test_inter_document_masking_manual_mask_shape_and_blocks():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    )

    mask = attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 1], [1, 2]], max_seq_len=3)

    expected_batch_0 = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )
    expected_batch_1 = torch.tensor(
        [
            [True, False, False],
            [False, True, True],
            [False, True, True],
        ]
    )

    assert mask.shape == (2, 3, 3)
    torch.testing.assert_close(mask[0].cpu(), expected_batch_0)
    torch.testing.assert_close(mask[1].cpu(), expected_batch_1)


def test_inter_document_masking_manual_forward_allows_mask():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    )

    inputs = torch.rand(1, 5, 4)
    mask = attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 3]], max_seq_len=5)

    output_masked = attention_layer(inputs, attention_masking_information=mask)
    output_doc_1 = attention_layer(inputs[:, :2, :])
    output_doc_2 = attention_layer(inputs[:, 2:, :])
    output_reference = torch.cat([output_doc_1, output_doc_2], dim=1)

    torch.testing.assert_close(output_masked, output_reference)


def test_inter_document_masking_manual_mask_symmetry_and_blocks():
    """
    Test to ensure that the inter-document masking is symmetric and correctly blocks attention
    between different documents within a batch.
    """
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    )

    in_batch_seq_lens = [[2, 1, 3], [1, 2, 1]]
    mask = attention_layer.prepare_inter_document_masking(in_batch_seq_lens=in_batch_seq_lens, max_seq_len=6)

    assert mask.shape == (2, 6, 6)
    for batch_index, doc_seq_lens in enumerate(in_batch_seq_lens):
        expected = torch.zeros((6, 6), dtype=torch.bool)
        cursor = 0
        for length in doc_seq_lens:
            expected[cursor : cursor + length, cursor : cursor + length] = True
            cursor += length
        torch.testing.assert_close(mask[batch_index].cpu(), expected)
        torch.testing.assert_close(mask[batch_index].cpu(), mask[batch_index].cpu().transpose(0, 1))


def test_inter_document_masking_manual_handles_empty_docs():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    )

    mask = attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 1], []], max_seq_len=3)

    assert mask.shape == (2, 3, 3)
    torch.testing.assert_close(mask[1].cpu(), torch.zeros((3, 3), dtype=torch.bool))


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
def test_inter_document_masking_device_and_dtype_propagation():
    attention_config = AttentionConfig(qkv_transforms=[])
    manual_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    ).cuda()

    manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 1]], max_seq_len=3)

    assert manual_mask.device == manual_layer.c_proj.weight.device
    assert manual_mask.dtype == torch.bool

    dao_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="dao_flash",
        attention_config=attention_config,
    ).cuda()

    indices, cu_seqlens, max_seqlen = dao_layer.prepare_inter_document_masking(
        in_batch_seq_lens=[[2, 1]], max_seq_len=3
    )

    assert indices.device == dao_layer.c_proj.weight.device
    assert cu_seqlens.device == dao_layer.c_proj.weight.device
    assert indices.dtype == torch.int64
    assert cu_seqlens.dtype == torch.int32
    assert max_seqlen == 2


def test_inter_document_masking_manual_float_mask_matches_bool():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="manual",
        attention_config=attention_config,
    )

    inputs = torch.rand(1, 3, 4)
    bool_mask = torch.tensor(
        [
            [
                [True, False, False],
                [True, True, False],
                [False, True, True],
            ]
        ]
    )
    float_mask = torch.where(bool_mask, torch.tensor(0.0), torch.tensor(float("-inf"))).to(inputs.dtype)

    output_bool = attention_layer(inputs, attention_masking_information=bool_mask)
    output_float = attention_layer(inputs, attention_masking_information=float_mask)

    torch.testing.assert_close(output_bool, output_float)


def test_inter_document_masking_dao_flash_empty_cases():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="dao_flash",
        attention_config=attention_config,
    )

    indices, cu_seqlens, max_seqlen = attention_layer.prepare_inter_document_masking(
        in_batch_seq_lens=[[2, 1], []], max_seq_len=3
    )
    assert indices.numel() == 3
    assert cu_seqlens.tolist() == [0, 2, 3]
    assert max_seqlen == 2


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_empty_docs_forward():
    attention_config = AttentionConfig(qkv_transforms=[])
    dao_layer = _get_random_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="dao_flash",
        attention_config=attention_config,
    )

    inputs = _get_random_input_seq((2, 3, 4))
    masking = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 1], []], max_seq_len=3)
    output = dao_layer(inputs, attention_masking_information=masking)

    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))


def test_inter_document_masking_dao_flash_validation_errors():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="dao_flash",
        attention_config=attention_config,
    )

    with pytest.raises(ValueError):
        attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[1, 1, 1, 1]], max_seq_len=3)

    with pytest.raises(ValueError):
        attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 2]], max_seq_len=3)

    with pytest.raises(ValueError):
        attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[], []], max_seq_len=3)


def test_inter_document_masking_pytorch_flash_not_supported():
    attention_config = AttentionConfig(qkv_transforms=[])
    attention_layer = _get_identity_attention_layer(
        n_head_q=2,
        n_head_kv=2,
        n_embd=4,
        attention_impl="pytorch_flash",
        attention_config=attention_config,
    )

    with pytest.raises(NotImplementedError):
        attention_layer.prepare_inter_document_masking(in_batch_seq_lens=[[1, 1]], max_seq_len=2)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
@pytest.mark.parametrize(
    "masked_attn_type, docwise_attn_type",
    [
        ("dao_flash", "manual"),
        ("dao_flash", "dao_flash"),
        ("manual", "manual"),
        ("manual", "dao_flash"),
    ],
)
def test_inter_document_masking_matches_docwise_attention(masked_attn_type, docwise_attn_type):
    torch.manual_seed(0)
    masked_layer, docwise_layer = _build_matching_attention_layers(
        masked_attn_type=masked_attn_type, docwise_attn_type=docwise_attn_type
    )

    inputs = _get_random_input_seq((1, 5, 16))
    mask = masked_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 3]], max_seq_len=5)

    output_masked = masked_layer(inputs, attention_masking_information=mask)
    output_doc_1 = docwise_layer(inputs[:, :2, :])
    output_doc_2 = docwise_layer(inputs[:, 2:, :])
    output_reference = torch.cat([output_doc_1, output_doc_2], dim=1)

    torch.testing.assert_close(output_masked, output_reference, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
@pytest.mark.parametrize(
    "masked_attn_type, docwise_attn_type",
    [
        ("dao_flash", "manual"),
        ("dao_flash", "dao_flash"),
        ("manual", "manual"),
        ("manual", "dao_flash"),
    ],
)
def test_inter_document_masking_matches_docwise_attention_gqa(masked_attn_type, docwise_attn_type):
    torch.manual_seed(0)
    masked_layer, docwise_layer = _build_matching_attention_layers(
        masked_attn_type=masked_attn_type, docwise_attn_type=docwise_attn_type, n_head_kv=2
    )

    inputs = _get_random_input_seq((1, 6, 16))
    mask = masked_layer.prepare_inter_document_masking(in_batch_seq_lens=[[1, 2, 3]], max_seq_len=6)

    output_masked = masked_layer(inputs, attention_masking_information=mask)
    output_doc_1 = docwise_layer(inputs[:, :1, :])
    output_doc_2 = docwise_layer(inputs[:, 1:3, :])
    output_doc_3 = docwise_layer(inputs[:, 3:, :])
    output_reference = torch.cat([output_doc_1, output_doc_2, output_doc_3], dim=1)

    torch.testing.assert_close(output_masked, output_reference, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_manual_matches_dao_flash_with_masks():
    torch.manual_seed(0)
    dao_layer, manual_layer = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((2, 5, 16))
    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 3], [1, 1, 2]], max_seq_len=5)
    manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 3], [1, 1, 2]], max_seq_len=5)

    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)
    output_manual = manual_layer(inputs, attention_masking_information=manual_mask)

    torch.testing.assert_close(output_dao, output_manual, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_blocks_cross_doc_leakage():
    torch.manual_seed(0)
    dao_layer, _ = _build_matching_dao_and_manual_attention()

    inputs = torch.zeros((1, 6, 16), dtype=torch.bfloat16, device="cuda")
    inputs[:, :2, :] = 1000.0

    mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 4]], max_seq_len=6)
    output_masked = dao_layer(inputs, attention_masking_information=mask)
    output_unmasked = dao_layer(inputs)

    assert not torch.allclose(output_masked[:, 2:, :], output_unmasked[:, 2:, :], atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_matches_manual_with_batchwise_splits():
    torch.manual_seed(0)
    dao_layer, manual_layer = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((2, 5, 16))
    sub_seq_lengths = [[2, 3], [1, 1, 3]]
    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=5)
    manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=5)

    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)
    output_manual = manual_layer(inputs, attention_masking_information=manual_mask)

    torch.testing.assert_close(output_dao, output_manual, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_with_padding_matches_manual_on_valid_tokens():
    torch.manual_seed(0)
    dao_layer, manual_layer = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((2, 6, 16))
    sub_seq_lengths = [[2, 2], [1, 1, 2]]
    valid_lengths = _sum_lengths_per_batch(sub_seq_lengths)

    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=6)
    manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=6)

    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)
    output_manual = manual_layer(inputs, attention_masking_information=manual_mask)

    for batch_index, valid_len in enumerate(valid_lengths):
        torch.testing.assert_close(
            output_dao[batch_index, :valid_len, :],
            output_manual[batch_index, :valid_len, :],
            atol=2.5e-3,
            rtol=0.016,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_with_padding_zeroes_dao_padded_outputs():
    torch.manual_seed(0)
    dao_layer, _ = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((2, 6, 16))
    sub_seq_lengths = [[2, 1], [1, 1, 1]]
    valid_lengths = _sum_lengths_per_batch(sub_seq_lengths)

    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=6)
    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)

    for batch_index, valid_len in enumerate(valid_lengths):
        if valid_len < inputs.size(1):
            torch.testing.assert_close(
                output_dao[batch_index, valid_len:, :],
                torch.zeros_like(output_dao[batch_index, valid_len:, :]),
            )


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_padded_gradients_are_zero():
    """
    Test to ensure that the gradients of padded tokens are zero when using DAO flash attention.
    This is tested by backpropagating through the padded outputs and checking the gradients.
    """
    torch.manual_seed(0)
    dao_layer, _ = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((2, 5, 16)).requires_grad_(True)
    sub_seq_lengths = [[2, 1], [1, 1]]
    valid_lengths = _sum_lengths_per_batch(sub_seq_lengths)

    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=5)
    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)

    loss = 0.0
    for batch_index, valid_len in enumerate(valid_lengths):
        if valid_len < inputs.size(1):
            loss = loss + output_dao[batch_index, valid_len:, :].sum()

    loss.backward()
    torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs.grad))


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_handles_single_token_docs():
    torch.manual_seed(0)
    dao_layer, manual_layer = _build_matching_dao_and_manual_attention()

    inputs = _get_random_input_seq((1, 5, 16))
    sub_seq_lengths = [[1, 1, 3]]
    dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=5)
    manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=5)

    output_dao = dao_layer(inputs, attention_masking_information=dao_mask)
    output_manual = manual_layer(inputs, attention_masking_information=manual_mask)

    torch.testing.assert_close(output_dao, output_manual, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
@pytest.mark.skipif(flash_attn_varlen_func is None, reason="This test requires flash-attn varlen support.")
def test_inter_document_masking_dao_flash_randomized_splits():
    torch.manual_seed(0)
    dao_layer, manual_layer = _build_matching_dao_and_manual_attention()
    generator = torch.Generator().manual_seed(123)

    inputs = _get_random_input_seq((1, 6, 16))
    for _ in range(3):
        sub_seq_lengths = [_generate_sub_seq_lengths(total_len=6, max_chunk=3, generator=generator)]
        dao_mask = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=6)
        manual_mask = manual_layer.prepare_inter_document_masking(in_batch_seq_lens=sub_seq_lengths, max_seq_len=6)

        output_dao = dao_layer(inputs, attention_masking_information=dao_mask)
        output_manual = manual_layer(inputs, attention_masking_information=manual_mask)

        torch.testing.assert_close(output_dao, output_manual, atol=2.5e-3, rtol=0.016)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires 1 GPU.")
def test_inter_document_masking_dao_flash_passes_expected_unpad_data(monkeypatch):
    attention_config = AttentionConfig(qkv_transforms=[])
    dao_layer = _get_random_attention_layer(
        n_head_q=4,
        n_head_kv=4,
        n_embd=16,
        attention_impl="dao_flash",
        attention_config=attention_config,
    )
    inputs = _get_random_input_seq((1, 5, 16))
    expected_masking = dao_layer.prepare_inter_document_masking(in_batch_seq_lens=[[2, 3]], max_seq_len=5)
    expected_indices, expected_cu_seqlens, expected_max_seqlen = expected_masking

    captured = {}

    def fake_flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        causal,
        softmax_scale,
        window_size,
    ):
        captured["unpad_len"] = q_unpad.shape[0]
        captured["cu_seqlens_q"] = cu_seqlens_q.detach().cpu()
        captured["cu_seqlens_k"] = cu_seqlens_k.detach().cpu()
        captured["max_seqlen_q"] = max_seqlen_q
        captured["max_seqlen_k"] = max_seqlen_k
        return torch.zeros_like(q_unpad)

    monkeypatch.setattr(gpt2_model, "flash_attn_func", object())
    monkeypatch.setattr(gpt2_model, "flash_attn_varlen_func", fake_flash_attn_varlen_func)

    output = dao_layer(inputs, attention_masking_information=expected_masking)

    assert output.shape == inputs.shape
    assert captured["unpad_len"] == expected_indices.numel()
    assert captured["cu_seqlens_q"].tolist() == expected_cu_seqlens.detach().cpu().tolist()
    assert captured["cu_seqlens_k"].tolist() == expected_cu_seqlens.detach().cpu().tolist()
    assert captured["max_seqlen_q"] == expected_max_seqlen
    assert captured["max_seqlen_k"] == expected_max_seqlen


def _build_matching_dao_and_manual_attention(n_head_kv: int = 4):
    return _build_matching_attention_layers(
        masked_attn_type="dao_flash", docwise_attn_type="manual", n_head_kv=n_head_kv
    )


def _build_matching_attention_layers(
    masked_attn_type: str, docwise_attn_type: str, n_head_kv: int = 4
) -> tuple[CausalSelfAttention, CausalSelfAttention]:
    attention_config = AttentionConfig(qkv_transforms=[])

    masked_layer = _get_random_attention_layer(
        n_head_q=4,
        n_head_kv=n_head_kv,
        n_embd=16,
        attention_impl=masked_attn_type,
        attention_config=attention_config,
    )
    docwise_layer = _get_random_attention_layer(
        n_head_q=4,
        n_head_kv=n_head_kv,
        n_embd=16,
        attention_impl=docwise_attn_type,
        attention_config=attention_config,
    )
    docwise_layer.load_state_dict(masked_layer.state_dict())
    return masked_layer, docwise_layer


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


def _get_identity_attention_layer(n_head_q, n_head_kv, n_embd, attention_impl, attention_config):
    self_attention_layer = CausalSelfAttention(
        n_head_q=n_head_q,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        bias=False,
        dropout=0.0,
        attention_config=attention_config,
        attention_impl=attention_impl,
    )
    with torch.no_grad():
        eye = torch.eye(n_embd, dtype=self_attention_layer.q_attn.weight.dtype)
        self_attention_layer.q_attn.weight.copy_(eye)
        self_attention_layer.k_attn.weight.copy_(eye)
        self_attention_layer.v_attn.weight.copy_(eye)
        self_attention_layer.c_proj.weight.copy_(eye)
    return self_attention_layer


def _generate_sub_seq_lengths(total_len: int, max_chunk: int, generator: torch.Generator) -> list[int]:
    lengths = []
    remaining = total_len
    while remaining > 0:
        next_len = int(torch.randint(1, min(max_chunk, remaining) + 1, (1,), generator=generator).item())
        lengths.append(next_len)
        remaining -= next_len
    return lengths


def _sum_lengths_per_batch(sub_seq_lengths: list[list[int]]) -> list[int]:
    return [sum(lengths) for lengths in sub_seq_lengths]
