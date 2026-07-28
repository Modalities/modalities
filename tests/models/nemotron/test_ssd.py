"""Correctness tests for the pure-PyTorch Mamba-2 SSD primitives.

The chunked scan is the numerical foundation of every Mamba layer, so it is validated against a
literal step-by-step transcription of the recurrence (:func:`ssd_recurrent_reference`) rather than
against itself.
"""

import pytest
import torch

from modalities.models.components.mamba2.ssd import (
    GatedRMSNorm,
    causal_depthwise_conv1d,
    ssd_chunked_scan,
    ssd_recurrent_reference,
)


def _make_ssd_inputs(
    batch_size: int = 2,
    seq_len: int = 24,
    num_heads: int = 4,
    head_dim: int = 8,
    num_groups: int = 2,
    state_dim: int = 6,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Builds a small, deterministic set of SSD inputs in float32."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    return {
        "x": randn(batch_size, seq_len, num_heads, head_dim),
        # dt must be strictly positive; softplus of a standard normal is a realistic range.
        "dt": torch.nn.functional.softplus(randn(batch_size, seq_len, num_heads)),
        # A is strictly negative, i.e. -exp(A_log) with A_log = log(U(1, 16)).
        "A": -torch.empty(num_heads).uniform_(1.0, 16.0, generator=generator),
        "B": randn(batch_size, seq_len, num_groups, state_dim),
        "C": randn(batch_size, seq_len, num_groups, state_dim),
        "D": torch.ones(num_heads),
    }


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8, 24])
def test_chunked_scan_matches_recurrent_reference(chunk_size):
    inputs = _make_ssd_inputs()
    expected = ssd_recurrent_reference(**inputs)
    actual = ssd_chunked_scan(**inputs, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk_size", [4, 8, 16])
def test_chunked_scan_is_invariant_to_chunk_size(chunk_size):
    # The block decomposition is an implementation detail: the mathematical result must not
    # depend on how the sequence is split.
    inputs = _make_ssd_inputs(seq_len=32)
    reference = ssd_chunked_scan(**inputs, chunk_size=32)
    actual = ssd_chunked_scan(**inputs, chunk_size=chunk_size)
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("seq_len", [1, 5, 7, 13])
def test_chunked_scan_handles_sequences_not_divisible_by_chunk_size(seq_len):
    inputs = _make_ssd_inputs(seq_len=seq_len)
    expected = ssd_recurrent_reference(**inputs)
    actual = ssd_chunked_scan(**inputs, chunk_size=4)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_chunked_scan_is_causal():
    # Perturbing token t must leave every output at position < t untouched.
    inputs = _make_ssd_inputs(seq_len=16)
    baseline = ssd_chunked_scan(**inputs, chunk_size=4)

    perturbed = {key: value.clone() for key, value in inputs.items()}
    perturb_at = 9
    perturbed["x"][:, perturb_at] += 5.0
    outputs = ssd_chunked_scan(**perturbed, chunk_size=4)

    torch.testing.assert_close(outputs[:, :perturb_at], baseline[:, :perturb_at], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(outputs[:, perturb_at], baseline[:, perturb_at])


def test_chunked_scan_zero_input_gives_zero_output_without_skip():
    inputs = _make_ssd_inputs()
    inputs["x"] = torch.zeros_like(inputs["x"])
    inputs["D"] = None
    outputs = ssd_chunked_scan(**inputs, chunk_size=4)
    torch.testing.assert_close(outputs, torch.zeros_like(outputs))


def test_chunked_scan_skip_connection_is_additive():
    inputs = _make_ssd_inputs()
    without_skip = ssd_chunked_scan(**{**inputs, "D": None}, chunk_size=4)
    scale = 0.5
    with_skip = ssd_chunked_scan(**{**inputs, "D": torch.full_like(inputs["D"], scale)}, chunk_size=4)
    expected = without_skip + scale * inputs["x"]
    torch.testing.assert_close(with_skip, expected, rtol=1e-4, atol=1e-5)


def test_chunked_scan_respects_initial_state():
    inputs = _make_ssd_inputs(seq_len=8)
    batch_size, _, num_heads, head_dim = inputs["x"].shape
    state_dim = inputs["B"].shape[-1]
    zero_state = torch.zeros(batch_size, num_heads, head_dim, state_dim)

    with_zero_state = ssd_chunked_scan(**inputs, chunk_size=4, initial_states=zero_state)
    without_state = ssd_chunked_scan(**inputs, chunk_size=4)
    torch.testing.assert_close(with_zero_state, without_state, rtol=1e-5, atol=1e-6)

    nonzero_state = torch.ones_like(zero_state)
    with_nonzero_state = ssd_chunked_scan(**inputs, chunk_size=4, initial_states=nonzero_state)
    assert not torch.allclose(with_nonzero_state, without_state)


def test_chunked_scan_final_state_matches_split_sequence_run():
    # Running the full sequence must equal running the first half, then the second half seeded
    # with the returned state. This pins down the inter-chunk state passing.
    inputs = _make_ssd_inputs(seq_len=16)
    full = ssd_chunked_scan(**inputs, chunk_size=4)

    def slice_inputs(start: int, stop: int) -> dict[str, torch.Tensor]:
        sliced = {key: value[:, start:stop] for key, value in inputs.items() if key in ("x", "dt", "B", "C")}
        sliced["A"] = inputs["A"]
        sliced["D"] = inputs["D"]
        return sliced

    first_half, state = ssd_chunked_scan(**slice_inputs(0, 8), chunk_size=4, return_final_state=True)
    second_half = ssd_chunked_scan(**slice_inputs(8, 16), chunk_size=4, initial_states=state)

    torch.testing.assert_close(torch.cat([first_half, second_half], dim=1), full, rtol=1e-4, atol=1e-4)


def test_chunked_scan_shares_bc_across_heads_of_a_group():
    # With one group, all heads must see the same B/C, which is what makes the group expansion
    # equivalent to explicitly broadcasting B/C.
    inputs = _make_ssd_inputs(num_heads=4, num_groups=1)
    grouped = ssd_chunked_scan(**inputs, chunk_size=4)

    expanded = dict(inputs)
    expanded["B"] = inputs["B"].expand(-1, -1, 4, -1).contiguous()
    expanded["C"] = inputs["C"].expand(-1, -1, 4, -1).contiguous()
    # With B/C already per-head, the expansion inside the scan is a no-op.
    per_head = ssd_chunked_scan(**expanded, chunk_size=4)
    torch.testing.assert_close(grouped, per_head, rtol=1e-5, atol=1e-6)


def test_chunked_scan_rejects_invalid_arguments():
    inputs = _make_ssd_inputs()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ssd_chunked_scan(**inputs, chunk_size=0)
    with pytest.raises(ValueError, match=r"x must have shape"):
        ssd_chunked_scan(**{**inputs, "x": inputs["x"][:, :, 0]}, chunk_size=4)
    with pytest.raises(ValueError, match="must be divisible by num_groups"):
        ssd_chunked_scan(**{**inputs, "x": inputs["x"][:, :, :3]}, chunk_size=4)


def test_chunked_scan_preserves_input_dtype():
    inputs = _make_ssd_inputs()
    bf16_inputs = {key: (value.bfloat16() if key != "A" else value) for key, value in inputs.items()}
    outputs = ssd_chunked_scan(**bf16_inputs, chunk_size=4)
    assert outputs.dtype == torch.bfloat16


def test_chunked_scan_is_differentiable():
    inputs = _make_ssd_inputs(seq_len=8)
    inputs["x"].requires_grad_(True)
    outputs = ssd_chunked_scan(**inputs, chunk_size=4)
    outputs.sum().backward()
    assert inputs["x"].grad is not None
    assert torch.isfinite(inputs["x"].grad).all()


def test_causal_depthwise_conv1d_is_causal_and_matches_manual_convolution():
    torch.manual_seed(0)
    batch_size, seq_len, channels, kernel_size = 2, 7, 5, 4
    x = torch.randn(batch_size, seq_len, channels)
    weight = torch.randn(channels, 1, kernel_size)
    bias = torch.randn(channels)

    actual = causal_depthwise_conv1d(x, weight=weight, bias=bias)
    assert actual.shape == x.shape

    # Manual reference: y[t, c] = bias[c] + sum_k w[c, 0, k] * x[t - (K - 1) + k, c]
    expected = torch.zeros_like(actual)
    for t in range(seq_len):
        for k in range(kernel_size):
            source = t - (kernel_size - 1) + k
            if source >= 0:
                expected[:, t] += weight[:, 0, k] * x[:, source]
    expected += bias
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_causal_depthwise_conv1d_does_not_leak_future_tokens():
    torch.manual_seed(0)
    x = torch.randn(1, 10, 3)
    weight = torch.randn(3, 1, 4)
    baseline = causal_depthwise_conv1d(x, weight=weight, bias=None)

    perturbed = x.clone()
    perturbed[:, 6] += 10.0
    outputs = causal_depthwise_conv1d(perturbed, weight=weight, bias=None)
    torch.testing.assert_close(outputs[:, :6], baseline[:, :6], rtol=1e-5, atol=1e-6)


def test_causal_depthwise_conv1d_is_channelwise_independent():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 4)
    weight = torch.randn(4, 1, 3)
    outputs = causal_depthwise_conv1d(x, weight=weight, bias=None)

    perturbed = x.clone()
    perturbed[:, :, 2] += 3.0
    perturbed_outputs = causal_depthwise_conv1d(perturbed, weight=weight, bias=None)
    unchanged = [0, 1, 3]
    torch.testing.assert_close(perturbed_outputs[:, :, unchanged], outputs[:, :, unchanged], rtol=1e-5, atol=1e-6)


def test_gated_rms_norm_normalizes_per_group():
    torch.manual_seed(0)
    hidden_size, num_groups = 8, 2
    norm = GatedRMSNorm(hidden_size=hidden_size, num_groups=num_groups)
    x = torch.randn(3, 5, hidden_size)
    # A constant gate contributes a constant positive factor, so the per-group root mean square
    # after normalization must be exactly 1 (the learnable weight is initialized to 1).
    gate = torch.full_like(x, 10.0)
    out = norm(x, gate=gate)

    grouped = out.reshape(3, 5, num_groups, hidden_size // num_groups)
    rms = grouped.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=2e-3, atol=2e-3)


def test_gated_rms_norm_groups_are_normalized_independently():
    # Scaling only the second group must leave the first group's output untouched: that is the
    # difference between grouped and global RMS normalization.
    torch.manual_seed(0)
    norm = GatedRMSNorm(hidden_size=8, num_groups=2)
    x = torch.randn(2, 3, 8)
    gate = torch.randn(2, 3, 8)
    baseline = norm(x, gate=gate)

    scaled = x.clone()
    scaled[..., 4:] *= 100.0
    outputs = norm(scaled, gate=gate)
    torch.testing.assert_close(outputs[..., :4], baseline[..., :4], rtol=1e-5, atol=1e-6)


def test_gated_rms_norm_applies_gate_before_normalization():
    torch.manual_seed(0)
    norm = GatedRMSNorm(hidden_size=4, num_groups=1)
    x = torch.randn(2, 3, 4)
    gate = torch.randn(2, 3, 4)

    expected_pre_norm = x.float() * torch.nn.functional.silu(gate.float())
    inv_rms = torch.rsqrt(expected_pre_norm.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    expected = expected_pre_norm * inv_rms
    torch.testing.assert_close(norm(x, gate=gate), expected, rtol=1e-5, atol=1e-6)


def test_gated_rms_norm_rejects_indivisible_group_count():
    with pytest.raises(ValueError, match="must be divisible by num_groups"):
        GatedRMSNorm(hidden_size=7, num_groups=2)
