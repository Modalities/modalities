import importlib.util
import math

import pytest
import torch

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer, SSDBackend

# The fused Triton kernels come from the optional `mamba` extra.
HAS_FUSED_KERNELS = all(
    importlib.util.find_spec(module_name) is not None for module_name in ("mamba_ssm", "causal_conv1d")
)

# Small but structurally faithful configuration: n_heads is a multiple of n_groups and the inner
# dimension (n_heads * head_dim) differs from n_embd, as in the real Nemotron-3 Nano.
MIXER_KWARGS = dict(
    n_embd=32,
    n_heads=4,
    head_dim=8,
    state_dim=6,
    n_groups=2,
    d_conv=4,
    chunk_size=4,
)


def _make_mixer(**overrides) -> Mamba2Mixer:
    torch.manual_seed(0)
    return Mamba2Mixer(**{**MIXER_KWARGS, **overrides})


def test_mixer_parameter_shapes_match_reference_layout():
    mixer = _make_mixer()
    d_inner = MIXER_KWARGS["n_heads"] * MIXER_KWARGS["head_dim"]
    group_state = MIXER_KWARGS["n_groups"] * MIXER_KWARGS["state_dim"]
    conv_dim = d_inner + 2 * group_state

    # in_proj packs [z, x, B, C, dt].
    assert mixer.in_proj.weight.shape == (2 * d_inner + 2 * group_state + MIXER_KWARGS["n_heads"], 32)
    assert mixer.in_proj.bias is None
    assert mixer.conv1d_weight.shape == (conv_dim, 1, MIXER_KWARGS["d_conv"])
    assert mixer.conv1d_bias.shape == (conv_dim,)
    assert mixer.A_log.shape == (MIXER_KWARGS["n_heads"],)
    assert mixer.D.shape == (MIXER_KWARGS["n_heads"],)
    assert mixer.dt_bias.shape == (MIXER_KWARGS["n_heads"],)
    assert mixer.out_proj.weight.shape == (32, d_inner)
    assert mixer.d_inner == d_inner
    assert mixer.conv_dim == conv_dim


def test_nemotron_3_nano_mixer_dimensions():
    # Model report Table 1: model dim 2688, 64 Mamba heads of dim 64, state dim 128, 8 groups.
    mixer = Mamba2Mixer(n_embd=2688, n_heads=64, head_dim=64, state_dim=128, n_groups=8, chunk_size=128)
    assert mixer.d_inner == 4096
    assert mixer.in_proj.out_features == 2 * 4096 + 2 * 8 * 128 + 64 == 10304
    assert mixer.conv_dim == 4096 + 2 * 8 * 128 == 6144
    assert mixer.norm.group_size == 512


def test_mixer_forward_preserves_shape():
    mixer = _make_mixer()
    x = torch.randn(2, 12, MIXER_KWARGS["n_embd"])
    out = mixer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("seq_len", [1, 3, 8, 13])
def test_mixer_handles_arbitrary_sequence_lengths(seq_len):
    mixer = _make_mixer()
    out = mixer(torch.randn(2, seq_len, MIXER_KWARGS["n_embd"]))
    assert out.shape == (2, seq_len, MIXER_KWARGS["n_embd"])
    assert torch.isfinite(out).all()


def test_mixer_is_causal():
    # The defining property of a sequence mixer used for causal LM: perturbing token t must not
    # change any output at a position before t. This covers the conv, the scan and the gating.
    mixer = _make_mixer().eval()
    x = torch.randn(1, 16, MIXER_KWARGS["n_embd"])
    with torch.no_grad():
        baseline = mixer(x)
        perturbed = x.clone()
        perturbed[:, 10] += 5.0
        outputs = mixer(perturbed)

    torch.testing.assert_close(outputs[:, :10], baseline[:, :10], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(outputs[:, 10], baseline[:, 10])


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 16])
def test_mixer_output_is_independent_of_chunk_size(chunk_size):
    reference = _make_mixer(chunk_size=16).eval()
    candidate = _make_mixer(chunk_size=chunk_size).eval()
    candidate.load_state_dict(reference.state_dict())

    x = torch.randn(2, 16, MIXER_KWARGS["n_embd"])
    with torch.no_grad():
        torch.testing.assert_close(candidate(x), reference(x), rtol=1e-4, atol=1e-4)


def test_mixer_batch_elements_are_independent():
    mixer = _make_mixer().eval()
    x = torch.randn(3, 10, MIXER_KWARGS["n_embd"])
    with torch.no_grad():
        batched = mixer(x)
        individually = torch.cat([mixer(x[i : i + 1]) for i in range(3)], dim=0)
    torch.testing.assert_close(batched, individually, rtol=1e-4, atol=1e-5)


def test_mixer_is_differentiable():
    mixer = _make_mixer()
    x = torch.randn(2, 8, MIXER_KWARGS["n_embd"], requires_grad=True)
    mixer(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mixer.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_reset_parameters_produces_the_reference_distributions():
    mixer = _make_mixer(n_heads=64, head_dim=8, n_groups=8)
    mixer.reset_parameters()

    # A = -exp(A_log) must lie in the sampled range [1, 16] in magnitude.
    A = torch.exp(mixer.A_log)
    assert A.min() >= 1.0 and A.max() <= 16.0
    assert mixer.A_log.dtype == torch.float32

    # softplus(dt_bias) must reproduce dt in [dt_min, dt_max].
    dt = torch.nn.functional.softplus(mixer.dt_bias)
    assert dt.min() >= mixer.dt_min * 0.99
    assert dt.max() <= mixer.dt_max * 1.01

    # D is the identity skip.
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))

    # conv1d uses the default nn.Conv1d bounds.
    bound = 1.0 / math.sqrt(mixer.conv1d_weight.size(1) * mixer.conv1d_weight.size(2))
    assert mixer.conv1d_bias.abs().max() <= bound


def test_reset_parameters_is_a_noop_on_meta_device():
    # The model factory builds on meta and materializes later; reset_parameters must not raise.
    with torch.device("meta"):
        mixer = Mamba2Mixer(**MIXER_KWARGS)
    mixer.reset_parameters()
    assert mixer.A_log.is_meta


@pytest.mark.skipif(HAS_FUSED_KERNELS, reason="mamba-ssm and causal-conv1d are installed")
def test_fused_backend_raises_a_helpful_error_when_kernels_are_missing():
    # Requesting a backend whose dependencies are absent must fail loudly at construction time
    # rather than silently falling back to a slower path. The message names only the packages that
    # are actually missing, so assert on the stable part rather than on a fixed package list -
    # exactly one of the two extras may be installed.
    with pytest.raises(ValueError, match=r"ssd_backend='fused' requires .* to be installed"):
        _make_mixer(ssd_backend=SSDBackend.FUSED)


def test_mixer_rejects_inconsistent_head_and_group_counts():
    with pytest.raises(ValueError, match="must be divisible by n_groups"):
        _make_mixer(n_heads=3, n_groups=2)


def test_mixer_rejects_invalid_kernel_and_init_ranges():
    with pytest.raises(ValueError, match="d_conv must be at least 1"):
        _make_mixer(d_conv=0)
    with pytest.raises(ValueError, match="A_init_range"):
        _make_mixer(A_init_range=(0.0, 16.0))
    with pytest.raises(ValueError, match="A_init_range"):
        _make_mixer(A_init_range=(16.0, 1.0))


def test_mixer_warns_when_native_backend_is_used_at_scale(caplog):
    with caplog.at_level("WARNING"):
        Mamba2Mixer(n_embd=2048, n_heads=8, head_dim=8, state_dim=4, n_groups=2)
    assert "native (pure-PyTorch) SSD backend" in caplog.text


@pytest.mark.parametrize("ssd_backend", [SSDBackend.NATIVE, "native"])
def test_backend_accepts_enum_and_string(ssd_backend):
    mixer = _make_mixer(ssd_backend=ssd_backend)
    assert mixer.ssd_backend == SSDBackend.NATIVE


# ------------------------------------------------------------------------------------------------
# Fused backend parity. Skipped unless the optional `mamba` extra is installed and a GPU is present.
# ------------------------------------------------------------------------------------------------

requires_fused_kernels = pytest.mark.skipif(
    not (HAS_FUSED_KERNELS and torch.cuda.is_available()),
    reason="requires the optional `mamba` extra and a CUDA device",
)

# The fused kernels require the head dimension to be a multiple of 8 and prefer realistic sizes.
FUSED_MIXER_KWARGS = dict(n_embd=256, n_heads=8, head_dim=32, state_dim=32, n_groups=2, d_conv=4, chunk_size=64)


@requires_fused_kernels
def test_fused_backend_matches_native_backend():
    # The native scan is validated against a step-by-step recurrence elsewhere, so agreement here
    # transitively validates the fused path against the mathematical definition.
    torch.manual_seed(0)
    native = Mamba2Mixer(**FUSED_MIXER_KWARGS, ssd_backend=SSDBackend.NATIVE).cuda().bfloat16().eval()
    fused = Mamba2Mixer(**FUSED_MIXER_KWARGS, ssd_backend=SSDBackend.FUSED).cuda().bfloat16().eval()
    fused.load_state_dict(native.state_dict())

    x = torch.randn(2, 128, FUSED_MIXER_KWARGS["n_embd"], device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), native(x), rtol=3e-2, atol=3e-2)


@requires_fused_kernels
def test_fused_backend_is_causal():
    mixer = Mamba2Mixer(**FUSED_MIXER_KWARGS, ssd_backend=SSDBackend.FUSED).cuda().bfloat16().eval()
    x = torch.randn(1, 128, FUSED_MIXER_KWARGS["n_embd"], device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = mixer(x)
        perturbed = x.clone()
        perturbed[:, 70] += 5.0
        outputs = mixer(perturbed)
    torch.testing.assert_close(outputs[:, :70], baseline[:, :70], rtol=2e-2, atol=2e-2)


@requires_fused_kernels
def test_fused_backend_gradients_match_native_backend():
    torch.manual_seed(0)
    native = Mamba2Mixer(**FUSED_MIXER_KWARGS, ssd_backend=SSDBackend.NATIVE).cuda().bfloat16()
    fused = Mamba2Mixer(**FUSED_MIXER_KWARGS, ssd_backend=SSDBackend.FUSED).cuda().bfloat16()
    fused.load_state_dict(native.state_dict())

    x = torch.randn(2, 128, FUSED_MIXER_KWARGS["n_embd"], device="cuda", dtype=torch.bfloat16)
    native_in, fused_in = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    native(native_in).float().sum().backward()
    fused(fused_in).float().sum().backward()
    torch.testing.assert_close(fused_in.grad, native_in.grad, rtol=5e-2, atol=5e-2)
