import pytest
import torch
import torch.nn as nn

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer
from modalities.models.components.moe.experts import ExpertsBackend, GroupedExperts
from modalities.models.components.moe.moe import MoE
from modalities.models.components.moe.router import TopKRouter
from modalities.models.nemotron.nemotron_attention import NemotronSelfAttention
from modalities.models.nemotron.nemotron_layers import (
    Mamba2Layer,
    NemotronAttentionLayer,
    NemotronMLPLayer,
    NemotronMoELayer,
)
from modalities.models.nemotron.nemotron_mlp import SquaredReLUMLP

N_EMBD = 16


def _norm() -> nn.Module:
    return nn.RMSNorm(normalized_shape=N_EMBD, eps=1e-5)


def _mamba_layer() -> Mamba2Layer:
    torch.manual_seed(0)
    return Mamba2Layer(
        norm=_norm(),
        mixer=Mamba2Mixer(n_embd=N_EMBD, n_heads=4, head_dim=8, state_dim=4, n_groups=2, chunk_size=4),
    )


def _attention_layer() -> NemotronAttentionLayer:
    torch.manual_seed(0)
    return NemotronAttentionLayer(
        norm=_norm(),
        attn=NemotronSelfAttention(n_embd=N_EMBD, n_head_q=4, n_head_kv=2, head_dim=8),
    )


def _moe_layer() -> NemotronMoELayer:
    torch.manual_seed(0)
    experts = GroupedExperts(n_embd=N_EMBD, ffn_hidden=24, num_experts=4, backend=ExpertsBackend.LOOPED)
    with torch.no_grad():
        experts.w1.normal_(0, 0.02)
        experts.w2.normal_(0, 0.02)
    return NemotronMoELayer(
        norm=_norm(),
        moe=MoE(
            router=TopKRouter(n_embd=N_EMBD, num_experts=4, top_k=2),
            experts=experts,
            shared_experts=SquaredReLUMLP(n_embd=N_EMBD, ffn_hidden=48),
        ),
    )


def _mlp_layer() -> NemotronMLPLayer:
    torch.manual_seed(0)
    return NemotronMLPLayer(norm=_norm(), mlp=SquaredReLUMLP(n_embd=N_EMBD, ffn_hidden=24))


ALL_LAYER_BUILDERS = [_mamba_layer, _attention_layer, _moe_layer, _mlp_layer]


@pytest.mark.parametrize("build_layer", ALL_LAYER_BUILDERS, ids=lambda f: f.__name__)
def test_layer_preserves_shape(build_layer):
    layer = build_layer()
    x = torch.randn(2, 8, N_EMBD)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("build_layer", ALL_LAYER_BUILDERS, ids=lambda f: f.__name__)
def test_layer_is_a_residual_around_its_operator(build_layer):
    # Every layer must be exactly x + operator(norm(x)); a missing residual is a silent
    # convergence bug rather than a crash, so assert the identity directly.
    layer = build_layer().eval()
    x = torch.randn(2, 6, N_EMBD)
    with torch.no_grad():
        expected = x + layer._operator(layer.norm(x))
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("build_layer", ALL_LAYER_BUILDERS, ids=lambda f: f.__name__)
def test_layer_owns_exactly_one_norm(build_layer):
    layer = build_layer()
    norms = [module for module in layer.modules() if isinstance(module, (nn.RMSNorm, nn.LayerNorm))]
    # The Mamba mixer contains its own internal gated norm, which is not an nn.RMSNorm.
    assert len(norms) == 1
    assert norms[0] is layer.norm


@pytest.mark.parametrize("build_layer", ALL_LAYER_BUILDERS, ids=lambda f: f.__name__)
def test_layer_is_differentiable(build_layer):
    layer = build_layer()
    x = torch.randn(2, 6, N_EMBD, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.norm.weight.grad is not None


@pytest.mark.parametrize("build_layer", [_mamba_layer, _attention_layer], ids=lambda f: f.__name__)
def test_sequence_mixing_layers_are_causal(build_layer):
    layer = build_layer().eval()
    x = torch.randn(1, 12, N_EMBD)
    with torch.no_grad():
        baseline = layer(x)
        perturbed = x.clone()
        perturbed[:, 7] += 5.0
        outputs = layer(perturbed)
    torch.testing.assert_close(outputs[:, :7], baseline[:, :7], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("build_layer", [_moe_layer, _mlp_layer], ids=lambda f: f.__name__)
def test_feed_forward_layers_act_position_wise(build_layer):
    # MoE and MLP layers must not mix across the sequence at all: changing one position may only
    # change that position's output.
    layer = build_layer().eval()
    x = torch.randn(1, 10, N_EMBD)
    with torch.no_grad():
        baseline = layer(x)
        perturbed = x.clone()
        perturbed[:, 4] += 3.0
        outputs = layer(perturbed)

    untouched = [i for i in range(10) if i != 4]
    torch.testing.assert_close(outputs[:, untouched], baseline[:, untouched], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(outputs[:, 4], baseline[:, 4])


def test_layer_names_are_stable_for_parameter_name_filters():
    # The weight-initialization and weight-decay filters match on parameter names, so the
    # submodule attribute names are part of the public contract.
    assert "mixer.in_proj.weight" in dict(_mamba_layer().named_parameters())
    assert "attn.q_attn.weight" in dict(_attention_layer().named_parameters())
    moe_params = dict(_moe_layer().named_parameters())
    assert "moe.router.gate.weight" in moe_params
    assert "moe.experts.w1" in moe_params
    assert "moe.shared_experts.c_fc.weight" in moe_params
    assert "mlp.c_fc.weight" in dict(_mlp_layer().named_parameters())
