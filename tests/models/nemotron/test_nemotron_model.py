import pytest
import torch
import torch.nn as nn

from modalities.models.components.mamba2.mamba2_mixer import Mamba2Mixer
from modalities.models.components.moe.experts import ExpertsBackend
from modalities.models.components.moe.moe import MoE
from modalities.models.components.norms import NormWrapperConfig
from modalities.models.nemotron.layer_pattern import LayerSymbol
from modalities.models.nemotron.nemotron_attention import NemotronSelfAttention
from modalities.models.nemotron.nemotron_layer_specs import (
    Mamba2LayerSpec,
    NemotronAttentionLayerSpec,
    NemotronMLPLayerSpec,
    NemotronMoELayerSpec,
)
from modalities.models.nemotron.nemotron_layers import (
    Mamba2Layer,
    NemotronAttentionLayer,
    NemotronMLPLayer,
    NemotronMoELayer,
)
from modalities.models.nemotron.nemotron_model import NemotronLLM, NemotronLLMConfig
from modalities.models.nemotron.nemotron_model_factory import NemotronModelFactory

N_EMBD = 128
VOCAB_SIZE = 256
SEQUENCE_LENGTH = 32
# Exercises all four layer types.
LAYER_PATTERN = "ME*-"

NORM_CONFIG = {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": N_EMBD, "eps": 1e-5}}


def _layer_specs(aux_loss_coeff: float = 0.0, num_shared_experts: int = 2) -> dict:
    return {
        "M": Mamba2LayerSpec(
            n_embd=N_EMBD,
            mamba_n_heads=8,
            mamba_head_dim=16,
            mamba_state_dim=8,
            mamba_n_groups=2,
            chunk_size=8,
            norm_config=NORM_CONFIG,
        ),
        "E": NemotronMoELayerSpec(
            n_embd=N_EMBD,
            num_experts=8,
            moe_ffn_hidden=32,
            top_k=2,
            route_scale=2.5,
            num_shared_experts=num_shared_experts,
            aux_loss_coeff=aux_loss_coeff,
            experts_backend=ExpertsBackend.LOOPED,
            norm_config=NORM_CONFIG,
        ),
        "*": NemotronAttentionLayerSpec(
            n_embd=N_EMBD,
            n_head_q=4,
            n_head_kv=2,
            head_dim=16,
            attention_implementation="manual",
            norm_config=NORM_CONFIG,
        ),
        "-": NemotronMLPLayerSpec(n_embd=N_EMBD, ffn_hidden=32, norm_config=NORM_CONFIG),
    }


def _model_kwargs(**overrides) -> dict:
    kwargs = dict(
        sample_key="input_ids",
        prediction_key="logits",
        sequence_length=SEQUENCE_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=len(LAYER_PATTERN),
        layer_pattern=LAYER_PATTERN,
        layer_specs=_layer_specs(),
        lm_head_norm_config=NORM_CONFIG,
    )
    kwargs.update(overrides)
    return kwargs


def _make_model(**overrides) -> NemotronLLM:
    torch.manual_seed(0)
    kwargs = _model_kwargs(**overrides)
    kwargs["lm_head_norm_config"] = NormWrapperConfig.model_validate(kwargs["lm_head_norm_config"])
    return NemotronLLM(**kwargs)


# --------------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------------


def test_module_tree_mirrors_gpt2_layout():
    # The wrapper components (activation checkpointing, FSDP block names, pipeline splitting)
    # address the model by these names, so the layout is part of the contract.
    model = _make_model()
    assert isinstance(model.transformer, nn.ModuleDict)
    assert isinstance(model.transformer.wte, nn.Embedding)
    # Activation checkpointing requires a ModuleDict at transformer.h.
    assert isinstance(model.transformer.h, nn.ModuleDict)
    assert isinstance(model.transformer.lm_head, nn.Linear)
    assert model.get_submodule("transformer.h") is model.transformer.h


def test_layers_are_built_according_to_the_pattern():
    model = _make_model()
    expected_types = [Mamba2Layer, NemotronMoELayer, NemotronAttentionLayer, NemotronMLPLayer]
    for layer_idx, expected_type in enumerate(expected_types):
        assert isinstance(model.transformer.h[str(layer_idx)], expected_type)


def test_repeated_layer_types_do_not_share_parameters():
    # This is the central reason layer specs are builders rather than instances: two Mamba layers
    # must have independent weights.
    model = _make_model(layer_pattern="MM", n_layer=2)
    first, second = model.transformer.h["0"], model.transformer.h["1"]
    assert first is not second
    assert first.mixer.in_proj.weight is not second.mixer.in_proj.weight
    assert not torch.equal(first.mixer.in_proj.weight, second.mixer.in_proj.weight)
    assert first.norm.weight is not second.norm.weight


def test_each_layer_owns_its_own_norm():
    model = _make_model()
    norm_ids = {id(model.transformer.h[idx].norm) for idx in model.transformer.h}
    assert len(norm_ids) == len(LAYER_PATTERN)


def test_embeddings_are_untied_by_default():
    model = _make_model()
    assert not model.has_tied_word_embeddings
    assert model.transformer.wte.weight is not model.transformer.lm_head.weight


def test_weight_tying_can_be_enabled():
    model = _make_model(use_weight_tying=True)
    assert model.has_tied_word_embeddings
    assert model.transformer.wte.weight is model.transformer.lm_head.weight


def test_lm_head_property_and_skip_toggle():
    model = _make_model()
    assert model.lm_head is model.transformer.lm_head

    inputs = torch.randint(0, VOCAB_SIZE, (2, 8))
    logits = model.forward_impl(inputs)
    assert logits.shape == (2, 8, VOCAB_SIZE)

    model.set_skip_lm_head(True)
    hidden = model.forward_impl(inputs)
    assert hidden.shape == (2, 8, N_EMBD)
    # The chunked loss applies the head itself, so the two must compose back to the logits.
    torch.testing.assert_close(model.lm_head(hidden), logits, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Forward pass
# --------------------------------------------------------------------------------------------


def test_forward_with_dict_input_returns_logits_under_prediction_key():
    model = _make_model()
    inputs = {"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))}
    out = model(inputs)
    assert list(out.keys()) == ["logits"]
    assert out["logits"].shape == (2, 8, VOCAB_SIZE)
    assert torch.isfinite(out["logits"]).all()


def test_forward_with_tensor_input_returns_logits_tensor():
    # Pipeline parallelism passes raw tensors between stages.
    model = _make_model()
    out = model(torch.randint(0, VOCAB_SIZE, (2, 8)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, VOCAB_SIZE)


def test_forward_exposes_aux_loss_when_configured():
    model = _make_model(layer_specs=_layer_specs(aux_loss_coeff=1e-2), aux_loss_key="moe_aux_loss")
    out = model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})
    # Logits must stay first: InferenceResultBatch derives its batch length from the first entry.
    assert list(out.keys()) == ["logits", "moe_aux_loss"]
    assert out["moe_aux_loss"].ndim == 0


def test_forward_omits_aux_loss_when_coefficient_is_zero():
    model = _make_model(aux_loss_key="moe_aux_loss")
    out = model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})
    assert "moe_aux_loss" not in out


def test_get_aux_loss_sums_over_moe_layers():
    model = _make_model(layer_pattern="EE", n_layer=2, layer_specs=_layer_specs(aux_loss_coeff=1e-2))
    model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})
    per_layer = [model.transformer.h[idx].moe.last_aux_loss for idx in model.transformer.h]
    torch.testing.assert_close(model.get_aux_loss(), torch.stack(per_layer).sum())


def test_get_aux_loss_is_none_without_moe_layers():
    model = _make_model(layer_pattern="M-", n_layer=2)
    model({"input_ids": torch.randint(0, VOCAB_SIZE, (1, 4))})
    assert model.get_aux_loss() is None


def test_get_moe_layers_finds_every_moe_block():
    model = _make_model(layer_pattern="MEE*", n_layer=4)
    moe_layers = model.get_moe_layers()
    assert len(moe_layers) == 2
    assert all(isinstance(layer, MoE) for layer in moe_layers)


def test_model_is_causal_end_to_end():
    # A causality bug anywhere in the stack (conv, scan, attention mask) leaks the target and
    # produces a suspiciously low loss, so assert it on the assembled model too.
    model = _make_model().eval()
    inputs = torch.randint(0, VOCAB_SIZE, (1, 16))
    with torch.no_grad():
        baseline = model.forward_impl(inputs)
        perturbed = inputs.clone()
        perturbed[:, 11] = (perturbed[:, 11] + 7) % VOCAB_SIZE
        outputs = model.forward_impl(perturbed)
    torch.testing.assert_close(outputs[:, :11], baseline[:, :11], rtol=1e-4, atol=1e-4)
    assert not torch.allclose(outputs[:, 11], baseline[:, 11])


def test_forward_rejects_sequences_longer_than_configured():
    model = _make_model()
    with pytest.raises(ValueError, match="maximum input sequence length"):
        model.forward_impl(torch.randint(0, VOCAB_SIZE, (1, SEQUENCE_LENGTH + 1)))


def test_model_is_differentiable_everywhere_except_unrouted_experts():
    model = _make_model()
    out = model({"input_ids": torch.randint(0, VOCAB_SIZE, (2, 8))})
    out["logits"].sum().backward()

    # Expert weights are sparse by construction; every other parameter must receive a gradient.
    sparse_params = {"transformer.h.1.moe.experts.w1", "transformer.h.1.moe.experts.w2"}
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"
        if name not in sparse_params:
            assert param.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


# --------------------------------------------------------------------------------------------
# Weight decay groups
# --------------------------------------------------------------------------------------------


def test_weight_decay_groups_partition_all_parameters():
    import re

    model = _make_model()
    groups = model.weight_decay_groups
    assigned: dict[str, list[str]] = {}
    for name, _ in model.named_parameters():
        matches = [group for group, patterns in groups.items() if any(re.search(p, name) for p in patterns)]
        assert len(matches) == 1, f"{name} matched {matches}, expected exactly one weight decay group"
        assigned.setdefault(matches[0], []).append(name)

    # Every declared group must be non-empty for this pattern, otherwise the optimizer factory
    # would raise when the group is excluded from weight decay.
    for group in groups:
        assert assigned.get(group), f"weight decay group '{group}' is empty"


def test_ssm_parameters_are_in_their_own_weight_decay_group():
    import re

    model = _make_model()
    ssm_patterns = model.weight_decay_groups["ssm"]
    ssm_params = [name for name, _ in model.named_parameters() if any(re.search(p, name) for p in ssm_patterns)]
    # A_log, D, dt_bias, conv1d_weight, conv1d_bias of the single Mamba layer.
    assert sorted(ssm_params) == [
        "transformer.h.0.mixer.A_log",
        "transformer.h.0.mixer.D",
        "transformer.h.0.mixer.conv1d_bias",
        "transformer.h.0.mixer.conv1d_weight",
        "transformer.h.0.mixer.dt_bias",
    ]


# --------------------------------------------------------------------------------------------
# Parameter counts
# --------------------------------------------------------------------------------------------


def test_parameter_count_matches_the_analytic_formula():
    model = _make_model(layer_pattern="ME*-", n_layer=4)
    actual = sum(p.numel() for p in model.parameters())

    embedding = VOCAB_SIZE * N_EMBD
    lm_head = N_EMBD * VOCAB_SIZE
    lm_head_norm = N_EMBD
    per_layer_norm = N_EMBD

    d_inner = 8 * 16
    group_state = 2 * 8
    mamba = (
        N_EMBD * (2 * d_inner + 2 * group_state + 8)  # in_proj
        + (d_inner + 2 * group_state) * 4  # conv1d weight
        + (d_inner + 2 * group_state)  # conv1d bias
        + 8 * 3  # A_log, D, dt_bias
        + d_inner  # mixer gated norm
        + d_inner * N_EMBD  # out_proj
    )
    moe = (
        N_EMBD * 8  # router gate
        + 8 * 2 * 32 * N_EMBD  # 8 routed experts, 2 matrices each
        + 2 * (2 * 32) * N_EMBD  # fused shared expert MLP of hidden 2 * 32
    )
    attention = (4 * 16 + 2 * 16 + 2 * 16) * N_EMBD + 4 * 16 * N_EMBD
    mlp = 2 * 32 * N_EMBD

    expected = embedding + lm_head + lm_head_norm + 4 * per_layer_norm + mamba + moe + attention + mlp
    assert actual == expected


def test_nemotron_3_nano_parameter_count_matches_the_model_report():
    # Model report: 31.6B total parameters, 3.2B active (3.6B including embeddings).
    # Reproducing both numbers from the component shapes is a strong check that the expert
    # layout (non-gated, 2 fused shared experts) and the layer pattern are right.
    n_embd, vocab_size = 2688, 131072
    pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    num_mamba = pattern.count("M")
    num_moe = pattern.count("E")
    num_attn = pattern.count("*")
    assert (num_mamba, num_moe, num_attn) == (23, 23, 6)

    d_inner = 64 * 64
    group_state = 8 * 128
    mamba = (
        n_embd * (2 * d_inner + 2 * group_state + 64)
        + (d_inner + 2 * group_state) * 4
        + (d_inner + 2 * group_state)
        + 64 * 3
        + d_inner
        + d_inner * n_embd
    )
    routed_experts = 128 * 2 * 1856 * n_embd
    shared_experts = 2 * 3712 * n_embd
    moe_total = n_embd * 128 + routed_experts + shared_experts
    moe_active = n_embd * 128 + 6 * 2 * 1856 * n_embd + shared_experts
    attention = (32 * 128 + 2 * 128 + 2 * 128) * n_embd + 32 * 128 * n_embd
    embeddings = 2 * vocab_size * n_embd

    total = num_mamba * mamba + num_moe * moe_total + num_attn * attention + embeddings
    active = num_mamba * mamba + num_moe * moe_active + num_attn * attention + embeddings

    assert total / 1e9 == pytest.approx(31.6, abs=0.1)
    assert active / 1e9 == pytest.approx(3.6, abs=0.1)
    # Excluding the input embedding gives the reported 3.2B active parameters.
    assert (active - vocab_size * n_embd) / 1e9 == pytest.approx(3.2, abs=0.1)


# --------------------------------------------------------------------------------------------
# Factory and config validation
# --------------------------------------------------------------------------------------------


def test_factory_builds_a_working_model():
    model = NemotronModelFactory.get_nemotron_model(**_model_kwargs())
    out = model({"input_ids": torch.randint(0, VOCAB_SIZE, (1, 4))})
    assert out["logits"].shape == (1, 4, VOCAB_SIZE)


def test_factory_supports_meta_device():
    model = NemotronModelFactory.get_nemotron_model(**_model_kwargs(), use_meta_device=True)
    assert all(p.is_meta for p in model.parameters())
    # Materializing must yield finite values once reset_parameters has run.
    materialized = model.to_empty(device="cpu")
    for module in materialized.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    for name, param in materialized.named_parameters():
        if "mixer" in name:
            assert torch.isfinite(param).all(), f"{name} is not finite after materialization"


def test_factory_rejects_weight_tying_on_meta_device():
    with pytest.raises(ValueError, match="Weight tying is not supported on the meta device"):
        NemotronModelFactory.get_nemotron_model(**_model_kwargs(), use_meta_device=True, use_weight_tying=True)


def test_config_rejects_layer_count_mismatch():
    with pytest.raises(ValueError, match="does not match the length of layer_pattern"):
        NemotronLLMConfig(**_model_kwargs(n_layer=7))


def test_config_rejects_missing_layer_spec():
    specs = _layer_specs()
    del specs["*"]
    with pytest.raises(ValueError, match=r"layer_pattern uses the layer types \['\*'\]"):
        NemotronLLMConfig(**_model_kwargs(layer_specs=specs))


def test_config_rejects_spec_registered_under_the_wrong_symbol():
    specs = _layer_specs()
    specs["M"] = specs["-"]
    with pytest.raises(ValueError, match="is a spec for layer type"):
        NemotronLLMConfig(**_model_kwargs(layer_specs=specs))


def test_config_warns_about_unused_layer_specs(caplog):
    with caplog.at_level("WARNING"):
        NemotronLLMConfig(**_model_kwargs(layer_pattern="ME", n_layer=2))
    assert "unused layer types" in caplog.text


@pytest.mark.parametrize("field,value", [("n_embd", 130), ("vocab_size", 250)])
def test_config_enforces_tensor_core_alignment(field, value):
    with pytest.raises(ValueError, match="should be divisible by 128"):
        NemotronLLMConfig(**_model_kwargs(**{field: value}))


def test_config_alignment_check_can_be_disabled():
    config = NemotronLLMConfig(**_model_kwargs(n_embd=130, enforce_tensor_core_alignment=False))
    assert config.n_embd == 130


def test_model_rejects_missing_spec_at_construction_time():
    specs = _layer_specs()
    del specs["E"]
    with pytest.raises(ValueError, match="No layer spec provided"):
        _make_model(layer_specs=specs)


# --------------------------------------------------------------------------------------------
# Layer specs
# --------------------------------------------------------------------------------------------


def test_layer_specs_report_their_symbol():
    specs = _layer_specs()
    assert specs["M"].symbol == LayerSymbol.MAMBA
    assert specs["E"].symbol == LayerSymbol.MOE
    assert specs["*"].symbol == LayerSymbol.ATTENTION
    assert specs["-"].symbol == LayerSymbol.MLP


def test_layer_spec_build_returns_independent_modules():
    spec = _layer_specs()["M"]
    first, second = spec.build(layer_idx=0), spec.build(layer_idx=1)
    assert first is not second
    assert first.mixer.in_proj.weight is not second.mixer.in_proj.weight


def test_mamba_spec_rejects_indivisible_group_count():
    with pytest.raises(ValueError, match="must be divisible by"):
        Mamba2LayerSpec(
            n_embd=N_EMBD,
            mamba_n_heads=6,
            mamba_head_dim=16,
            mamba_state_dim=8,
            mamba_n_groups=4,
            norm_config=NORM_CONFIG,
        )


def test_attention_spec_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="must be divisible by n_head_kv"):
        NemotronAttentionLayerSpec(n_embd=N_EMBD, n_head_q=3, n_head_kv=2, head_dim=16, norm_config=NORM_CONFIG)


def test_moe_spec_rejects_top_k_above_num_experts():
    with pytest.raises(ValueError, match="must not exceed num_experts"):
        NemotronMoELayerSpec(n_embd=N_EMBD, num_experts=4, moe_ffn_hidden=32, top_k=5, norm_config=NORM_CONFIG)


def test_moe_spec_rejects_unknown_router_dtype():
    with pytest.raises(ValueError, match="router_dtype must be"):
        NemotronMoELayerSpec(
            n_embd=N_EMBD,
            num_experts=4,
            moe_ffn_hidden=32,
            top_k=2,
            router_dtype="float64",
            norm_config=NORM_CONFIG,
        )


def test_moe_spec_fuses_shared_experts_into_one_mlp():
    # The reference implementation represents N shared experts as a single MLP of N times the
    # per-expert hidden size (moe_shared_expert_intermediate_size = 2 * 1856 for Nemotron).
    spec = NemotronMoELayerSpec(
        n_embd=N_EMBD,
        num_experts=8,
        moe_ffn_hidden=32,
        top_k=2,
        num_shared_experts=2,
        experts_backend=ExpertsBackend.LOOPED,
        norm_config=NORM_CONFIG,
    )
    assert spec.config.shared_expert_ffn_hidden == 64
    layer = spec.build(layer_idx=0)
    assert layer.moe.shared_experts.c_fc.out_features == 64


def test_moe_spec_can_disable_shared_experts():
    spec = NemotronMoELayerSpec(
        n_embd=N_EMBD,
        num_experts=8,
        moe_ffn_hidden=32,
        top_k=2,
        num_shared_experts=0,
        experts_backend=ExpertsBackend.LOOPED,
        norm_config=NORM_CONFIG,
    )
    assert spec.config.shared_expert_ffn_hidden == 0
    assert spec.build(layer_idx=0).moe.shared_experts is None


def test_specs_produce_the_expected_component_types():
    specs = _layer_specs()
    assert isinstance(specs["M"].build(0).mixer, Mamba2Mixer)
    assert isinstance(specs["E"].build(0).moe, MoE)
    assert isinstance(specs["*"].build(0).attn, NemotronSelfAttention)
