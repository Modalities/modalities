"""Weight-initialization tests for the Nemotron hybrid model.

The interesting property here is *separation of concerns*: the generic, regex-driven initializer
handles all linear and embedding weights, while the state space parameters (A_log, D, dt_bias,
conv1d) follow their own distributions defined in ``Mamba2Mixer.reset_parameters``. If a regex ever
matched an SSM parameter, the model would still train but with a silently broken state space.
"""

import re

import pytest
import torch

from modalities.models.components.moe.experts import ExpertsBackend, GroupedExperts
from modalities.models.model_factory import ModelFactory
from modalities.nn.model_initialization.composed_initialization import ComposedInitializationRoutines
from modalities.nn.model_initialization.parameter_name_filters import (
    NAMED_PARAMETER_INIT_GROUPS,
    SupportWeightInitModels,
    WeightInitTypes,
)
from tests.models.nemotron.test_nemotron_model import _make_model

SSM_PARAMETER_SUFFIXES = ("A_log", "D", "dt_bias", "conv1d_weight", "conv1d_bias")


def _all_init_regexes() -> list[str]:
    groups = NAMED_PARAMETER_INIT_GROUPS[SupportWeightInitModels.NEMOTRON]
    regexes: list[str] = []
    for weight_init_type in WeightInitTypes:
        regex_filter = groups[weight_init_type]
        regexes.extend(regex_filter.weights)
        regexes.extend(regex_filter.biases or [])
    return regexes


def test_nemotron_is_a_supported_weight_init_model():
    assert SupportWeightInitModels("nemotron") is SupportWeightInitModels.NEMOTRON
    groups = NAMED_PARAMETER_INIT_GROUPS[SupportWeightInitModels.NEMOTRON]
    for weight_init_type in WeightInitTypes:
        assert groups[weight_init_type] is not None


def test_plain_init_covers_every_linear_and_embedding_weight():
    model = _make_model()
    plain = NAMED_PARAMETER_INIT_GROUPS[SupportWeightInitModels.NEMOTRON][WeightInitTypes.PLAIN]
    patterns = plain.weights

    for name, param in model.named_parameters():
        is_ssm = name.endswith(SSM_PARAMETER_SUFFIXES)
        is_norm = ".norm." in name or "lm_head_norm" in name
        if is_ssm or is_norm:
            continue
        assert any(re.fullmatch(p, name) for p in patterns), f"{name} is not covered by the plain init filter"


def test_no_init_regex_matches_a_state_space_parameter():
    # The load-bearing assertion of this file.
    model = _make_model()
    regexes = _all_init_regexes()
    for name, _ in model.named_parameters():
        if name.endswith(SSM_PARAMETER_SUFFIXES):
            matching = [p for p in regexes if re.fullmatch(p, name)]
            assert not matching, f"SSM parameter {name} would be overwritten by {matching}"


def test_no_init_regex_matches_a_normalization_weight():
    # Norms are initialized to one at construction time and must stay that way.
    model = _make_model()
    regexes = _all_init_regexes()
    for name, _ in model.named_parameters():
        if ".norm." in name or "lm_head_norm" in name:
            matching = [p for p in regexes if re.fullmatch(p, name)]
            assert not matching, f"Normalization parameter {name} would be overwritten by {matching}"


def test_scaled_init_covers_every_residual_output_projection():
    scaled = NAMED_PARAMETER_INIT_GROUPS[SupportWeightInitModels.NEMOTRON][WeightInitTypes.SCALED]
    model = _make_model()
    matched = {name for name, _ in model.named_parameters() if any(re.fullmatch(p, name) for p in scaled.weights)}
    # One per layer: the projection that writes back into the residual stream.
    assert matched == {
        "transformer.h.0.mixer.out_proj.weight",
        "transformer.h.1.moe.experts.w2",
        "transformer.h.1.moe.shared_experts.c_proj.weight",
        "transformer.h.2.attn.c_proj.weight",
        "transformer.h.3.mlp.c_proj.weight",
    }


def test_composed_initializer_runs_and_preserves_ssm_distributions():
    model = _make_model()
    ssm_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith(SSM_PARAMETER_SUFFIXES)
    }
    assert ssm_before, "test model must contain a Mamba layer"

    initializer = ComposedInitializationRoutines.get_composed_model_initializer(
        model_type=SupportWeightInitModels.NEMOTRON,
        weight_init_type=WeightInitTypes.SCALED,
        mean=0.0,
        std=0.02,
        num_layers=model.n_layer,
        seed=42,
    )
    initializer.initialize_in_place(model)

    for name, before in ssm_before.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], before, msg=f"{name} was modified")


def test_composed_initializer_sets_linear_weights_to_the_requested_std():
    model = _make_model()
    std = 0.02
    initializer = ComposedInitializationRoutines.get_composed_model_initializer(
        model_type=SupportWeightInitModels.NEMOTRON,
        weight_init_type=WeightInitTypes.PLAIN,
        mean=0.0,
        std=std,
        seed=42,
    )
    initializer.initialize_in_place(model)

    # The embedding is the largest single tensor, so its empirical std is the most reliable.
    embedding_std = model.transformer.wte.weight.std().item()
    assert embedding_std == pytest.approx(std, rel=0.1)


def test_scaled_init_shrinks_residual_projections_relative_to_plain():
    std, num_layers = 0.02, 4
    plain_model = _make_model()
    scaled_model = _make_model()

    for weight_init_type, model in ((WeightInitTypes.PLAIN, plain_model), (WeightInitTypes.SCALED, scaled_model)):
        initializer = ComposedInitializationRoutines.get_composed_model_initializer(
            model_type=SupportWeightInitModels.NEMOTRON,
            weight_init_type=weight_init_type,
            mean=0.0,
            std=std,
            num_layers=num_layers if weight_init_type == WeightInitTypes.SCALED else None,
            seed=42,
        )
        initializer.initialize_in_place(model)

    plain_proj = plain_model.transformer.h["0"].mixer.out_proj.weight.std().item()
    scaled_proj = scaled_model.transformer.h["0"].mixer.out_proj.weight.std().item()
    # Scaled init divides by sqrt(2 * num_layers).
    assert scaled_proj == pytest.approx(plain_proj / (2 * num_layers) ** 0.5, rel=0.15)


def test_grouped_experts_are_initialized_at_construction():
    # Regression guard: w1/w2 are raw nn.Parameter(torch.empty(...)) tensors, so without an
    # explicit reset_parameters they would contain uninitialized memory.
    experts = GroupedExperts(n_embd=16, ffn_hidden=24, num_experts=4, backend=ExpertsBackend.LOOPED)
    for weight in (experts.w1, experts.w2):
        assert torch.isfinite(weight).all()
        assert weight.abs().max() < 1.0


def test_grouped_experts_reset_parameters_is_a_noop_on_meta_device():
    with torch.device("meta"):
        experts = GroupedExperts(n_embd=16, ffn_hidden=24, num_experts=4, backend=ExpertsBackend.LOOPED)
    experts.reset_parameters()
    assert experts.w1.is_meta


def test_freshly_built_model_has_no_uninitialized_parameters():
    # Every parameter must be finite straight after construction, before any initializer runs.
    model = _make_model()
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} contains non-finite values after construction"
    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), f"buffer {name} contains non-finite values after construction"


def test_meta_device_model_is_fully_initialized_by_the_model_factory():
    # This is the real production path: build on meta, then let ModelFactory materialize, run
    # reset_parameters recursively and apply the configured initializer.
    from modalities.models.nemotron.nemotron_model_factory import NemotronModelFactory
    from tests.models.nemotron.test_nemotron_model import _model_kwargs

    model = NemotronModelFactory.get_nemotron_model(**_model_kwargs(), use_meta_device=True)
    assert all(p.is_meta for p in model.parameters())

    initializer = ComposedInitializationRoutines.get_composed_model_initializer(
        model_type=SupportWeightInitModels.NEMOTRON,
        weight_init_type=WeightInitTypes.PLAIN,
        mean=0.0,
        std=0.02,
        seed=42,
    )
    if not torch.cuda.is_available():
        pytest.skip("ModelFactory.get_weight_initialized_model materializes onto CUDA")

    model = ModelFactory.get_weight_initialized_model(model=model, model_initializer=initializer)
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"{name} is not finite after initialization"

    # The SSM parameters must have their own distributions, not the normal one.
    mixer = model.transformer.h["0"].mixer
    A = torch.exp(mixer.A_log)
    assert A.min() >= 1.0 and A.max() <= 16.0
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))


def test_initializer_matches_parameters_behind_wrapper_modules():
    # Regression guard for a silent-correctness bug: activation checkpointing (and torch.compile,
    # and FSDP1) insert wrapper segments into parameter FQNs. The initialization filters are written
    # against the plain model, so those segments must be stripped before matching - otherwise every
    # per-layer regex silently fails and the model keeps its default initialization.
    from modalities.config.config import ActivationCheckpointedModelConfig
    from modalities.models.model_factory import ModelFactory
    from modalities.nn.model_initialization.initialization_routines import normalize_parameter_name
    from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
        ActivationCheckpointingVariants,
    )

    assert (
        normalize_parameter_name("transformer.h.0._checkpoint_wrapped_module.mixer.in_proj.weight")
        == "transformer.h.0.mixer.in_proj.weight"
    )
    assert normalize_parameter_name("_orig_mod.transformer.wte.weight") == "transformer.wte.weight"

    model = ModelFactory.get_activation_checkpointed_fsdp2_model_(
        ac_variant=ActivationCheckpointingVariants.FULL_ACTIVATION_CHECKPOINTING,
        layers_fqn="transformer.h",
        model=_make_model(),
        ac_fun_params=ActivationCheckpointedModelConfig.FullACParams(),
    )
    # The wrapper really is in the names, so this test would pass trivially otherwise.
    assert any("_checkpoint_wrapped_module" in name for name, _ in model.named_parameters())

    std = 0.02
    initializer = ComposedInitializationRoutines.get_composed_model_initializer(
        model_type=SupportWeightInitModels.NEMOTRON,
        weight_init_type=WeightInitTypes.PLAIN,
        mean=0.0,
        std=std,
        seed=42,
    )
    initializer.initialize_in_place(model)

    # A per-layer parameter behind the wrapper must have been initialized to the requested std.
    in_proj = model.get_submodule("transformer.h.0").mixer.in_proj.weight
    assert in_proj.std().item() == pytest.approx(std, rel=0.15)
