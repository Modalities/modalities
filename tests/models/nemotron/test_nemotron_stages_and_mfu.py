"""Tests for pipeline stage generation, the MFU calculator and the full 30B-A3B config."""

import os
from pathlib import Path

import pytest
import torch

from modalities.config.config import load_app_config_dict
from modalities.models.nemotron.layer_pattern import LayerSymbol
from modalities.models.nemotron.nemotron_stages_generator import (
    DEFAULT_LAYER_WEIGHTS,
    NemotronStagesGenerator,
    NemotronStagesGeneratorConfig,
)
from modalities.utils.nemotron_mfu import NemotronMFUCalculator
from tests.models.nemotron.test_nemotron_model import _make_model

NEMOTRON_3_NANO_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
FULL_CONFIG_PATH = Path(__file__).parents[3] / "config_files" / "training" / "config_nemotron3_nano_30b_a3b_fsdp2.yaml"


def _load_full_config() -> dict:
    """Loads the 30B-A3B config, supplying the environment values the resolvers expect.

    The config interpolates the distributed-launch environment (``LOCAL_RANK`` etc.) and the
    experiment paths, none of which exist in a plain pytest process.
    """
    launch_env = {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}
    previous = {key: os.environ.get(key) for key in launch_env}
    os.environ.update(launch_env)
    try:
        return load_app_config_dict(
            config_file_path=FULL_CONFIG_PATH,
            experiment_id="test",
            experiments_root_path=Path("/tmp/modalities_experiments"),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------------------------
# Stages generator
# --------------------------------------------------------------------------------------------


def _all_module_names(generator: NemotronStagesGenerator) -> set[str]:
    return {fqn for fqns, _ in generator._get_potential_split_points() for fqn in fqns}


@pytest.mark.parametrize(
    "layer_pattern,num_layers_per_stage,pp_dims",
    [
        ("MEM*EM-E", 6, 2),
        ("MEM*EM-E", 3, 4),
        # Cost-skewed pattern: all expensive layers first. Greedy packing against a fixed cap
        # exhausts its budget early here, which is what used to drop the trailing modules.
        ("EEEEMMMM", 6, 2),
        ("MMMMEEEE", 6, 2),
        (NEMOTRON_3_NANO_PATTERN, 14, 4),
        (NEMOTRON_3_NANO_PATTERN, 28, 2),
    ],
)
def test_stages_cover_every_module_exactly_once(layer_pattern, num_layers_per_stage, pp_dims):
    # Regression guard: a partitioner that drops the lm_head produces a model that trains without
    # an output layer instead of failing loudly.
    generator = NemotronStagesGenerator(layer_pattern=layer_pattern)
    stages = generator.get_stages(num_layers_per_stage=num_layers_per_stage, pp_dims=pp_dims)
    flat = [fqn for stage in stages for fqn in stage]

    assert set(flat) == _all_module_names(generator), "some module was not assigned to any stage"
    assert len(flat) == len(set(flat)), "a module was assigned to more than one stage"
    assert all(stage for stage in stages), "an empty pipeline stage was produced"
    assert len(stages) % pp_dims == 0


def test_stages_include_the_language_model_head():
    generator = NemotronStagesGenerator(layer_pattern="EEEEMMMM")
    stages = generator.get_stages(num_layers_per_stage=6, pp_dims=2)
    assert "transformer.lm_head" in stages[-1]
    assert "transformer.lm_head_norm" in stages[-1]
    assert "transformer.wte" in stages[0]


def test_stages_preserve_model_order():
    generator = NemotronStagesGenerator(layer_pattern="MEMEM*E-")
    stages = generator.get_stages(num_layers_per_stage=6, pp_dims=2)
    flat = [fqn for stage in stages for fqn in stage]
    layer_indices = [int(fqn.split(".")[-1]) for fqn in flat if fqn.startswith("transformer.h.")]
    assert layer_indices == sorted(layer_indices)


def test_stage_weights_are_near_balanced_for_the_full_model():
    generator = NemotronStagesGenerator(layer_pattern=NEMOTRON_3_NANO_PATTERN)
    split_points = generator._get_potential_split_points()
    weight_by_fqn = {fqns[0]: weight for fqns, weight in split_points}
    stages = generator.get_stages(num_layers_per_stage=14, pp_dims=4)

    stage_weights = [sum(weight_by_fqn[fqn] for fqn in stage if fqn in weight_by_fqn) for stage in stages]
    ideal = sum(weight_by_fqn.values()) / len(stages)
    # The slowest stage sets pipeline throughput, so the imbalance must stay small.
    assert max(stage_weights) <= ideal * 1.1


def test_stages_generator_rejects_more_stages_than_modules():
    generator = NemotronStagesGenerator(layer_pattern="ME")
    with pytest.raises(ValueError, match="Cannot build"):
        generator.get_stages(num_layers_per_stage=1, pp_dims=6)


def test_stages_generator_rejects_non_divisible_stage_counts():
    generator = NemotronStagesGenerator(layer_pattern="MEM*EM-E")
    with pytest.raises(ValueError, match="not divisible by parallel dimensions"):
        generator.get_stages(num_layers_per_stage=4, pp_dims=2)


def test_stages_generator_rejects_invalid_layers_per_stage():
    generator = NemotronStagesGenerator(layer_pattern="ME")
    with pytest.raises(ValueError, match="num_layers_per_stage must be at least 1"):
        generator.get_stages(num_layers_per_stage=0, pp_dims=1)


def test_moe_layers_are_weighted_more_heavily_than_mamba_layers():
    # The point of this generator: equal *counts* of layers would give unbalanced stages, because
    # an MoE layer performs far more work per token than a Mamba layer.
    assert DEFAULT_LAYER_WEIGHTS[LayerSymbol.MOE] > DEFAULT_LAYER_WEIGHTS[LayerSymbol.MAMBA]
    assert DEFAULT_LAYER_WEIGHTS[LayerSymbol.MAMBA] > DEFAULT_LAYER_WEIGHTS[LayerSymbol.ATTENTION]


def test_weighted_split_balances_cost_rather_than_layer_count():
    # An "all MoE first, all Mamba second" pattern: a count-based split would put 4 layers in each
    # stage, but the MoE half costs 50% more. The weighted split must give the MoE half fewer
    # layers so that the two stages take a comparable amount of time.
    generator = NemotronStagesGenerator(layer_pattern="EEEEMMMM")
    stages = generator.get_stages(num_layers_per_stage=6, pp_dims=2)
    first_stage_layers = [f for f in stages[0] if f.startswith("transformer.h.")]
    last_stage_layers = [f for f in stages[-1] if f.startswith("transformer.h.")]
    assert len(first_stage_layers) < len(last_stage_layers)


def test_custom_layer_weights_are_honoured():
    generator = NemotronStagesGenerator(
        layer_pattern="ME",
        layer_weights={"M": 1, "E": 1, "*": 1, "-": 1},
    )
    split_points = generator._get_potential_split_points()
    layer_weights = [weight for fqns, weight in split_points if fqns[0].startswith("transformer.h.")]
    assert layer_weights == [1, 1]


def test_default_layer_weights_are_applied_per_layer_type():
    generator = NemotronStagesGenerator(layer_pattern="ME*-")
    split_points = generator._get_potential_split_points()
    layer_weights = [weight for fqns, weight in split_points if fqns[0].startswith("transformer.h.")]
    assert layer_weights == [
        DEFAULT_LAYER_WEIGHTS[LayerSymbol.MAMBA],
        DEFAULT_LAYER_WEIGHTS[LayerSymbol.MOE],
        DEFAULT_LAYER_WEIGHTS[LayerSymbol.ATTENTION],
        DEFAULT_LAYER_WEIGHTS[LayerSymbol.MLP],
    ]


def test_stage_fqns_resolve_against_a_real_model():
    model = _make_model(layer_pattern="ME*-", n_layer=4)
    generator = NemotronStagesGenerator(layer_pattern="ME*-")
    stages = generator.get_stages(num_layers_per_stage=4, pp_dims=2)
    for stage in stages:
        for fqn in stage:
            assert model.get_submodule(fqn) is not None, fqn


def test_stages_generator_config_validates_the_pattern():
    with pytest.raises(ValueError, match="Invalid layer symbol"):
        NemotronStagesGeneratorConfig(layer_pattern="MEX")


def test_stages_generator_config_requires_weights_for_every_used_layer_type():
    with pytest.raises(ValueError, match=r"missing an entry for the layer types \['E'\]"):
        NemotronStagesGeneratorConfig(layer_pattern="ME", layer_weights={"M": 1})


def test_nemotron_3_nano_splits_into_four_pipeline_stages():
    generator = NemotronStagesGenerator(layer_pattern=NEMOTRON_3_NANO_PATTERN)
    total_weight = sum(weight for _, weight in generator._get_potential_split_points())
    stages = generator.get_stages(num_layers_per_stage=14, pp_dims=4)
    assert len(stages) % 4 == 0
    flat = [fqn for stage in stages for fqn in stage]
    assert len(flat) == 52 + 3  # 52 layers + wte + lm_head_norm + lm_head
    assert total_weight == 23 * 2 + 23 * 3 + 6 * 1 + 2 + 2  # mamba + moe + attention + in + out


# --------------------------------------------------------------------------------------------
# MFU calculator
# --------------------------------------------------------------------------------------------


def test_active_parameter_count_excludes_unrouted_experts():
    model = _make_model(layer_pattern="E", n_layer=1)
    total = sum(p.numel() for p in model.parameters())
    active = NemotronMFUCalculator.count_active_parameters(model)

    moe = model.transformer.h["0"].moe
    routed = sum(p.numel() for p in moe.experts.parameters())
    expected = total - int(routed * (1 - moe.router.top_k / moe.router.num_experts))
    assert active == expected
    # top-2 of 8 experts: the routed experts are the dominant parameter block, so the active
    # count must be substantially smaller than the total.
    assert active < total


def test_active_parameter_count_equals_total_for_a_dense_model():
    model = _make_model(layer_pattern="M-", n_layer=2)
    total = sum(p.numel() for p in model.parameters())
    assert NemotronMFUCalculator.count_active_parameters(model) == total


def test_nemotron_3_nano_active_parameter_ratio():
    # Model report: 31.6B total, 3.6B active including embeddings -> roughly a 9x sparsity factor.
    total, active = 31.6e9, 3.6e9
    assert total / active == pytest.approx(8.8, abs=0.3)


def test_flops_per_token_counts_only_attention_layers_quadratically():
    # 6 of 52 layers attend; a formula that charged the quadratic term to all 52 would inflate the
    # theoretical FLOPs by roughly 8x and report a correspondingly deflated MFU.
    common = dict(num_active_params=1_000_000, sequence_length=8192, n_head_q=32, head_dim=128)
    six_layers = NemotronMFUCalculator._get_theoretical_flops_per_token(num_attention_layers=6, **common)
    all_layers = NemotronMFUCalculator._get_theoretical_flops_per_token(num_attention_layers=52, **common)
    assert all_layers > six_layers

    dense_only = NemotronMFUCalculator._get_theoretical_flops_per_token(num_attention_layers=0, **common)
    assert dense_only == 6 * 1_000_000
    assert six_layers - dense_only == 12 * 6 * 8192 * 32 * 128


def test_flops_per_token_scales_linearly_in_active_parameters():
    common = dict(num_attention_layers=6, sequence_length=1024, n_head_q=8, head_dim=64)
    small = NemotronMFUCalculator._get_theoretical_flops_per_token(num_active_params=1_000, **common)
    large = NemotronMFUCalculator._get_theoretical_flops_per_token(num_active_params=2_000, **common)
    assert large - small == 6 * 1_000


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak performance lookup requires a GPU")
def test_mfu_calculator_produces_a_finite_utilization():
    model = _make_model(layer_pattern="ME*-", n_layer=4).cuda()
    calculator = NemotronMFUCalculator(
        layer_pattern="ME*-",
        sequence_length=32,
        n_embd=model.n_embd,
        n_head_q=4,
        head_dim=16,
        num_active_params=NemotronMFUCalculator.count_active_parameters(model),
        world_size=1,
        # The shared peak-performance lookup expects an FSDP-wrapped model or a list of pipeline
        # stages; a single-element list is the unwrapped equivalent.
        model_parts=[model],
    )
    mfu = calculator.compute(num_samples_per_second=torch.tensor(10.0))
    assert torch.isfinite(mfu)
    assert mfu > 0


# --------------------------------------------------------------------------------------------
# The full 30B-A3B configuration
# --------------------------------------------------------------------------------------------


def test_full_config_file_is_parseable():
    assert FULL_CONFIG_PATH.exists(), FULL_CONFIG_PATH
    config_dict = _load_full_config()
    assert config_dict["model_raw"]["variant_key"] == "nemotron"


def test_full_config_matches_the_published_architecture():
    config = _load_full_config()["model_raw"]["config"]
    assert config["n_layer"] == 52
    assert config["n_embd"] == 2688
    assert config["layer_pattern"] == NEMOTRON_3_NANO_PATTERN
    assert config["vocab_size"] == 131072
    assert config["use_weight_tying"] is False

    mamba = config["layer_specs"]["M"]["config"]
    assert (mamba["mamba_n_heads"], mamba["mamba_head_dim"]) == (64, 64)
    assert (mamba["mamba_state_dim"], mamba["mamba_n_groups"]) == (128, 8)
    assert mamba["d_conv"] == 4 and mamba["chunk_size"] == 128

    moe = config["layer_specs"]["E"]["config"]
    assert moe["num_experts"] == 128
    assert moe["moe_ffn_hidden"] == 1856
    assert moe["top_k"] == 6
    assert moe["num_shared_experts"] == 2
    assert moe["shared_expert_ffn_hidden_per_expert"] == 1856
    assert moe["score_function"] == "sigmoid"
    assert moe["use_expert_bias"] is True
    assert moe["route_scale"] == 2.5
    assert moe["aux_loss_coeff"] == pytest.approx(1e-4)

    attention = config["layer_specs"]["*"]["config"]
    assert (attention["n_head_q"], attention["n_head_kv"], attention["head_dim"]) == (32, 2, 128)


def test_full_config_wires_load_balancing_and_the_auxiliary_loss():
    config_dict = _load_full_config()

    optimizer = config_dict["optimizer"]
    assert optimizer["variant_key"] == "moe_load_balanced"
    assert optimizer["config"]["expert_bias_update_rate"] == pytest.approx(1e-3)
    excluded = optimizer["config"]["optimizer"]["config"]["weight_decay_groups_excluded"]
    assert set(excluded) == {"embedding", "layernorm", "ssm", "router"}

    loss = config_dict["loss_fn"]
    assert loss["variant_key"] == "weighted_sum"
    variants = [term["variant_key"] for term in loss["config"]["losses"]]
    assert variants == ["clm_cross_entropy_loss", "moe_aux_loss"]
    assert config_dict["model_raw"]["config"]["aux_loss_key"] == "moe_aux_loss"


def test_full_config_declares_every_layer_class_as_an_fsdp_block():
    config_dict = _load_full_config()
    block_names = set(config_dict["fsdp_model"]["config"]["block_names"])
    assert block_names == {"Mamba2Layer", "NemotronMoELayer", "NemotronAttentionLayer", "NemotronMLPLayer"}
    assert config_dict["activation_checkpointed_model"]["config"]["layers_fqn"] == "transformer.h"


def test_full_config_layer_specs_validate():
    # Build the layer specs from the real config to confirm the published hyperparameters satisfy
    # every constraint (head/group divisibility, grouped_mm alignment, top_k <= num_experts).
    from modalities.models.nemotron.nemotron_layer_specs import (
        Mamba2LayerSpec,
        NemotronAttentionLayerSpec,
        NemotronMoELayerSpec,
    )

    config = _load_full_config()["model_raw"]["config"]
    norm_config = {"norm_type": "pytorch_rms_norm", "config": {"normalized_shape": 2688, "eps": 1e-5}}

    mamba_spec = Mamba2LayerSpec(**{**config["layer_specs"]["M"]["config"], "norm_config": norm_config})
    assert mamba_spec.config.mamba_n_heads * mamba_spec.config.mamba_head_dim == 4096

    moe_spec = NemotronMoELayerSpec(**{**config["layer_specs"]["E"]["config"], "norm_config": norm_config})
    assert moe_spec.config.shared_expert_ffn_hidden == 3712

    attention_spec = NemotronAttentionLayerSpec(**{**config["layer_specs"]["*"]["config"], "norm_config": norm_config})
    assert attention_spec.config.n_head_q // attention_spec.config.n_head_kv == 16


def test_mfu_calculator_derives_active_params_when_omitted(monkeypatch):
    # Hardcoding num_active_params in a config silently goes stale when the layer pattern or the
    # expert count changes, so omitting it must derive the value from the model. The GPU peak
    # performance lookup is stubbed out: it needs an FSDP-wrapped model, which is unrelated to the
    # derivation being tested here (the wrapped path is covered by the 4-GPU config run).
    monkeypatch.setattr(
        "modalities.utils.mfu.MFUCalculatorABC._get_theoretical_gpu_peak_performance",
        staticmethod(lambda model_parts, world_size: 1.0),
    )
    model = _make_model(layer_pattern="ME*-", n_layer=4)
    expected = NemotronMFUCalculator.count_active_parameters(model)

    common = dict(layer_pattern="ME*-", sequence_length=32, n_embd=model.n_embd, n_head_q=4, head_dim=16)
    explicit = NemotronMFUCalculator(**common, num_active_params=expected, world_size=1, model_parts=model)
    derived = NemotronMFUCalculator(**common, world_size=1, model_parts=model)
    assert derived._theoretical_flops_per_token == explicit._theoretical_flops_per_token
    # And the derived value really is the sparse one, not the total parameter count.
    assert expected < sum(p.numel() for p in model.parameters())


def test_mfu_calculator_rejects_deriving_active_params_from_pipeline_stages():
    # Each stage holds only a subset of the layers, so summing per-stage counts would be wrong.
    model = _make_model(layer_pattern="ME", n_layer=2)
    with pytest.raises(ValueError, match="list of pipeline stages"):
        NemotronMFUCalculator.count_active_parameters([model, model])
    with pytest.raises(ValueError, match="no model was provided"):
        NemotronMFUCalculator.count_active_parameters(None)


def test_mfu_calculator_config_allows_omitting_active_params():
    from modalities.utils.nemotron_mfu import NemotronMFUCalculatorConfig

    config = NemotronMFUCalculatorConfig(
        layer_pattern="ME",
        sequence_length=32,
        n_embd=128,
        n_head_q=4,
        head_dim=16,
        world_size=1,
        model_parts=_make_model(layer_pattern="ME", n_layer=2),
    )
    assert config.num_active_params is None
