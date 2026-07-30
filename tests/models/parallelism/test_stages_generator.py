"""Tests that pipeline stage generation assigns every module to exactly one stage.

The generator used to pack split points greedily against a fixed per-stage weight cap while looping
exactly ``num_virtual_stages`` times. Anything left over when the packing did not fit was silently
discarded - in practice the output split point, producing a pipeline with no ``lm_head`` on any
stage. Uniform per-layer weights (as in GPT2) always happen to fit, so this only surfaced for a
generator with non-uniform weights.
"""

import pytest

from modalities.models.parallelism.stages_generator import GPT2LLMStagesGenerator, StagesGenerator


class _NonUniformStagesGenerator(StagesGenerator):
    """A generator whose layers have differing computational cost, e.g. a hybrid MoE stack."""

    def __init__(self, layer_weights: list[int], input_layer_equivalence: int = 2, output_layer_equivalence: int = 2):
        super().__init__(
            num_model_layers=len(layer_weights),
            input_layer_equivalence=input_layer_equivalence,
            output_layer_equivalence=output_layer_equivalence,
        )
        self._layer_weights = layer_weights

    def _get_potential_split_points(self) -> list[tuple[list[str], int]]:
        return [
            (["transformer.wte"], self._input_layer_equivalence),
            *[([f"transformer.h.{i}"], w) for i, w in enumerate(self._layer_weights)],
            (["transformer.lm_head_norm", "transformer.lm_head"], self._output_layer_equivalence),
        ]


def _all_modules(generator: StagesGenerator) -> set[str]:
    return {fqn for fqns, _ in generator._get_potential_split_points() for fqn in fqns}


@pytest.mark.parametrize(
    "layer_weights,num_layers_per_stage,pp_dims",
    [
        # Expensive layers first: the greedy pack exhausts its budget early and used to drop the tail.
        ([3, 3, 3, 3, 2, 2, 2, 2], 6, 2),
        ([2, 2, 2, 2, 3, 3, 3, 3], 6, 2),
        ([3, 2, 3, 2, 3, 1, 3, 2], 6, 2),
        ([3, 2, 3, 2, 3, 1, 3, 2], 3, 4),
        # A Nemotron-3 Nano-shaped stack: 23 Mamba (2), 23 MoE (3), 6 attention (1).
        ([2, 3] * 23 + [1] * 6, 14, 4),
        ([2, 3] * 23 + [1] * 6, 28, 2),
    ],
)
def test_every_module_is_assigned_exactly_once(layer_weights, num_layers_per_stage, pp_dims):
    generator = _NonUniformStagesGenerator(layer_weights)
    stages = generator.get_stages(num_layers_per_stage=num_layers_per_stage, pp_dims=pp_dims)
    flat = [fqn for stage in stages for fqn in stage]

    assert set(flat) == _all_modules(generator), "a module was dropped"
    assert len(flat) == len(set(flat)), "a module was assigned to more than one stage"
    assert all(stage for stage in stages), "an empty pipeline stage was produced"
    assert len(stages) % pp_dims == 0


def test_output_layer_is_never_dropped():
    # The concrete failure: transformer.lm_head vanished from every stage.
    generator = _NonUniformStagesGenerator([3, 3, 3, 3, 2, 2, 2, 2])
    stages = generator.get_stages(num_layers_per_stage=6, pp_dims=2)
    assert "transformer.lm_head" in stages[-1]
    assert "transformer.lm_head_norm" in stages[-1]
    assert "transformer.wte" in stages[0]


def test_stages_preserve_model_order():
    generator = _NonUniformStagesGenerator([2, 3, 2, 3, 1, 2, 3, 2])
    stages = generator.get_stages(num_layers_per_stage=6, pp_dims=2)
    flat = [fqn for stage in stages for fqn in stage]
    layer_indices = [int(f.split(".")[-1]) for f in flat if f.startswith("transformer.h.")]
    assert layer_indices == sorted(layer_indices)


def test_stage_weights_are_near_balanced():
    layer_weights = [2, 3] * 23 + [1] * 6
    generator = _NonUniformStagesGenerator(layer_weights)
    weight_by_fqn = {fqns[0]: w for fqns, w in generator._get_potential_split_points()}
    stages = generator.get_stages(num_layers_per_stage=14, pp_dims=4)

    stage_weights = [sum(weight_by_fqn[f] for f in stage if f in weight_by_fqn) for stage in stages]
    ideal = sum(weight_by_fqn.values()) / len(stages)
    # The slowest stage sets pipeline throughput, so the imbalance must stay small.
    assert max(stage_weights) <= ideal * 1.15


def test_rejects_more_stages_than_split_points():
    generator = _NonUniformStagesGenerator([1, 1])
    with pytest.raises(ValueError, match="Cannot build"):
        generator.get_stages(num_layers_per_stage=1, pp_dims=6)


# --------------------------------------------------------------------------------------------
# GPT2 behaviour must be unchanged: it was never affected, and its stages should stay complete.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_model_layers,num_layers_per_stage,pp_dims",
    [(12, 7, 2), (12, 14, 1), (4, 3, 2), (24, 13, 2), (8, 5, 2)],
)
def test_gpt2_stages_remain_complete(num_model_layers, num_layers_per_stage, pp_dims):
    generator = GPT2LLMStagesGenerator(num_model_layers=num_model_layers)
    stages = generator.get_stages(num_layers_per_stage=num_layers_per_stage, pp_dims=pp_dims)
    flat = [fqn for stage in stages for fqn in stage]

    assert set(flat) == _all_modules(generator)
    assert len(flat) == len(set(flat))
    assert len(stages) % pp_dims == 0
    layer_indices = [int(f.split(".")[-1]) for f in flat if f.startswith("transformer.h.")]
    assert layer_indices == sorted(layer_indices)


def test_non_divisible_stage_count_still_raises():
    generator = GPT2LLMStagesGenerator(num_model_layers=12)
    with pytest.raises(ValueError, match="not divisible by parallel dimensions"):
        generator.get_stages(num_layers_per_stage=5, pp_dims=2)
