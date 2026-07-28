"""Pipeline stage generation for the Nemotron hybrid model.

The GPT2 stages generator assumes every layer costs the same. That is a poor assumption for a hybrid
MoE model: a mixture-of-experts layer performs far more work per token than a Mamba-2 layer, and an
attention layer sits somewhere in between. Splitting such a stack into equal *counts* of layers
produces badly unbalanced pipeline stages, and the slowest stage sets the throughput of the whole
pipeline. This generator therefore assigns a per-layer-type computational weight.
"""

import math
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from modalities.models.nemotron.layer_pattern import LayerSymbol, parse_layer_pattern
from modalities.models.parallelism.pipeline_parallelism import StagesGenerator

# Relative computational weights per layer type. The pipeline packer works with integers, so the
# weights are expressed on a scale where the cheapest layer type is 1.
DEFAULT_LAYER_WEIGHTS: dict[LayerSymbol, int] = {
    LayerSymbol.MAMBA: 2,
    LayerSymbol.ATTENTION: 1,
    LayerSymbol.MOE: 3,
    LayerSymbol.MLP: 1,
}


class NemotronStagesGeneratorConfig(BaseModel):
    """
    Configuration of :class:`NemotronStagesGenerator`.

    Attributes:
        layer_pattern (str): The model's layer pattern, one symbol per layer.
        input_layer_equivalence (int): Computational weight of the embedding stage, expressed in
            units of the per-layer weights.
        output_layer_equivalence (int): Computational weight of the head stage.
        layer_weights (dict[str, int] | None): Optional override of the per-layer-type weights.
    """

    layer_pattern: str
    input_layer_equivalence: Annotated[int, Field(strict=True, ge=1)] = 2
    output_layer_equivalence: Annotated[int, Field(strict=True, ge=1)] = 2
    layer_weights: dict[str, Annotated[int, Field(strict=True, ge=1)]] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "NemotronStagesGeneratorConfig":
        layer_symbols = parse_layer_pattern(self.layer_pattern)
        if self.layer_weights is not None:
            missing = sorted({symbol.value for symbol in layer_symbols} - set(self.layer_weights))
            if missing:
                raise ValueError(f"layer_weights is missing an entry for the layer types {missing}.")
        return self


class NemotronStagesGenerator(StagesGenerator):
    """Generates fully qualified module names per pipeline stage, weighted by layer type."""

    def __init__(
        self,
        layer_pattern: str,
        input_layer_equivalence: int = 2,
        output_layer_equivalence: int = 2,
        layer_weights: dict[str, int] | None = None,
    ):
        """
        Initializes the NemotronStagesGenerator.

        Args:
            layer_pattern (str): The model's layer pattern.
            input_layer_equivalence (int): Computational weight of the embedding stage.
            output_layer_equivalence (int): Computational weight of the head stage.
            layer_weights (dict[str, int] | None): Optional override of the per-layer-type weights.
        """
        self._layer_symbols = parse_layer_pattern(layer_pattern)
        super().__init__(
            num_model_layers=len(self._layer_symbols),
            input_layer_equivalence=input_layer_equivalence,
            output_layer_equivalence=output_layer_equivalence,
        )
        if layer_weights is None:
            self._layer_weights = dict(DEFAULT_LAYER_WEIGHTS)
        else:
            self._layer_weights = {LayerSymbol(symbol): weight for symbol, weight in layer_weights.items()}

    def _get_potential_split_points(self) -> list[tuple[list[str], int]]:
        """
        Returns the candidate pipeline split points with their computational weights.

        Returns:
            list[tuple[list[str], int]]: Fully qualified module names per split point, paired with
                the relative cost of that split point.
        """
        return [
            (["transformer.wte"], self._input_layer_equivalence),
            *[
                ([f"transformer.h.{layer_idx}"], self._layer_weights[symbol])
                for layer_idx, symbol in enumerate(self._layer_symbols)
            ],
            (["transformer.lm_head_norm", "transformer.lm_head"], self._output_layer_equivalence),
        ]

    def get_stages(self, num_layers_per_stage: int, pp_dims: int) -> list[list[str]]:
        """
        Partitions the model into pipeline stages, balancing computational cost.

        The base-class implementation packs split points greedily against a fixed per-stage weight
        cap. With uniform layer weights (as in GPT2) that always consumes every split point, but
        with the per-layer-type weights used here a greedy pass can exhaust its stage budget before
        reaching the end and silently drop the trailing modules - producing a model with no output
        layer on any stage. This override therefore uses a partitioner that is guaranteed to assign
        every module to exactly one stage.

        The algorithm finds the smallest per-stage weight cap for which a left-to-right greedy pass
        fits within the requested number of stages (binary search over the cap), and then splits the
        heaviest stages further until the requested count is reached. That yields contiguous,
        complete, and near-optimally balanced stages.

        Args:
            num_layers_per_stage (int): Target number of layers per stage. Together with the input
                and output equivalences this determines the number of virtual stages.
            pp_dims (int): Pipeline parallel degree. The number of virtual stages must be a multiple
                of this.

        Raises:
            ValueError: If the number of virtual stages is not a multiple of ``pp_dims``, or if there
                are fewer split points than requested stages.

        Returns:
            list[list[str]]: Fully qualified module names per stage, in model order.
        """
        if num_layers_per_stage < 1:
            raise ValueError(f"num_layers_per_stage must be at least 1, got {num_layers_per_stage}.")

        num_virtual_stages = math.ceil(
            (self._num_model_layers + self._input_layer_equivalence + self._output_layer_equivalence)
            / num_layers_per_stage
        )
        if num_virtual_stages % pp_dims != 0:
            raise ValueError(
                f"Number of virtual stages {num_virtual_stages} is not divisible by parallel dimensions "
                f"{pp_dims}. For reference: {self._num_model_layers=} {self._input_layer_equivalence=} "
                f"{self._output_layer_equivalence=} {num_layers_per_stage=}"
            )

        split_points = self._get_potential_split_points()
        if num_virtual_stages > len(split_points):
            raise ValueError(
                f"Cannot build {num_virtual_stages} pipeline stages from only {len(split_points)} "
                f"split points. Increase num_layers_per_stage or reduce the pipeline degree."
            )

        groups = _partition_contiguous(split_points, num_parts=num_virtual_stages)
        return [[fqn for fqns, _ in group for fqn in fqns] for group in groups]


def _greedy_pack(split_points: list[tuple[list[str], int]], weight_cap: int) -> list[list[tuple[list[str], int]]]:
    """
    Packs split points left to right into contiguous groups, each at most ``weight_cap`` heavy.

    Args:
        split_points (list[tuple[list[str], int]]): The split points with their weights.
        weight_cap (int): Maximum weight per group. Must be at least the heaviest single split point.

    Returns:
        list[list[tuple[list[str], int]]]: The resulting groups. Every split point is assigned.
    """
    groups: list[list[tuple[list[str], int]]] = []
    current: list[tuple[list[str], int]] = []
    current_weight = 0
    for split_point in split_points:
        weight = split_point[1]
        if current and current_weight + weight > weight_cap:
            groups.append(current)
            current, current_weight = [], 0
        current.append(split_point)
        current_weight += weight
    if current:
        groups.append(current)
    return groups


def _best_binary_split(group: list[tuple[list[str], int]]) -> int:
    """
    Finds the index at which splitting a group minimizes the weight of its heavier half.

    Args:
        group (list[tuple[list[str], int]]): The group to split. Must contain at least two entries.

    Returns:
        int: The split index, in ``[1, len(group) - 1]``.
    """
    weights = [weight for _, weight in group]
    total = sum(weights)
    best_index, best_cost = 1, None
    prefix = 0
    for index in range(1, len(group)):
        prefix += weights[index - 1]
        cost = max(prefix, total - prefix)
        if best_cost is None or cost < best_cost:
            best_index, best_cost = index, cost
    return best_index


def _partition_contiguous(
    split_points: list[tuple[list[str], int]], num_parts: int
) -> list[list[tuple[list[str], int]]]:
    """
    Partitions split points into exactly ``num_parts`` contiguous, non-empty, balanced groups.

    Args:
        split_points (list[tuple[list[str], int]]): The split points with their weights, in order.
        num_parts (int): The exact number of groups to produce.

    Raises:
        ValueError: If there are fewer split points than requested groups.

    Returns:
        list[list[tuple[list[str], int]]]: The groups, covering every split point exactly once.
    """
    if num_parts > len(split_points):
        raise ValueError(f"Cannot partition {len(split_points)} split points into {num_parts} groups.")
    if num_parts == 1:
        return [list(split_points)]

    weights = [weight for _, weight in split_points]
    low, high = max(weights), sum(weights)
    feasible_cap = high
    while low <= high:
        candidate = (low + high) // 2
        if len(_greedy_pack(split_points, weight_cap=candidate)) <= num_parts:
            feasible_cap = candidate
            high = candidate - 1
        else:
            low = candidate + 1

    groups = _greedy_pack(split_points, weight_cap=feasible_cap)
    # The binary search only guarantees "at most num_parts" groups. Split the heaviest splittable
    # group until the requested count is reached; pipeline parallelism needs exactly this many.
    while len(groups) < num_parts:
        splittable = [index for index, group in enumerate(groups) if len(group) > 1]
        heaviest = max(splittable, key=lambda index: sum(weight for _, weight in groups[index]))
        group = groups.pop(heaviest)
        split_index = _best_binary_split(group)
        groups.insert(heaviest, group[split_index:])
        groups.insert(heaviest, group[:split_index])
    return groups
