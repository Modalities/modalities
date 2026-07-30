# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import math
from abc import ABC, abstractmethod


class StagesGenerator(ABC):
    def __init__(self, num_model_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1):
        self._num_model_layers = num_model_layers
        self._input_layer_equivalence = input_layer_equivalence
        self._output_layer_equivalence = output_layer_equivalence

    def get_stages(self, num_layers_per_stage: int, pp_dims: int) -> list[list[str]]:
        """
        Generate FQNs for each stage in a model.

        Args:
            num_layers_per_stage (int): Number of layers per stage.
            pp_dims (int): Number of pipeline parallel dimensions.

        Returns:
            list[list[str]]: A list containing FQNs for each stage.
        """

        # calculate the number of stages
        num_virtual_stages = math.ceil(
            (self._num_model_layers + self._input_layer_equivalence + self._output_layer_equivalence)
            / num_layers_per_stage
        )
        if num_virtual_stages % pp_dims != 0:
            raise ValueError(
                f"Number of virtual stages {num_virtual_stages} is not divisible by parallel dimensions {pp_dims}. "
                f"For reference: {self._num_model_layers=} {self._input_layer_equivalence=} "
                f"{self._output_layer_equivalence=} {num_layers_per_stage=}"
            )

        # Potential split points for GPT-2 model with each potential split point
        # listing the FQNs of the modules in that stage and the computational weight.
        # The computational weight of the input and output modules are estimated
        # based on the number of layers they correspond to.
        potential_split_points = self._get_potential_split_points()
        if num_virtual_stages > len(potential_split_points):
            raise ValueError(
                f"Cannot build {num_virtual_stages} pipeline stages from only "
                f"{len(potential_split_points)} split points. Increase num_layers_per_stage or "
                f"reduce the pipeline degree."
            )
        # Pack the split points into contiguous stages, balancing computational weight.
        #
        # This used to pack greedily against a fixed per-stage weight cap, looping exactly
        # num_virtual_stages times. When the packing did not happen to fit, whatever was left over
        # was never assigned to any stage and was silently discarded - typically the output split
        # point, producing a pipeline with no lm_head on any stage. Uniform per-layer weights (as
        # in GPT2) always happen to fit, which is why this went unnoticed; any generator with
        # non-uniform weights hits it.
        groups = _partition_contiguous(potential_split_points, num_parts=num_virtual_stages)
        return [[fqn for fqns, _ in group for fqn in fqns] for group in groups]

    @abstractmethod
    def _get_potential_split_points(self) -> list[tuple[list[str], int]]:
        """
        Returns a list of potential split points for the GPT-2 model.

        Args:
            num_model_layers (int): Total number of layers in the model.
            input_layer_equivalence (int): Number of layers corresponding to the input layer.
            output_layer_equivalence (int): Number of layers corresponding to the output layer.

        Returns:
            list[tuple[list[str], int]]: A list containing tuples of FQNs and their computational weights.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class GPT2LLMStagesGenerator(StagesGenerator):
    def __init__(self, num_model_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1):
        super().__init__(num_model_layers, input_layer_equivalence, output_layer_equivalence)

    def _get_potential_split_points(
        self,
    ) -> list[tuple[list[str], int]]:
        """
        Returns a list of potential split points for the GPT-2 model.

        Args:
            num_model_layers (int): Total number of layers in the model.
            input_layer_equivalence (int): Number of layers corresponding to the input layer.
            output_layer_equivalence (int): Number of layers corresponding to the output layer.

        Returns:
            list[tuple[list[str], int]]: A list containing tuples of FQNs and their computational weights.
        """

        # Potential split points for GPT-2 model with each potential split point
        # listing the FQNs of the modules in that stage and the computational weight.
        # The computational weight of the input and output modules are estimated
        # based on the number of layers they correspond to.
        potential_split_points = [
            (
                ["transformer.wte", "transformer.wpe", "transformer.drop"],
                self._input_layer_equivalence,
            ),
            *[([f"transformer.h.{i}"], 1) for i in range(self._num_model_layers)],
            (["transformer.lm_head_norm", "transformer.lm_head"], self._output_layer_equivalence),
        ]

        return potential_split_points


def _greedy_pack(split_points: list[tuple[list[str], int]], weight_cap: int) -> list[list[tuple[list[str], int]]]:
    """
    Packs split points left to right into contiguous groups, each at most ``weight_cap`` heavy.

    Unlike a fixed-stage-count loop, this consumes every split point: a group is closed and a new
    one started whenever the cap would be exceeded.

    Args:
        split_points (list[tuple[list[str], int]]): The split points with their weights, in order.
        weight_cap (int): Maximum weight per group. Must be at least the heaviest split point.

    Returns:
        list[list[tuple[list[str], int]]]: The resulting groups, covering every split point.
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
        group (list[tuple[list[str], int]]): The group to split, with at least two entries.

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

    Finds the smallest per-group weight cap for which a left-to-right greedy pass fits within
    ``num_parts`` groups (binary search over the cap), then splits the heaviest splittable group
    until the requested count is reached. Every split point is assigned exactly once.

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
