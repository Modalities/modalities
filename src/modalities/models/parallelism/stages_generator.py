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
        Generate FQNs for each stage in a GPT-2 model.

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

        stages_per_rank = num_virtual_stages // pp_dims
        if stages_per_rank != 1:
            raise ValueError(
                f"Stages per rank {stages_per_rank} must be 1 for single-stage schedules. "
                f"Please adjust {num_layers_per_stage=} to ensure each PP rank has exactly one stage."
            )

        # Potential split points for GPT-2 model with each potential split point
        # listing the FQNs of the modules in that stage and the computational weight.
        # The computational weight of the input and output modules are estimated
        # based on the number of layers they correspond to.
        potential_split_points = self._get_potential_split_points()
        # Calculate the weight per stage based on the total weight and number of stages
        weight_per_stage = math.ceil(sum(weight for _, weight in potential_split_points) / num_virtual_stages)
        # pack the stages with the layers
        next_split_point = 0
        module_names_per_stage: list[list[str]] = []
        for _ in range(num_virtual_stages):
            stage_fqns = []
            stage_weight = 0
            while next_split_point < len(potential_split_points):
                fqns, weight = potential_split_points[next_split_point]
                if weight > weight_per_stage:
                    raise ValueError(
                        f"Weight of {weight} for {fqns} exceeds weight per stage {weight_per_stage}. "
                        "Please adjust the number of stages or the weight distribution."
                    )
                if stage_weight + weight > weight_per_stage:
                    break
                stage_fqns.extend(fqns)
                stage_weight += weight
                next_split_point += 1
            module_names_per_stage.append(stage_fqns)

        return module_names_per_stage

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
            (["transformer.wte", "transformer.wpe", "transformer.drop"], self._input_layer_equivalence),
            *[([f"transformer.h.{i}"], 1) for i in range(self._num_model_layers)],
            (["transformer.lm_head_norm", "transformer.lm_head"], self._output_layer_equivalence),
        ]

        return potential_split_points
