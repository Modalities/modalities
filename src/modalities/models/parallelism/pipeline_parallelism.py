# Some portions of this implementation are inspired, adapted, or refactored
# from Meta's open-source project TorchTitan,
# licensed under the BSD 3-Clause License.

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining.schedules import PipelineScheduleSingle, get_schedule_class

from modalities.running_env.fsdp.device_mesh import ParallelismDegrees


class FQNsPerStageGenerator(ABC):
    def generate_fqns_per_stage(
        self, num_stages: int, num_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1
    ) -> list[list[str]]:
        """
        Generate FQNs for each stage in a GPT-2 model.

        Args:
            num_stages (int): Number of stages in the pipeline.
            num_layers (int): Total number of layers in the model.
            input_layer_equivalence (int): Number of layers corresponding to the input layer.
            output_layer_equivalence (int): Number of layers corresponding to the output layer.

        Returns:
            list[list[str]]: A list containing FQNs for each stage.
        """

        # Potential split points for GPT-2 model with each potential split point
        # listing the FQNs of the modules in that stage and the computational weight.
        # The computational weight of the input and output modules are estimated
        # based on the number of layers they correspond to.
        potential_split_points = self._get_potential_split_points(
            num_layers=num_layers,
            input_layer_equivalence=input_layer_equivalence,
            output_layer_equivalence=output_layer_equivalence,
        )
        # Calculate the weight per stage based on the total weight and number of stages
        weight_per_stage = math.ceil(sum(weight for _, weight in potential_split_points) / num_stages)
        # pack the stages with the layers
        next_split_point = 0
        module_names_per_stage: list[list[str]] = []
        for _ in range(num_stages):
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
    def _get_potential_split_points(
        self, num_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1
    ) -> list[tuple[list[str], int]]:
        """
        Returns a list of potential split points for the GPT-2 model.

        Args:
            num_layers (int): Total number of layers in the model.
            input_layer_equivalence (int): Number of layers corresponding to the input layer.
            output_layer_equivalence (int): Number of layers corresponding to the output layer.

        Returns:
            list[tuple[list[str], int]]: A list containing tuples of FQNs and their computational weights.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class GPT2LLMFQNsPerStageGenerator(FQNsPerStageGenerator):
    def _get_potential_split_points(
        self, num_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1
    ) -> list[tuple[list[str], int]]:
        """
        Returns a list of potential split points for the GPT-2 model.

        Args:
            num_layers (int): Total number of layers in the model.
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
            (["transformer.wte", "transformer.wpe"], input_layer_equivalence),
            *[([f"transformer.h.{i}"], 1) for i in range(num_layers)],
            (["transformer.lm_head_norm", "transformer.lm_head"], output_layer_equivalence),
        ]

        return potential_split_points


class PipelineFactory:
    """Pipeline factory class to create pipelined models."""

    @staticmethod
    def create_pipeline_model(
        num_layers: int,
        fqns_per_stage_generator: FQNsPerStageGenerator,
        device_mesh: DeviceMesh,
        pp_schedule_name: str,
        num_layers_per_stage: int,
        input_layer_equivalence: Optional[int] = 1,
        output_layer_equivalence: Optional[int] = 1,
    ) -> torch.nn.Module:
        device_mesh[ParallelismDegrees.PP.value]
        pp_dims = device_mesh.size(ParallelismDegrees.PP.value)
        schedule_class = get_schedule_class(pp_schedule_name)
        is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
        if not is_single_stage_schedule:
            raise ValueError(
                f"Unsupported pipeline schedule: {pp_schedule_name}. We only support single-stage schedules."
            )

        # calculate the number of stages
        num_virtual_stages = math.ceil(
            (num_layers + input_layer_equivalence + output_layer_equivalence) / num_layers_per_stage
        )
        if num_virtual_stages % pp_dims != 0:
            raise ValueError(
                f"Number of virtual stages {num_virtual_stages} is not divisible by parallel dimensions {pp_dims}. "
                f"For reference: {num_layers=} {input_layer_equivalence=} "
                f"{output_layer_equivalence=} {num_layers_per_stage=}"
            )

        stages_per_rank = num_virtual_stages // pp_dims
        if stages_per_rank != 1:
            raise ValueError(
                f"Stages per rank {stages_per_rank} must be 1 for single-stage schedules. "
                f"Please adjust {num_layers_per_stage=} to ensure each PP rank has exactly one stage."
            )

        fqns_per_stage_generator.generate_fqns_per_stage(
            num_stages=num_virtual_stages,
            num_layers=num_layers,
            input_layer_equivalence=input_layer_equivalence,
            output_layer_equivalence=output_layer_equivalence,
        )

    @staticmethod
    def create_gpt2_model_splitter():
        """Create a GPT-2 model splitter for pipeline parallelism."""
        pass
