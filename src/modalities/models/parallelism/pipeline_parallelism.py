# Some portions of this implementation are inspired and/or adapted
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
    @abstractmethod
    def generate_fqns_per_stage(
        self, num_stages: int, num_layers: int, input_layer_equivalence: int = 1, output_layer_equivalence: int = 1
    ) -> list[list[str]]:
        """
        Generate a list of fully qualified names (FQNs) for each pipeline stage.

        Args:
            num_stages (int): Number of stages in the pipeline.
            num_layers (int): Total number of layers in the model.
            input_layer_equivalence (int): Determines to how many transformer layers
                the input layer corresponds. Default is 1.
            output_layer_equivalence (int): Determines to how many transformer layers
                the output layer corresponds. Default is 1.

        Returns:
            list[list[str]]: A list containing an FQN list for each stage.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


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
