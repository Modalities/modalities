from enum import Enum
from math import prod
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from modalities.exceptions import ConfigError
from modalities.utils.logger_utils import get_logger

logger = get_logger("model_factory")


class DeviceMeshConfig(BaseModel):
    # inspired by ParallelDims class in
    # https://github.com/pytorch/torchtitan/blob/cfc0f4e08dc71685cdcb394464187d2eeedd1a5f/torchtitan/parallelisms/parallel_dims.py#L15
    device_type: str = "cuda"
    # if -1, we will calculate the shard degree based on the world size and other parallel degrees
    data_parallel_replicate_degree: Annotated[int, Field(strict=True, ge=-1)] = 1
    data_parallel_shard_degree: Annotated[int, Field(strict=True, ge=-1)]
    tensor_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    pipeline_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    context_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    enable_loss_parallel: Optional[bool] = False
    world_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def _validate(self):
        for d in (
            self.context_parallel_degree,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
        ):
            if d < 1:
                raise ConfigError(
                    "Parallelism degree must be >= 1, except for data_parallel_shard_degree "
                    "and data_parallel_replicate_degree"
                )

        if not (self.data_parallel_shard_degree == -1 or self.data_parallel_shard_degree >= 1):
            raise ConfigError("data_parallel_shard_degree must be -1 or >= 1")
        if not (self.data_parallel_replicate_degree == -1 or self.data_parallel_replicate_degree >= 1):
            raise ConfigError("data_parallel_replicate_degree must be -1 or >= 1")

        if self.data_parallel_replicate_degree == -1 and self.data_parallel_shard_degree == -1:
            raise ConfigError("At most one of data_parallel_replicate_degree and data_parallel_shard_degree can be -1")

        if self.data_parallel_shard_degree == -1:
            # set the shard degree to the world size divided by the product of all other parallel degrees
            self.data_parallel_shard_degree = self.world_size // (
                self.data_parallel_replicate_degree
                * self.context_parallel_degree
                * self.tensor_parallel_degree
                * self.pipeline_parallel_degree
            )
        if self.data_parallel_replicate_degree == -1:
            # set the replicate degree to the world size divided by the product of all other parallel degrees
            self.data_parallel_replicate_degree = self.world_size // (
                self.data_parallel_shard_degree
                * self.context_parallel_degree
                * self.tensor_parallel_degree
                * self.pipeline_parallel_degree
            )
        if (
            self.data_parallel_shard_degree
            * self.data_parallel_replicate_degree
            * self.tensor_parallel_degree
            * self.pipeline_parallel_degree
            * self.context_parallel_degree
            != self.world_size
        ):
            raise ConfigError(
                f"Invalid parallel dims: data_parallel_shard_degree({self.data_parallel_shard_degree}) * "
                f"data_parallel_replicate_degree({self.data_parallel_replicate_degree}) * "
                f"tensor_parallel_degree({self.tensor_parallel_degree}) *"
                f"* pipeline_parallel_degree({self.pipeline_parallel_degree}) *"
                f"context_parallel_degree({self.context_parallel_degree})!= WORLD_SIZE({self.world_size})"
            )
        if self.enable_loss_parallel and self.tensor_parallel_degree <= 1:
            raise ConfigError(f"{self.enable_loss_parallel=} requires tensor_parallel_degree > 1")
        return self


class ParallelismDegrees(Enum):
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    CP = "cp"
    TP = "tp"
    PP = "pp"


def get_device_mesh(
    device_type: str,
    data_parallel_replicate_degree: int,
    data_parallel_shard_degree: int,
    tensor_parallel_degree: int,
    pipeline_parallel_degree: int,
    context_parallel_degree: int,
    enable_loss_parallel: bool,
    world_size: int,
) -> DeviceMesh:
    """
    Gets the device mesh for the specified parallelism degrees.

    Args:
        device_type (str): The device type.
        data_parallel_replicate_degree (int): The data parallel replicate degree.
        data_parallel_shard_degree (int): The data parallel shard degree.
        tensor_parallel_degree (int): The tensor parallel degree.
        pipeline_parallel_degree (int): The pipeline parallel degree.
        context_parallel_degree (int): The context parallel degree.
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        world_size (int): The world size.

    Returns:
        DeviceMesh: The device mesh.
    """
    dims = []
    names = []
    for dim, name in zip(
        [
            pipeline_parallel_degree,
            data_parallel_replicate_degree,
            data_parallel_shard_degree,
            context_parallel_degree,
            tensor_parallel_degree,
        ],
        [
            ParallelismDegrees.PP.value,
            ParallelismDegrees.DP_REPLICATE.value,
            ParallelismDegrees.DP_SHARD.value,
            ParallelismDegrees.CP.value,
            ParallelismDegrees.TP.value,
        ],
        strict=True,
    ):
        if dim > 1 or name == ParallelismDegrees.DP_SHARD.value:
            dims.append(dim)
            names.append(name)
    names = tuple(names)
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
    logger.info(f"{device_mesh=} | {world_size=} | {enable_loss_parallel=}")
    # TODO: Torch Titan had some more checks here. We need to check if we also need those:
    # https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/distributed/parallel_dims.py#L86-L104
    return device_mesh


def get_parallel_degree(device_mesh: DeviceMesh, parallelism_methods: list[ParallelismDegrees]) -> int:
    """Gets the number of parallel ranks (i.e., the parallelism degree)
    from the device mesh for a specific parallelism method.
    Args:
        device_mesh (DeviceMesh): The device mesh.
        parallelism_methods (list[ParallelismDegrees]): The parallelism methods.
    Returns:
        int: The number of parallel ranks for the specified parallelism method.
    """
    if device_mesh.mesh_dim_names is None:
        raise ValueError("device_mesh.mesh_dim_names is None")

    return prod(
        device_mesh.size(device_mesh.mesh_dim_names.index(method.value))
        for method in parallelism_methods
        if method.value in device_mesh.mesh_dim_names
    )


def has_parallelism_method(device_mesh: DeviceMesh | None, parallelism_method: ParallelismDegrees) -> bool:
    """Checks if the device mesh has the specified parallelism method.

    Args:
        device_mesh (DeviceMesh | None): The device mesh.
        parallelism_method (ParallelismDegrees): The parallelism method.

    Returns:
        bool: True if the device mesh has the specified parallelism method, False otherwise.
    """
    return (
        device_mesh is not None
        and (mesh_dim_names := device_mesh.mesh_dim_names) is not None
        and parallelism_method.value in mesh_dim_names
    )


def get_mesh_for_parallelism_method(device_mesh: DeviceMesh, parallelism_method: ParallelismDegrees) -> DeviceMesh:
    """Gets the sub-mesh for the specified parallelism method.

    Args:
        device_mesh (DeviceMesh): The device mesh.
        parallelism_method (ParallelismDegrees): The parallelism method.

    Returns:
        DeviceMesh: The sub-mesh for the specified parallelism method.
    """
    if not has_parallelism_method(device_mesh, parallelism_method):
        raise ValueError(f"Device mesh does not have parallelism method {parallelism_method}.")
    return device_mesh[parallelism_method.value]


def get_parallel_rank(device_mesh: DeviceMesh, parallelism_method: ParallelismDegrees) -> int:
    """Gets the parallel rank ID for the specified parallelism method.

    Args:
        device_mesh (DeviceMesh): The device mesh.
        parallelism_method (ParallelismDegrees): The parallelism method.

    Returns:
        int: The parallel rank ID for the specified parallelism method.
    """
    sub_mesh = get_mesh_for_parallelism_method(device_mesh=device_mesh, parallelism_method=parallelism_method)
    coordinate = sub_mesh.get_coordinate()
    if coordinate is None:
        raise ValueError(f"Current rank is not part of the sub-mesh for {parallelism_method}.")
    if len(coordinate) != 1:
        raise ValueError(f"Expected coordinate length 1 for {parallelism_method}, got {len(coordinate)}.")
    return coordinate[0]
