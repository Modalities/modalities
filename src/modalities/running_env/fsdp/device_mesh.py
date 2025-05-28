from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from modalities.exceptions import ConfigError
from modalities.util import print_rank_0
from modalities.utils.logging import get_logger

logger = get_logger("model_factory")


class DeviceMeshConfig(BaseModel):
    # inspired by ParallelDims class in
    # https://github.com/pytorch/torchtitan/blob/cfc0f4e08dc71685cdcb394464187d2eeedd1a5f/torchtitan/parallelisms/parallel_dims.py#L15
    device_type: str = "cuda"
    data_parallel_replicate_degree: Annotated[int, Field(strict=True, gt=0)]
    # if -1, we will calculate the shard degree based on the world size and other parallel degrees
    data_parallel_shard_degree: Annotated[int, Field(strict=True, ge=-1)]
    tensor_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    pipeline_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    context_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    enable_loss_parallel: Optional[bool] = False
    world_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def _validate(self):
        for d in (
            self.data_parallel_replicate_degree,
            self.context_parallel_degree,
            self.tensor_parallel_degree,
            self.pipeline_parallel_degree,
        ):
            if d < 1:
                raise ConfigError("Parallelism degree must be >= 1, except for data_parallel_shard_degree")

        if not (self.data_parallel_shard_degree == -1 or self.data_parallel_shard_degree >= 1):
            raise ConfigError("data_parallel_shard_degree must be -1 or >= 1")

        if self.data_parallel_shard_degree == -1:
            # set the shard degree to the world size divided by the product of all other parallel degrees
            self.data_parallel_shard_degree = self.world_size // (
                self.data_parallel_replicate_degree
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
    """Gets the device mesh for the specified parallelism degrees.

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
        if dim > 1:
            dims.append(dim)
            names.append(name)
    names = tuple(names)
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
    print_rank_0(f"{device_mesh=} | {world_size=} | {enable_loss_parallel=}")
    # TODO: Torch Titan had some more checks here. We need to check if we also need those:
    # https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/distributed/parallel_dims.py#L86-L104
    return device_mesh
