from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from modalities.exceptions import ConfigError
from modalities.util import print_rank_0


class DeviceMeshConfig(BaseModel):
    # inspired by ParallelDims class in
    # https://github.com/pytorch/torchtitan/blob/cfc0f4e08dc71685cdcb394464187d2eeedd1a5f/torchtitan/parallelisms/parallel_dims.py#L15
    device_type: str = "cuda"
    data_parallel_degree: Annotated[int, Field(strict=True, gt=0)]
    tensor_parallel_degree: Annotated[int, Field(strict=True, gt=0)]
    pipeline_parallel_degree: Annotated[int, Field(strict=True, gt=0)]

    # TODO add sequence parallel degree

    enable_loss_parallel: bool = False
    world_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def _validate(self):
        if self.data_parallel_degree == -1:
            self.data_parallel_degree = self.world_size // (self.tensor_parallel_degree * self.pipeline_parallel_degree)
        if self.data_parallel_degree * self.tensor_parallel_degree * self.pipeline_parallel_degree != self.world_size:
            raise ConfigError(
                f"Invalid parallel dims: dp({self.data_parallel_degree}) * tp({self.tensor_parallel_degree}) "
                "* pp({self.pipeline_parallel_degree}) != WORLD_SIZE({self.world_size})"
            )


def get_device_mesh(
    device_type: str,
    data_parallel_degree: int,
    tensor_parallel_degree: int,
    pipeline_parallel_degree: int,
    enable_loss_parallel: bool,
    world_size: int,
) -> DeviceMesh:
    dims = []
    names = []
    for d, name in zip(
        [pipeline_parallel_degree, data_parallel_degree, tensor_parallel_degree], ["pp", "dp", "tp"], strict=True
    ):
        if d > 1:
            dims.append(d)
            names.append(name)
    names = tuple(names)
    device_mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
    print_rank_0(f"{device_mesh=} | {world_size=} | {enable_loss_parallel=}")
    return device_mesh
