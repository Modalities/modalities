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
    expert_parallel_degree: Annotated[int, Field(strict=True, gt=0)] = 1
    enable_loss_parallel: Optional[bool] = False
    world_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def _validate(self):
        for d in (
            self.context_parallel_degree,
            self.expert_parallel_degree,
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
        if self.expert_parallel_degree > 1:
            # Expert parallelism is folded into data parallel sharding:
            # the dp_shard ranks act as data parallel ranks for dense layers and are split into
            # (dp_shard // ep) x ep for the experts.
            if self.data_parallel_shard_degree % self.expert_parallel_degree != 0:
                raise ConfigError(
                    f"expert_parallel_degree({self.expert_parallel_degree}) must divide "
                    f"data_parallel_shard_degree({self.data_parallel_shard_degree})."
                )
            unsupported = {
                "tensor_parallel_degree": self.tensor_parallel_degree,
                "pipeline_parallel_degree": self.pipeline_parallel_degree,
                "context_parallel_degree": self.context_parallel_degree,
            }
            unsupported = {k: v for k, v in unsupported.items() if v > 1}
            if unsupported:
                raise ConfigError(f"Expert parallelism is not yet supported in combination with {unsupported}.")
        return self


class ParallelismDegrees(Enum):
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    CP = "cp"
    EP = "ep"
    EFSDP = "efsdp"
    TP = "tp"
    PP = "pp"


def get_device_mesh(
    device_type: str,
    data_parallel_replicate_degree: int,
    data_parallel_shard_degree: int,
    tensor_parallel_degree: int,
    pipeline_parallel_degree: int,
    context_parallel_degree: int,
    expert_parallel_degree: int,
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
        expert_parallel_degree (int): The expert parallel degree.
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
    if expert_parallel_degree > 1:
        _init_expert_meshes(device_mesh, expert_parallel_degree)
    # TODO: Torch Titan had some more checks here. We need to check if we also need those:
    # https://github.com/pytorch/torchtitan/blob/b291ad662493b63d25b038a30a915082d3617baf/torchtitan/distributed/parallel_dims.py#L86-L104
    return device_mesh


class ExpertMeshes:
    """Device meshes used for the experts of MoE layers when expert parallelism is enabled.

    Expert parallelism is folded into data parallel sharding: dp_shard is split into
    (efsdp = dp_shard // ep, ep), with ep as the innermost dimension so that EP groups
    consist of consecutive ranks (i.e., stay within a node for ep <= GPUs per node).

    Attributes:
        ep (DeviceMesh): 1D mesh over which experts are distributed (Shard(0) on the expert dim).
        expert_fsdp (DeviceMesh): Mesh over which each expert shard is additionally FSDP-sharded
            (1D "efsdp", or 2D ("dp_replicate", "efsdp") for HSDP).
        gradient_divide_factor (int): Number of data parallel ranks contributing tokens to each expert,
            i.e. the factor that expert gradients have to be divided by to match the dense gradients.
    """

    def __init__(self, ep: DeviceMesh, expert_fsdp: DeviceMesh, gradient_divide_factor: int):
        self.ep = ep
        self.expert_fsdp = expert_fsdp
        self.gradient_divide_factor = gradient_divide_factor


_EXPERT_MESHES_ATTR = "_modalities_expert_meshes"


def _init_expert_meshes(device_mesh: DeviceMesh, expert_parallel_degree: int) -> None:
    dp_shard_degree = device_mesh[ParallelismDegrees.DP_SHARD.value].size()
    efsdp_degree = dp_shard_degree // expert_parallel_degree
    # NOTE: pass the dim as an index; torch 2.10's _unflatten rejects valid dim names (inverted membership check).
    sparse_mesh = device_mesh._unflatten(
        device_mesh.mesh_dim_names.index(ParallelismDegrees.DP_SHARD.value),
        (efsdp_degree, expert_parallel_degree),
        (ParallelismDegrees.EFSDP.value, ParallelismDegrees.EP.value),
    )
    if has_parallelism_method(device_mesh, ParallelismDegrees.DP_REPLICATE):
        expert_fsdp = sparse_mesh[(ParallelismDegrees.DP_REPLICATE.value, ParallelismDegrees.EFSDP.value)]
        dp_replicate_degree = device_mesh[ParallelismDegrees.DP_REPLICATE.value].size()
    else:
        expert_fsdp = sparse_mesh[ParallelismDegrees.EFSDP.value]
        dp_replicate_degree = 1
    expert_meshes = ExpertMeshes(
        ep=sparse_mesh[ParallelismDegrees.EP.value],
        expert_fsdp=expert_fsdp,
        gradient_divide_factor=dp_replicate_degree * dp_shard_degree,
    )
    setattr(device_mesh, _EXPERT_MESHES_ATTR, expert_meshes)
    logger.info(f"{sparse_mesh=} | {expert_parallel_degree=} | {efsdp_degree=}")


def get_expert_meshes(device_mesh: DeviceMesh) -> ExpertMeshes | None:
    """Returns the expert meshes if expert parallelism is enabled for the given device mesh, else None."""
    return getattr(device_mesh, _EXPERT_MESHES_ATTR, None)


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
