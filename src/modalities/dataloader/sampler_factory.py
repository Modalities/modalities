from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from torch.distributed.device_mesh import DeviceMesh

from modalities.config.pydantic_if_types import PydanticDatasetIFType, PydanticDeviceMeshIFType
from modalities.dataloader.dataset import Dataset
from modalities.dataloader.samplers import ResumableDistributedSampler
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees


class ResumableDistributedMultiDimSamplerConfig(BaseModel):
    dataset: PydanticDatasetIFType
    device_mesh: PydanticDeviceMeshIFType
    data_parallel_key: ParallelismDegrees
    epoch: Annotated[int, Field(strict=True, ge=0)] = 0
    shuffle: Optional[bool] = False
    seed: Optional[int] = 0
    drop_last: Literal[True] = True
    skip_num_global_samples: Annotated[int, Field(strict=True, ge=0)] = 0


class SamplerFactory:
    """
    Factory class for creating samplers.
    """

    @staticmethod
    def create_resumable_distributed_multi_dim_sampler(
        dataset: Dataset,
        device_mesh: DeviceMesh,
        data_parallel_key: ParallelismDegrees,
        epoch: Optional[int] = 0,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        skip_num_global_samples: Optional[int] = 0,
    ) -> ResumableDistributedSampler:
        dp_rank = device_mesh[data_parallel_key.value].get_coordinate()[0]
        num_replicas = device_mesh[data_parallel_key.value].size()

        sampler = ResumableDistributedSampler(
            dataset=dataset,
            rank=dp_rank,
            num_replicas=num_replicas,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        return sampler
