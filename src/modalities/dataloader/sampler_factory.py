from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field
from torch.distributed.device_mesh import DeviceMesh

from modalities.config.pydantic_if_types import PydanticDatasetIFType, PydanticDeviceMeshIFType
from modalities.dataloader.dataset import Dataset
from modalities.dataloader.samplers import ResumableDistributedSampler


class ResumableDistributedMultiDimSamplerConfig(BaseModel):
    dataset: PydanticDatasetIFType
    device_mesh: PydanticDeviceMeshIFType
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
        epoch: Optional[int] = 0,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        skip_num_global_samples: Optional[int] = 0,
    ) -> ResumableDistributedSampler:
        rank = device_mesh.get_coordinate()[0]  # TODO make generic for multiple dimensions
        num_replicas = device_mesh.size(0)  # TODO make generic for multiple dimensions

        sampler = ResumableDistributedSampler(
            dataset=dataset,
            rank=rank,
            num_replicas=num_replicas,
            epoch=epoch,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            skip_num_global_samples=skip_num_global_samples,
        )
        return sampler
