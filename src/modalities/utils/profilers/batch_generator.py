from abc import ABC

import torch
from pydantic import BaseModel

from modalities.batch import DatasetBatch
from modalities.config.lookup_enum import LookupEnum


class DatasetBatchGeneratorIF(ABC):
    def get_dataset_batch(self) -> DatasetBatch:
        raise NotImplementedError


class DataTypeEnum(LookupEnum):
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    int64 = torch.int64


class RandomDatasetBatchGeneratorConfig(BaseModel):
    dims: dict[str, int]
    data_type: DataTypeEnum
    min_val: int
    max_val: int


class RandomDatasetBatchGenerator(DatasetBatchGeneratorIF):
    def __init__(self, dims: dict[str, int], data_type: DataTypeEnum, min_val: int, max_val: int):
        self._dims = dims
        self._data_type = data_type
        self._min_val = min_val
        self._max_val = max_val
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def get_dataset_batch(self) -> DatasetBatch:
        size = tuple(self._dims.values())
        if self._data_type == DataTypeEnum.int64:
            inputs = torch.randint(low=self._min_val, high=self._max_val, size=size, device=self._device)
            targets = torch.randint(low=self._min_val, high=self._max_val, size=size, device=self._device)
        elif self._data_type in {DataTypeEnum.float32, DataTypeEnum.bfloat16}:
            inputs = (
                torch.rand(size=size, device=self._device, dtype=self._data_type.value)
                * (self._max_val - self._min_val)
                + self._min_val
            )
            targets = (
                torch.rand(size=size, device=self._device, dtype=self._data_type.value)
                * (self._max_val - self._min_val)
                + self._min_val
            )
        else:
            raise ValueError(f"Unsupported data type: {self._data_type}")

        batch = DatasetBatch(
            samples={
                "input_ids": inputs,
            },
            targets={
                "target_ids": targets,
            },
        )
        return batch
