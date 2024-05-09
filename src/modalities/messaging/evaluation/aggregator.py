from enum import Enum
from typing import Dict, Generic, List, TypeVar

import torch
import torch.distributed as dist

from modalities.running_env.fsdp.reducer import Reducer

T = TypeVar("T", Enum)


class DistributedAggregator(Generic[T]):
    def __init__(self):
        self.key_to_value: Dict[T, float] = {}

    def add_value(self, key: T, value: float | int):
        if key not in self.key_to_value:
            self.key_to_value[key] = 0

        self.key_to_value[key] += value

    def remove_keys(self):
        self.key_to_value = {}

    def get_all_reduced_values(
        self,
        keys: List[T],
        reduce_operation: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
    ) -> Dict[T, torch.Tensor]:
        # we clone the value so that we can always resync the value without side-effects
        cloned_value = torch.FloatTensor([self.key_to_value[key] for key in keys]).cuda()
        value = Reducer.reduce(
            tensor=cloned_value,
            operation=reduce_operation,
        )
        reduced_dict = {key: value[i] for i, key in enumerate(keys)}
        return reduced_dict
