from __future__ import annotations

from typing import Dict, Generic, Hashable, TypeVar

import torch
import torch.distributed as dist

from modalities.running_env.fsdp.reducer import Reducer

KeyType = TypeVar("KeyType", bound=Hashable)


class Aggregator(Generic[KeyType]):

    def __init__(self, initial_values: Dict[KeyType, torch.Tensor] = {}) -> None:
        self._key_to_value = initial_values

    def add_values(self, value_dict: Dict[KeyType, torch.Tensor]):
        for key, value in value_dict.items():
            self.add_value(key, value)

    def add_value(self, key: KeyType, value: torch.Tensor):
        if key not in self._key_to_value:
            self._key_to_value[key] = value
        else:
            self._key_to_value[key] += value

    def get_all_reduced_value(
        self, key: KeyType, reduce_operation: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        # we clone the value so that we can always resync the value without side-effects
        cloned_value = self._key_to_value[key].clone()
        value = Reducer.reduce(tensor=cloned_value, operation=reduce_operation)
        return value
