from __future__ import annotations

from typing import Dict, Generic, Hashable, TypeVar

import torch
import torch.distributed as dist

from modalities.running_env.fsdp.reducer import Reducer

KeyType = TypeVar("KeyType", bound=Hashable)


class Aggregator(Generic[KeyType]):
    def __init__(self):
        self.key_to_value: Dict[KeyType, torch.Tensor] = {}

    def add_values(self, value_dict: Dict[KeyType, torch.Tensor]):
        for key, value in value_dict.items():
            self.add_value(key, value)

    def add_value(self, key: KeyType, value: torch.Tensor):
        if key not in self.key_to_value:
            self.key_to_value[key] = value
        else:
            self.key_to_value[key] += value

    # # FIXME: Remove as we can always just instantiate a new aggregator
    # def remove_keys(self):
    #     self.key_to_value = {}

    # def remove_key(self, key: KeyType):
    #     self.key_to_value.pop(key)

    def get_all_reduced_value(
        self, key: KeyType, reduce_operation: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        # we clone the value so that we can always resync the value without side-effects
        cloned_value = self.key_to_value[key].clone()
        value = Reducer.reduce(tensor=cloned_value, operation=reduce_operation)
        return value
