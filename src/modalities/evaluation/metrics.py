from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar

import torch
import torch.distributed as dist
from torch._tensor import Tensor

from modalities.batch import InferenceResultBatch
from modalities.running_env.fsdp.reducer import Reducer

# losses
# 1) loss_fun(result_batch) -> sum(tensors)/len(batch) -> loss per sample
# 2) sum over all batches / #batches
# 3) sum over ranks / # ranks
# -> sum(all_loss_values) / (batch_size * #batches * #ranks)

# metrics
# 1) metric(batch) -> float
# 2) sum over

T = TypeVar("T")


class StatefulMetricFactory:
    pass


class Aggregator(Generic[T]):
    def __init__(self):
        self.key_to_value: Dict[T, torch.Tensor] = {}

    def add_values(self, value_dict: Dict[T, torch.Tensor]):
        for key, value in value_dict.items():
            self.add_value(key, value)

    def add_value(self, key: T, value: torch.Tensor):
        if key not in self.key_to_value:
            self.key_to_value[key] = value
        else:
            self.key_to_value[key] += value

    # FIXME: Remove as we can always just instantiate a new aggregator
    def remove_keys(self):
        self.key_to_value = {}

    def remove_key(self, key: T):
        self.key_to_value.pop(key)

    def get_all_reduced_value(
        self, key: T, reduce_operation: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        # we clone the value so that we can always resync the value without side-effects
        cloned_value = self.key_to_value[key].clone()
        value = Reducer.reduce(tensor=cloned_value, operation=reduce_operation)
        return value


class StatefulMeasureIF(Generic[T], ABC):
    @abstractmethod
    def add_result_batch(self, result_batch: InferenceResultBatch):
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
    ) -> torch.Tensor:
        raise NotImplementedError


class StatefulMeasureABC(StatefulMeasureIF):
    def __init__(
        self, aggregate_keys_and_init: Dict[str, torch.Tensor], reduce_ops: Dict[T, dist.ReduceOp.RedOpType]
    ) -> None:
        self._aggregator = Aggregator[T](initial_key_values=aggregate_keys_and_init)

    def add_result(self, batch_result: InferenceResultBatch) -> None:
        res = self._postprocess_result_batch(batch_result)

        for key, value in res.items():
            self._aggregator.add_value(key, value)

    def compute(self) -> torch.Tensor:
        synced_vals = {}
        for key in self._aggregator:
            synced_vals[key] = self._aggregator.get_all_reduced_value(
                key,
                self.reduce_ops[key],
            )

        return self._calc_measure(synced_vals)

    @abstractmethod
    def _calc_measure(self) -> float:
        pass

    @abstractmethod
    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[T, torch.Tensor]:
        raise NotImplementedError


class PerplexityStatefulMeasure(StatefulMeasureABC):
    def __init__(self, aggregate_keys_and_init: Dict[str, Tensor], reduce_op) -> None:
        super().__init__(aggregate_keys_and_init, reduce_op)

    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[T, torch.Tensor]:
        pass

    def calc_measure() -> float:
        pass
