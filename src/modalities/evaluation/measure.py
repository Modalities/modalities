from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, List

import torch
import torch.distributed as dist

from modalities.batch import InferenceResultBatch
from modalities.evaluation.aggregator import Aggregator, KeyType

# losses
# 1) loss_fun(result_batch) -> sum(tensors)/len(batch) -> loss per sample
# 2) sum over all batches / #batches
# 3) sum over ranks / # ranks
# -> sum(all_loss_values) / (batch_size * #batches * #ranks)

# metrics
# 1) metric(batch) -> float
# 2) sum over


class AggregativeMeasureFactory(Generic[KeyType]):
    def create(self) -> AggregativeMeasure:
        raise NotImplementedError


class AggregativeMeasure(Generic[KeyType], ABC):
    def __init__(self, aggregate_keys: List[KeyType], reduce_ops: Dict[KeyType, dist.ReduceOp.RedOpType]) -> None:
        self._aggregator = Aggregator[KeyType]()
        self._aggregate_keys = aggregate_keys
        self._reduce_ops = reduce_ops

    def add(self, batch_result: InferenceResultBatch) -> None:
        res = self._postprocess_result_batch(batch_result)

        for key, value in res.items():
            self._aggregator.add_value(key, value)

    def compute(self) -> float:
        synced_vals: Dict[KeyType, torch.Tensor] = {}
        for key in self._aggregate_keys:
            synced_vals[key] = self._aggregator.get_all_reduced_value(
                key,
                self._reduce_ops[key],
            )

        return self._calc_measure(synced_vals)

    @abstractmethod
    def _postprocess_result_batch(self, batch_result: InferenceResultBatch) -> Dict[KeyType, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def _calc_measure(self, values: Dict[KeyType, torch.Tensor]) -> float:
        raise NotImplementedError
