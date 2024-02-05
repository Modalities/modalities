from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, List

import torch
import torch.distributed as dist

from modalities.batch import InferenceResultBatch
from modalities.evaluation.aggregator import Aggregator, KeyType


class AggregativeMeasureFactory(Generic[KeyType]):

    def create(self, local_rank: int) -> AggregativeMeasure:
        raise NotImplementedError


class AggregativeMeasure(Generic[KeyType], ABC):

    def __init__(
        self,
        aggregate_keys: List[KeyType],
        reduce_ops: Dict[KeyType, dist.ReduceOp.RedOpType],
        tag: str,
        local_rank: int,
    ) -> None:
        self._device = torch.device(local_rank)
        self._aggregator = Aggregator[KeyType](
            initial_values={k: torch.zeros(1).to(self._device) for k in aggregate_keys}
        )
        self._aggregate_keys = aggregate_keys
        self._reduce_ops = reduce_ops
        self._tag = tag

    @property
    def tag(self) -> str:
        return self._tag

    def add(self, batch_result: InferenceResultBatch) -> None:
        res = self._postprocess_result_batch(batch_result)

        for key, value in res.items():
            self._aggregator.add_value(key, value.to(self._device))

    def compute(self) -> torch.Tensor:
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
    def _calc_measure(self, values: Dict[KeyType, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
