from __future__ import annotations

from enum import Enum
from types import TracebackType
from typing import Any, Callable, Iterable, Tuple, TypeVar

import torch
import torch.distributed as dist

from modalities.evaluation.aggregator import Aggregator
from modalities.util import TimeRecorder


class ThroughputAggregationKeys(Enum):
    NUM_SAMPLES = "NUM_SAMPLES"
    FORWARD_BACKWARD_TIME = "FORWARD_BACKWARD_TIME"


class ThroughputAggregator:
    def __init__(self) -> None:
        self.reset()

    def start(self) -> None:
        self._recorder.start()

    def stop(self, processed_batch_size: int) -> None:
        self._recorder.stop()
        self._num_samples += processed_batch_size

    def reset(self):
        self._recorder = TimeRecorder()
        self._num_samples = 0

    def compute_samples_per_second(self, local_rank: int) -> torch.Tensor:
        aggregator = Aggregator[ThroughputAggregationKeys]()
        num_samples = torch.tensor(self._num_samples).to(torch.device(local_rank))
        recorded_time = torch.tensor(self._recorder.delta_t).to(torch.device(local_rank))
        aggregator.add_values(
            {
                ThroughputAggregationKeys.NUM_SAMPLES: num_samples,
                ThroughputAggregationKeys.FORWARD_BACKWARD_TIME: recorded_time,
            }
        )
        synced_num_samples = aggregator.get_all_reduced_value(
            ThroughputAggregationKeys.NUM_SAMPLES, reduce_operation=dist.ReduceOp.SUM
        )
        synced_foward_backward_time = aggregator.get_all_reduced_value(
            ThroughputAggregationKeys.FORWARD_BACKWARD_TIME, reduce_operation=dist.ReduceOp.MAX
        )
        return synced_num_samples / synced_foward_backward_time


T = TypeVar("T", bound=Any)


def start_throughput_measurement(
    iterable: Iterable[T],
    throughput_aggregatgor_factory: Callable[[], ThroughputAggregator] = ThroughputAggregator,
) -> Iterable[Tuple[ThroughputAggregator, T]]:
    a = throughput_aggregatgor_factory()
    a.start()
    for e in iterable:
        yield a, e
        a.start()


class ThroughputAggregationContext:
    def __init__(
        self,
        num_samples: int,
        local_rank: int,
        throughput_aggregator_factory: Callable[[], ThroughputAggregator] = ThroughputAggregator,
    ) -> None:
        self._local_rank = local_rank
        self._num_samples = num_samples
        self._throughput_aggregator_factory = throughput_aggregator_factory

    @property
    def samples_per_second(self) -> torch.Tensor:
        return self._samples_per_second

    def __enter__(self):
        self._agg = self._throughput_aggregator_factory()
        self._agg.start()
        return self

    def __exit__(
        self,
        type,  # type: ignore
        value: None | BaseException,
        traceback: None | TracebackType,
    ):
        self._agg.stop(self._num_samples)
        self._samples_per_second = self._agg.compute_samples_per_second(self._local_rank)
