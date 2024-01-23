from enum import Enum
from typing import Callable, Dict, Generic, TypeVar

import torch
import torch.distributed as dist

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

    def add_value(self, key: T, value: torch.Tensor):
        if key not in self.key_to_value:
            self.key_to_value[key] = value
        else:
            self.key_to_value[key] += value

    def remove_key(self, key: T):
        self.key_to_value.pop(key)

    def remove_keys(self):
        self.key_to_value = {}

    def get_all_reduced_value(
        self,
        key: T,
        reduce_operation: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM,
        postprocessing_fun: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> torch.Tensor:
        # we clone the value so that we can always resync the value without side-effects
        cloned_value = self.key_to_value[key].clone()
        value = Reducer.reduce(
            tensor=cloned_value,
            operation=reduce_operation,
            post_processing_fun=postprocessing_fun,  # lambda t: t[0] / t[1],
        )
        return value


class StatefulMetric(Generic[T]):
    def __init__(self, tag: str, aggregator: Aggregator[T]):
        self.tag = tag
        self.aggregator = aggregator

    def add_result_batch(self, result_batch: InferenceResultBatch):
        pass

    def compute(
        self,
    ) -> torch.Tensor:
        pass


class AccuracyStatefulMetric:
    class Keys(Enum):
        tp = "TP"
        fp = "FP"

    def __init__(self, prediction_subscription_key: str, target_subscription_key: str) -> None:
        # Non-packed validation target with generated samples of different lengths
        # t_0 | t1  t2  t3  t4 t5  t6  EoS 0   0   0   |  <- target
        # t_0 | t1  t2  t3  t4 t5  t6  EoS             |  <- sample 1
        # t_0 | t1  t2  t3  t4 t5  t6  t7  t8  t9  t10 |  <- sample 2
        # t_0 | t1  t2  t3  t4 t5  t6  t7  t8  EoS     |  <- sample 3
        # t_0 | t1  t2  t3  t4 EoS t6  t7  t8  t9  t10 |  <- sample 4
        #
        # Sample 3: we calculate the accuracy metric until t7 and disregard t8 - t10
        # Sample 4: we would calculate the accuracy by taking the misses (t6 and EoS) into account

        self.prediction_subscription_key = prediction_subscription_key
        self.target_subscription_key = target_subscription_key
        self.aggregator: Aggregator = Aggregator[AccuracyStatefulMetric.Keys]()

    def add_result_batch(self, result_batch: InferenceResultBatch):
        # logic for what we want track
        pass

    def compute(
        self,
    ) -> torch.Tensor:
        # logic for how we compute the metric from what we wanted to track
        # We call get_all_reduced_value from Aggregator to sync the values
        # across the ranks

        pass
