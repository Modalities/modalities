import time
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import Callable, Dict, Generic, Type, TypeVar

import torch
import torch.distributed as dist
from pydantic import ValidationError

from modalities.exceptions import TimeRecorderStateError
from modalities.running_env.fsdp.reducer import Reducer


def parse_enum_by_name(name: str, enum_type: Type[Enum]) -> Enum:
    try:
        val = enum_type[name]
        return val
    except KeyError:
        raise ValidationError(f"Invalid {enum_type} member name: {name}")


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07__14-31-22'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    return date_of_run


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def compute_number_of_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TimeRecorderStates(Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class TimeRecorder:
    """Class with context manager to record execution time"""

    def __init__(self):
        self.delta_t: float = 0
        self.time_s: float = -1
        self._state: TimeRecorderStates = TimeRecorderStates.STOPPED

    def start(self):
        if self._state == TimeRecorderStates.RUNNING:
            raise TimeRecorderStateError("Cannot start a running TimeRecorder.")
        self.time_s = time.perf_counter()
        self._state = TimeRecorderStates.RUNNING

    def stop(self):
        if self._state == TimeRecorderStates.STOPPED:
            raise TimeRecorderStateError("Cannot stop an already stopped TimeRecorder.")
        self.delta_t += time.perf_counter() - self.time_s
        self._state = TimeRecorderStates.STOPPED

    def reset(self):
        if self._state == TimeRecorderStates.RUNNING:
            raise TimeRecorderStateError("Can only reset a stopped TimeRecorder.")
        self.delta_t = 0
        self.time_s = -1

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        type,  # type: ignore
        value: None | BaseException,
        traceback: None | TracebackType,
    ):
        self.stop()

    def __repr__(self) -> str:
        return f"{self.delta_t}s"


T = TypeVar("T")


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
