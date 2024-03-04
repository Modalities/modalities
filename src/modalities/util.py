import time
import warnings
from datetime import datetime
from enum import Enum
from types import TracebackType
from typing import Callable, Dict, Generic, Type, TypeVar

import torch
import torch.distributed as dist
from pydantic import ValidationError

from modalities.exceptions import TimeRecorderStateError
from modalities.running_env.fsdp.reducer import Reducer


def get_callback_interval_in_batches_per_rank(
    callback_interval_in_samples: int, local_train_micro_batch_size: int, world_size: int, gradient_acc_steps: int
):
    num_local_train_micro_batches_exact = callback_interval_in_samples / local_train_micro_batch_size / world_size
    num_local_train_micro_batches_ret = max(
        callback_interval_in_samples // local_train_micro_batch_size // world_size, 1
    )
    if num_local_train_micro_batches_exact != num_local_train_micro_batches_ret:
        warnings.warn(
            f"Calculated callback_interval_in_batches_per_rank is not an integer."
            f"Clipping {num_local_train_micro_batches_exact} to {num_local_train_micro_batches_ret} "
        )
    assert (
        num_local_train_micro_batches_ret % gradient_acc_steps == 0
    ), "callback_interval_in_batches_per_rank must be divisible by gradient_acc_steps"
    return num_local_train_micro_batches_ret


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

    def is_running(self) -> bool:
        return self._state == TimeRecorderStates.RUNNING

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
