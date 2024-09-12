import hashlib
import time
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Callable, Dict, Generic, Optional, Type, TypeVar

import torch
import torch.distributed as dist
from pydantic import ValidationError
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.types import Number

from modalities.exceptions import TimeRecorderStateError
from modalities.running_env.fsdp.reducer import Reducer


def print_rank_0(message: str):
    """If torch.distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def warn_rank_0(message: str):
    """If torch.distributed is initialized, print only on rank 0."""
    message_with_color_code = f"\033[91m {message} \033[00m"
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            warnings.warn(message_with_color_code)
    else:
        warnings.warn(message_with_color_code)


def parse_enum_by_name(name: str, enum_type: Type[Enum]) -> Enum:
    try:
        return enum_type[name]
    except KeyError:
        raise ValidationError(f"Invalid {enum_type} member name: {name}")


def get_callback_interval_in_batches_per_rank(
    local_callback_interval_in_samples: int, local_train_micro_batch_size: int, gradient_acc_steps: int
):
    num_local_train_micro_batches_exact = local_callback_interval_in_samples / local_train_micro_batch_size
    num_local_train_micro_batches_ret = max(local_callback_interval_in_samples // local_train_micro_batch_size, 1)
    if num_local_train_micro_batches_exact != num_local_train_micro_batches_ret:
        warnings.warn(
            f"Calculated callback_interval_in_batches_per_rank is not an integer."
            f"Clipping {num_local_train_micro_batches_exact} to {num_local_train_micro_batches_ret} "
        )
    assert (
        num_local_train_micro_batches_ret % gradient_acc_steps == 0
    ), "callback_interval_in_batches_per_rank must be divisible by gradient_acc_steps"
    return num_local_train_micro_batches_ret


def get_experiment_id_of_run(config_file_path: Path, hash_length: Optional[int] = 8) -> str:
    """create experiment ID including the date and time for file save uniqueness
    example: 2022-05-07__14-31-22_fdh1xaj2'
    """
    hash = hashlib.sha256(str(config_file_path).encode()).hexdigest()[:hash_length]
    date_of_run = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    experiment_id = f"{date_of_run}_{hash}"
    return experiment_id


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def get_local_number_of_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_number_of_trainable_parameters(model: FSDP) -> Number:
    num_params = get_local_number_of_trainable_parameters(model)
    num_params_tensor = torch.tensor(num_params).cuda()
    dist.all_reduce(num_params_tensor, op=dist.ReduceOp.SUM)
    total_num_params = num_params_tensor.item()
    # For HYBRID sharding, divide by sharding factor to get the correct number of parameters
    # TODO: Define constant instead of hardcoding string
    if model.sharding_strategy.name == "HYBRID_SHARD":
        # Assumes that CUDA is available and each node has the same number of GPUs
        # Note: Per default FSDP constructs process groups for the user to shard intra-node and replicate inter-node.
        # However, users can also provide their own sharding process groups (currently not supported in Modalities)
        # which would require to adapt the code.
        sharding_factor_hybrid_sharding = dist.get_world_size() // torch.cuda.device_count()
        total_num_params = total_num_params // sharding_factor_hybrid_sharding

    return total_num_params


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


def get_module_class_from_name(module: torch.nn.Module, name: str) -> Type[torch.nn.Module] | None:
    """From Accelerate source code
    (https://github.com/huggingface/accelerate/blob/1f7a79b428749f45187ec69485f2c966fe21926e/src/accelerate/utils/dataclasses.py#L1902)
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
