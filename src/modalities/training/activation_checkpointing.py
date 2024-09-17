from functools import partial
from typing import List

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from modalities.util import get_module_class_from_name


def is_module_to_apply_activation_checkpointing(
    submodule: torch.nn.Module, activation_checkpointing_modules: List[type]
) -> bool:
    return isinstance(submodule, tuple(activation_checkpointing_modules))


def apply_activation_checkpointing_inplace(model: torch.nn.Module, activation_checkpointing_modules: List[str]):
    activation_checkpointing_module_types = [
        get_module_class_from_name(model, m) for m in activation_checkpointing_modules
    ]
    if not isinstance(model, FSDP):
        raise ValueError("activation checkpointing can only be applied to FSDP wrapped models!")
    non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT, debug=False)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: is_module_to_apply_activation_checkpointing(
            submodule, activation_checkpointing_module_types
        ),
    )
