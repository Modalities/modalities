from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from modalities.models.gpt2.gpt2_model import GPT2Block


def is_module_to_apply_activation_checkpointing(submodule: torch.nn.Module):
    return isinstance(submodule, GPT2Block)


def apply_activation_checkpointing_inplace(model: torch.nn.Module) -> None:
    assert isinstance(model, FSDP), "activation checkpointing can only be applied to FSDP wrapped models!"
    non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT, debug=True)

    return apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=is_module_to_apply_activation_checkpointing
    )
