import torch
from functools import partial

from torch.distributed.fsdp import FSDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   checkpoint_wrapper,
   CheckpointImpl,
   apply_activation_checkpointing_wrapper,
)

from llm_gym.models.gpt2.gpt2_model import Block


def apply_activation_checkpointing_inplace(model: FSDP) -> None:
    assert isinstance(model, FSDP), f"activation checkpointing can only be applied to FSDP wrapped models!"
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, Block)

    return apply_activation_checkpointing_wrapper(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
