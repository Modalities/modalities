import os

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

# TODO find a solution for github actions
# to install this as a dependency
# from pkg_resources import packaging
from torch.distributed.fsdp import MixedPrecision

from modalities.config.lookup_enum import LookupEnum


def is_running_with_torchrun():
    # Check for one of the environment variables set by torchrun
    return "LOCAL_RANK" in os.environ


def has_bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        # TODO find a solution for github actions
        # to install this as a dependency
        # and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )


# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

bfSixteen_working = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

megatron_strategy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    # buffer_dtype=torch.bfloat16,
)

fpThirtytwo = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

no_mixed_precision = None


class MixedPrecisionSettings(LookupEnum):
    FP_16 = fpSixteen
    BF_16 = bfSixteen
    BF_16_WORKING = bfSixteen_working
    FP_32 = fpThirtytwo
    MIXED_PRECISION_MEGATRON = megatron_strategy
    NO_MIXED_PRECISION = no_mixed_precision
