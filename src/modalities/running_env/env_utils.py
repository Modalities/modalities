from enum import Enum

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from pkg_resources import packaging
from torch.distributed.fsdp import MixedPrecision


def has_bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
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

fpThirtytwo = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)


class MixedPrecisionSettings(Enum):
    FP_16 = fpSixteen
    BF_16 = bfSixteen
    BF_16_WORKING = bfSixteen_working
    BF_32 = fpThirtytwo
