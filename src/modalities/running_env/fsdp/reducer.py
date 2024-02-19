from typing import Callable

import torch
import torch.distributed as dist


class Reducer:
    @staticmethod
    def reduce(
        tensor: torch.Tensor,
        operation: dist.ReduceOp.RedOpType,
        post_processing_fun: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        dist.all_reduce(tensor, op=operation)
        if post_processing_fun is not None:
            tensor = post_processing_fun(tensor)
        return tensor
