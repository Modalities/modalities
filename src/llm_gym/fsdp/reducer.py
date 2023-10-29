from typing import Callable
import torch.distributed as dist
import torch


class Reducer:

    @staticmethod
    def reduce(tensor: torch.Tensor, operation: dist.ReduceOp, callback_fun: Callable[[torch.Tensor], None] = None,
               post_processing_fun: Callable[[torch.Tensor], torch.Tensor] = None):
        dist.all_reduce(tensor, op=operation)
        if post_processing_fun is not None:
            tensor = post_processing_fun(tensor)
        if callback_fun is not None:
            callback_fun(tensor)
        return tensor
