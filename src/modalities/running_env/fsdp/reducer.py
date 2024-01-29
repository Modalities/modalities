import torch
import torch.distributed as dist


class Reducer:
    @staticmethod
    def reduce(
        tensor: torch.Tensor,
        operation: dist.ReduceOp.RedOpType,
    ):
        dist.all_reduce(tensor, op=operation)
        return tensor
