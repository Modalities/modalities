from llm_gym.env_utils import has_bfloat_support, bfSixteen
from llm_gym.gpt2.gpt2_model import NNModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
import torch.distributed as dist
import torch


class FSDPRunner:
    def __init__(self) -> None:
        dist.init_process_group("nccl")

    def run():

        dist.destroy_process_group()

    @staticmethod
    def wrap_fsdp_model(model: NNModel, local_rank: int) -> FSDP:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
        torch.cuda.set_device(local_rank)

        if has_bfloat_support():
            mp_policy = bfSixteen
        else:
            mp_policy = None  # defaults to fp32

        # model is on CPU before input to FSDP
        model = FSDP(model,
                     auto_wrap_policy=None,
                     mixed_precision=mp_policy,
                     sharding_strategy=sharding_strategy,
                     device_id=torch.cuda.current_device())
        return model
