from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.util import compute_number_of_trainable_parameters


class ModelFactory:
    @staticmethod
    def get_checkpointed_model(
        checkpoint_loading: CheckpointLoadingIF, checkpoint_path: Path, model: nn.Module
    ) -> nn.Module:
        wrapped_model = checkpoint_loading.load_model_checkpoint(
            file_path=checkpoint_path,
            model=model,
        )
        return wrapped_model

    @staticmethod
    def get_fsdp_wrapped_model(
        model: nn.Module,
        sync_module_states: bool,
        block_names: List[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ) -> FSDP:
        print(
            f"Unsharded number of parameters on rank {dist.get_rank()}: {compute_number_of_trainable_parameters(model)}"
        )
        # Here, FSDPTransformerAutoWrapPolicyFactory is hardcoded and should be passed in instead!
        # we also might want to have different auto wrap policies later...
        # fsdp_auto_wrap_factory = FSDPTransformerAutoWrapPolicyFactory(model=model, block_names=block_names)

        # model is on CPU before input to FSDP
        # FSDP(
        #     model,
        #     auto_wrap_policy=fsdp_auto_wrap_factory.get_auto_wrap_policy(),
        #     mixed_precision=mixed_precision_settings.value,
        #     sharding_strategy=sharding_strategy,
        #     device_id=torch.cuda.current_device(),
        #     sync_module_states=sync_module_states,
        #     use_orig_params=True,
        # )
        device_id = dist.get_rank() % torch.cuda.device_count()
        model = model.to(device_id).bfloat16()
        ddp_model = DDP(model, device_ids=[device_id], output_device=device_id)
        print(
            f"Sharded number of parameters on rank {dist.get_rank()}:"
            f"{compute_number_of_trainable_parameters(ddp_model)}"
        )

        return ddp_model
