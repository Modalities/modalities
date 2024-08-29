from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.config.config import PrecisionEnum
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory


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
        # Here, FSDPTransformerAutoWrapPolicyFactory is hardcoded and should be passed in instead!
        # we also might want to have different auto wrap policies later...
        fsdp_auto_wrap_factory = FSDPTransformerAutoWrapPolicyFactory(model=model, block_names=block_names)

        # model is on CPU before input to FSDP
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=fsdp_auto_wrap_factory.get_auto_wrap_policy(),
            mixed_precision=mixed_precision_settings.value,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
        )
        return fsdp_model

    @staticmethod
    def get_ddp_wrapped_model(
        model: nn.Module,
        local_rank: int,
        precision: Optional[PrecisionEnum] = None,
    ) -> DDP:
        device_id = local_rank
        if precision is not None:
            model = model.to(device_id, dtype=precision.value)
        else:
            model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id], output_device=device_id)

        return ddp_model

    @staticmethod
    def get_torch_model(model: nn.Module, device: torch.device, precision: Optional[PrecisionEnum] = None) -> nn.Module:
        if precision is not None:
            model = model.to(device, dtype=precision.value)
        else:
            model = model.to(device)
        return model
