from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.models.gpt2.gpt2_model import GPT2LLM, ActivationType, PositionTypes, WeightInitializationConfig
from modalities.nn.attention import AttentionConfig
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
    def get_gpt2_model(
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        weight_init: WeightInitializationConfig,
        attention_config: AttentionConfig,
        attention_norm: nn.Module,
        ffn_norm: nn.Module,
        lm_head_norm: nn.Module,
    ):
        model = GPT2LLM(
            sample_key=sample_key,
            prediction_key=prediction_key,
            poe_type=poe_type,
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            bias=bias,
            activation_type=activation_type,
            weight_init=weight_init,
            attention_config=attention_config,
            attention_norm=attention_norm,
            ffn_norm=ffn_norm,
            lm_head_norm=lm_head_norm,
        )
        return model
