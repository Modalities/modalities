from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.running_env.env_utils import MixedPrecisionSettings
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory
from modalities.training.activation_checkpointing import apply_activation_checkpointing_inplace
from modalities.util import get_local_number_of_trainable_parameters


class ModelFactory:
    """Model factory class to create models."""

    @staticmethod
    def get_checkpointed_model(
        checkpoint_loading: CheckpointLoadingIF,
        checkpoint_path: Path,
        model: nn.Module,
    ) -> nn.Module:
        """
        Loads a checkpointed model from the given checkpoint path.

        Args:
            checkpoint_loading (CheckpointLoadingIF): The checkpoint loading approach used to load the model checkpoint.
            checkpoint_path (Path): The path to the checkpoint file.
            model (nn.Module): The model to be loaded with the checkpoint.

        Returns:
            nn.Module: The loaded wrapped model.

        """
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
        """
        Get the FSDP-wrapped model.

        Args:
            model (nn.Module): The original model to be wrapped.
            sync_module_states (bool): Whether to synchronize module states across ranks.
            block_names (List[str]): List of block names.
            mixed_precision_settings (MixedPrecisionSettings): Mixed precision settings.
            sharding_strategy (ShardingStrategy): Sharding strategy.

        Returns:
            FSDP: The FSDP-wrapped model.

        Note:
            'FSDPTransformerAutoWrapPolicyFactory` is hardcoded and should be passed in instead.
            Different auto wrap policies may be supported in the future.
        """
        print(
            f"Unsharded number of parameters on rank {dist.get_rank()}: "
            f"{get_local_number_of_trainable_parameters(model)}"
        )
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
            use_orig_params=True,
        )
        print(
            f"Sharded number of parameters on rank {dist.get_rank()}:"
            f"{get_local_number_of_trainable_parameters(fsdp_model)}"
        )

        return fsdp_model

    @staticmethod
    def get_weight_initalized_model(model: nn.Module, model_initializer: ModelInitializationIF) -> nn.Module:
        """
        Initializes the given model with weights using the provided model initializer.

        Args:
            model (nn.Module): The model to be initialized with weights.
            model_initializer (ModelInitializationIF): The model initializer object.

        Returns:
            nn.Module: The initialized model.
        """
        model_initializer.initialize_in_place(model)
        return model

    @staticmethod
    def get_activation_checkpointed_model(model: FSDP, activation_checkpointing_modules: List[str]) -> FSDP:
        """Apply activation checkpointing to the given model (in-place operation).

        Args:
            model (FSDP): The FSDP-wrapped model to apply activation checkpointing to.
            activation_checkpointing_modules (List[str]): List of module names to apply activation checkpointing to.

        Raises:
            ValueError: Activation checkpointing can only be applied to FSDP-wrapped models!

        Returns:
            FSDP: The model with activation checkpointing applied.
        """
        if len(activation_checkpointing_modules) > 0:
            if isinstance(model, FSDP):
                apply_activation_checkpointing_inplace(
                    model=model,
                    activation_checkpointing_modules=activation_checkpointing_modules,
                )
            else:
                raise ValueError(
                    "Activation checkpointing can only be applied to FSDP-wrapped models! "
                    f"Current model type: {type(model)}"
                )
        return model
