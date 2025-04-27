import itertools
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.fsdp import ShardingStrategy
from typing_extensions import deprecated

from modalities.checkpointing.checkpoint_loading import FSDP1CheckpointLoadingIF
from modalities.config.config import SelectiveActivationCheckpointedModelConfig
from modalities.exceptions import ModelStateError
from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    AttentionConfig,
    AttentionImplementation,
    LayerNormWrapperConfig,
    PositionTypes,
)
from modalities.models.model import ActivationType
from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings, MixedPrecisionSettings
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory
from modalities.training.activation_checkpointing.activation_checkpointing import (
    SelectiveActivationCheckpointing,
    apply_activation_checkpointing_inplace,
)
from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
    SelectiveActivationCheckpointingVariants,
)
from modalities.util import get_local_number_of_trainable_parameters, get_module_class_from_name
from modalities.utils.logger_utils import get_logger

logger = get_logger("model_factory")


class ModelFactory:
    """Model factory class to create models."""

    @staticmethod
    def _is_model_on_meta_device(model: nn.Module) -> bool:
        meta_counter = 0
        param_counter = 0
        for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
            if tensor.device == torch.device("meta"):
                meta_counter += 1
            param_counter += 1

        if meta_counter > 0 and meta_counter < param_counter:
            raise ModelStateError("Either all or none of the parameters and buffers must be on meta device!")
        return meta_counter > 0

    @staticmethod
    def get_fsdp1_checkpointed_model(
        checkpoint_loading: FSDP1CheckpointLoadingIF,
        checkpoint_path: Path,
        model: nn.Module,
    ) -> nn.Module:
        """
        Loads a FSDP1 checkpointed model from the given checkpoint path.

        Args:
            checkpoint_loading (FSDP1CheckpointLoadingIF): The checkpoint loading
                approach used to load the model checkpoint.
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

    @deprecated(
        "With version 0.4, we upgraded FSDP to FSDP 2.0. "
        "Use GeneralModelFactory.get_fsdp2_wrapped_model(...) instead.",
        category=FutureWarning,
    )
    @staticmethod
    def get_fsdp1_wrapped_model(
        model: nn.Module,
        sync_module_states: bool,
        block_names: list[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ) -> FSDP1:
        """
        Get the FSDP1-wrapped model.

        Args:
            model (nn.Module): The original model to be wrapped.
            sync_module_states (bool): Whether to synchronize module states across ranks.
            block_names (list[str]): List of block names.
            mixed_precision_settings (MixedPrecisionSettings): Mixed precision settings.
            sharding_strategy (ShardingStrategy): Sharding strategy.

        Returns:
            FSDP1: The FSDP1-wrapped model.

        Note:
            'FSDPTransformerAutoWrapPolicyFactory` is hardcoded and should be passed in instead.
            Different auto wrap policies may be supported in the future.
        """
        if ModelFactory._is_model_on_meta_device(model=model):
            raise ModelStateError("Meta device initialization is not supported for FSDP1. Use FSDP2 instead.")

        logger.info(
            f"Rank {dist.get_rank()} unsharded number of parameters: "
            f"{get_local_number_of_trainable_parameters(model)}"
        )
        # Here, FSDPTransformerAutoWrapPolicyFactory is hardcoded and should be passed in instead!
        # we also might want to have different auto wrap policies later...
        fsdp_auto_wrap_factory = FSDPTransformerAutoWrapPolicyFactory(model=model, block_names=block_names)

        # model is on CPU before input to FSDP1
        fsdp_model = FSDP1(
            model,
            auto_wrap_policy=fsdp_auto_wrap_factory.get_auto_wrap_policy(),
            mixed_precision=mixed_precision_settings.value,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            use_orig_params=True,
        )
        logger.info(
            f"Rank {dist.get_rank()} sharded number of parameters: "
            f"{get_local_number_of_trainable_parameters(fsdp_model)}"
        )
        return fsdp_model

    @staticmethod
    def get_fsdp2_wrapped_model(
        model: nn.Module,
        block_names: list[str],
        device_mesh: DeviceMesh,
        mixed_precision_settings: FSDP2MixedPrecisionSettings,
        reshard_after_forward: bool,
    ) -> FSDP2:
        """Get the FSDP2-wrapped model.

        Based on https://github.com/pytorch/torchtitan/blob/de9fd2b9ea7e763c9182e0df81fc32c2618cc0b6/torchtitan/parallelisms/parallelize_llama.py#L459
        and https://github.com/pytorch/torchtitan/blob/43584e0a4e72645e25cccd05d86f9632587a8beb/docs/fsdp.md
        NOTE: Torch Titan already implement pipeline parallelism. We skip that here for now.

        Args:
            model (nn.Module): The original model to be wrapped.
            block_names (list[str]): List of block names.
            device_mesh (DeviceMesh): The device mesh.
            mixed_precision_settings (FSDP2MixedPrecisionSettings): Mixed precision settings.
            reshard_after_forward (bool): Whether to reshard after forward.

        Returns:
            FSDP2: The FSDP2-wrapped model.
        """
        logger.info(
            f"Rank {dist.get_rank()} unsharded number of parameters (possibly on meta device): "
            f"{get_local_number_of_trainable_parameters(model)}"
        )
        # map the block names to the actual block class (e.b., GPT2Block)
        block_types = tuple([get_module_class_from_name(model, b) for b in block_names])

        mp_policy = MixedPrecisionPolicy(
            param_dtype=mixed_precision_settings.param_dtype.value,
            reduce_dtype=mixed_precision_settings.reduce_dtype.value,
        )

        fsdp_config = {"mesh": device_mesh[ParallelismDegrees.DP_SHARD.value], "mp_policy": mp_policy}

        modules = list(model.modules())
        # we first shard all the blocks
        for module_id, module in enumerate(modules):
            if isinstance(module, block_types):
                # As an optimization, we do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately.
                reshard_block_after_forward = reshard_after_forward and int(module_id) < len(modules) - 1
                fully_shard(
                    module,
                    **fsdp_config,
                    reshard_after_forward=reshard_block_after_forward,
                )
        # finally, we shard the entire model
        fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)
        logger.info(
            f"Rank {dist.get_rank()} sharded number of parameters: "
            f"{get_local_number_of_trainable_parameters(model)}"
        )
        return model

    @staticmethod
    def get_weight_initalized_model(model: nn.Module, model_initializer: ModelInitializationIF) -> nn.Module:
        """
        Initializes the given model with weights using the provided model initializer.
        The model can be on a meta device.

        Args:
            model (nn.Module): The model to be initialized with weights.
            model_initializer (ModelInitializationIF): The model initializer object.

        Returns:
            nn.Module: The initialized model.
        """

        def reset_parameters_if_function_exists(module: nn.Module):
            # Recursively apply to all submodules
            for submodule in module.children():
                reset_parameters_if_function_exists(submodule)

            # Check if the module has the `reset_parameters` method
            if hasattr(module, "reset_parameters") and callable(getattr(module, "reset_parameters")):
                module.reset_parameters()

        if ModelFactory._is_model_on_meta_device(model=model):
            # Allocate buffers and sharded parameters on GPU
            model = model.to_empty(device="cuda")

        # initialize the weights if they are on a meta device
        # call reset_parameters on all nn.Modules that implement this function
        # (normally all norms)
        with torch.no_grad():
            reset_parameters_if_function_exists(module=model)
            model_initializer.initialize_in_place(model)
        return model

    @staticmethod
    def get_activation_checkpointed_model(model: FSDP1, activation_checkpointing_modules: list[str]) -> FSDP1:
        """Apply activation checkpointing to the given model (in-place operation).

        Args:
            model (FSDP1): The FSDP1-wrapped model to apply activation checkpointing to.
            activation_checkpointing_modules (list[str]): List of module names to apply activation checkpointing to.

        Raises:
            ValueError: Activation checkpointing can only be applied to FSDP1-wrapped models!

        Returns:
            FSDP1: The model with activation checkpointing applied.
        """
        if len(activation_checkpointing_modules) > 0:
            if isinstance(model, FSDP1):
                apply_activation_checkpointing_inplace(
                    model=model,
                    activation_checkpointing_modules=activation_checkpointing_modules,
                )
            else:
                raise ValueError(
                    "Activation checkpointing can only be applied to FSDP1-wrapped models! "
                    f"Current model type: {type(model)}"
                )
        return model

    @staticmethod
    def get_selective_activation_checkpointed_model(
        sac_variant: SelectiveActivationCheckpointingVariants,
        layers_fqn: str,
        model: nn.Module,
        sac_fun_params: (
            SelectiveActivationCheckpointedModelConfig.FullACParams
            | SelectiveActivationCheckpointedModelConfig.SelectiveLayerACParams
            | SelectiveActivationCheckpointedModelConfig.SelectiveOpACParams
        ),
    ) -> nn.Module:
        if not isinstance(model, (nn.Module, FSDP1)):
            raise ValueError(
                "Selective activation checkpointing can only be applied to FSDP1-wrapped or nn.Module models! "
                f"Current model type: {type(model)}"
            )
        SelectiveActivationCheckpointing.apply_selective_activation_checkpointing_(
            model=model,
            layers_fqn=layers_fqn,
            sac_variant=sac_variant,
            sac_fun_params=sac_fun_params,
        )
        return model

    @staticmethod
    def get_compiled_model(model: nn.Module, block_names: list[str], fullgraph: bool, debug: bool) -> nn.Module:
        """Apply torch.compile to each transformer block, which makes compilation efficient due to
        repeated structure. Alternatively one can compile the whole model (after applying DP).
        Inspired by: https://github.com/pytorch/torchtitan/blob/6b2912a9b53464bfef744e62100716271b2b248f/torchtitan/parallelisms/parallelize_llama.py#L275

        Note: With fullgraph=True, we enforce the block to be compiled as a whole, which raises an error on
              graph breaks and maximizes speedup.

        Args:
            model (nn.Module): The model to be compiled.
            block_names (list[str]): List of block names to be compiled individually.
            fullgraph (bool): Flag enforcing the block to be compiled without graph breaks.
            debug (bool): Flag to enable debug mode.

        Returns:
            nn.Module: The compiled model.
        """

        def get_parent_module_and_child_name(child_module: nn.Module, model: nn.Module) -> tuple[nn.Module, str]:
            selected_parent_candidate, selected_child_name = None, None
            num_candidates = 0
            for _, parent_candidate in model.named_modules():
                for child_name, child_candidate in parent_candidate.named_children():
                    if child_candidate is child_module:
                        selected_parent_candidate = parent_candidate
                        selected_child_name = child_name
                        num_candidates += 1
            if num_candidates == 0:
                raise ModelStateError("No valid parent candidate")
            elif num_candidates > 1:
                raise ModelStateError("Multiple valid parent candidates")
            else:
                return selected_parent_candidate, selected_child_name

        # get all block types that we want to compile individually
        block_types = []
        for name in block_names:
            module_class = get_module_class_from_name(model, name)
            if module_class is not None:
                block_types.append(module_class)
            else:
                raise ValueError("None of the provided block_names match any modules in the model")

        block_types = tuple(block_types)

        for _, module in model.named_modules():
            if isinstance(module, block_types):
                options = {"trace.enabled": True} if debug else {}
                compiled_module = torch.compile(module, fullgraph=fullgraph, options=options)
                parent_module, child_name = get_parent_module_and_child_name(child_module=module, model=model)
                parent_module.register_module(name=child_name, module=compiled_module)
        return model


class GPT2ModelFactory:
    @staticmethod
    def get_gpt2_model(
        sample_key: str,
        prediction_key: str,
        poe_type: PositionTypes,
        sequence_length: int,
        vocab_size: int,
        n_layer: int,
        n_head_q: int,
        n_head_kv: int,
        n_embd: int,
        ffn_hidden: int,
        dropout: float,
        bias: bool,
        activation_type: ActivationType,
        attention_implementation: AttentionImplementation,
        attention_config: AttentionConfig,
        attention_norm_config: LayerNormWrapperConfig,
        ffn_norm_config: LayerNormWrapperConfig,
        lm_head_norm_config: LayerNormWrapperConfig,
        use_weight_tying: bool,
        use_meta_device: Optional[bool] = False,
        seed: int = None,
    ) -> GPT2LLM:
        config = dict(
            sample_key=sample_key,
            prediction_key=prediction_key,
            poe_type=poe_type,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head_q=n_head_q,
            n_head_kv=n_head_kv,
            n_embd=n_embd,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            bias=bias,
            activation_type=activation_type,
            attention_implementation=attention_implementation,
            attention_config=attention_config,
            attention_norm_config=attention_norm_config,
            ffn_norm_config=ffn_norm_config,
            lm_head_norm_config=lm_head_norm_config,
            seed=seed,
            use_weight_tying=use_weight_tying,
        )

        if use_meta_device:
            with torch.device("meta"):
                model = GPT2LLM(**config)
        else:
            model = GPT2LLM(**config)
        return model
