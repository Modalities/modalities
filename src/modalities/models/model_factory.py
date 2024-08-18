import itertools
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from typing_extensions import deprecated

from modalities.checkpointing.checkpoint_loading import CheckpointLoadingIF
from modalities.exceptions import ModelStateError
from modalities.models.components.layer_norms import LayerNormConfig
from modalities.models.gpt2.gpt2_model import GPT2LLM, AttentionConfig, AttentionImplementation, PositionTypes
from modalities.models.model import ActivationType
from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings, MixedPrecisionSettings
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory
from modalities.util import get_local_number_of_trainable_parameters, get_module_class_from_name


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

    @deprecated(
        "With version 0.3, we upgraded FSDP to FSDP 2.0. Use get_fsdp_2_wrapped_model(...) instead.",
        category=FutureWarning,
    )
    @staticmethod
    def get_fsdp_wrapped_model(
        model: nn.Module,
        sync_module_states: bool,
        block_names: List[str],
        mixed_precision_settings: MixedPrecisionSettings,
        sharding_strategy: ShardingStrategy,
    ) -> FSDP:
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
    def get_fsdp_2_wrapped_model(
        model: nn.Module,
        block_names: List[str],
        device_mesh: DeviceMesh,
        mixed_precision_settings: FSDP2MixedPrecisionSettings,
        reshard_after_forward: bool,
    ) -> nn.Module:
        """
        Based on https://github.com/pytorch/torchtitan/blob/de9fd2b9ea7e763c9182e0df81fc32c2618cc0b6/torchtitan/parallelisms/parallelize_llama.py#L459
        and https://github.com/pytorch/torchtitan/blob/43584e0a4e72645e25cccd05d86f9632587a8beb/docs/fsdp.md
        NOTE: Torch Titan already implement pipeline parallelism. We skip that here for now.
        """

        print(
            f"Unsharded number of parameters on rank {dist.get_rank()}: "
            f"{get_local_number_of_trainable_parameters(model)}"
        )
        # map the block names to the actual block class (e.b., GPT2Block)
        block_types = tuple([get_module_class_from_name(model, b) for b in block_names])

        mp_policy = MixedPrecisionPolicy(
            param_dtype=mixed_precision_settings.param_dtype.value,
            reduce_dtype=mixed_precision_settings.reduce_dtype.value,
        )

        fsdp_config = {"mesh": device_mesh["dp"], "mp_policy": mp_policy}

        modules = list(model.modules())
        # we first shard all the blocks
        for module_id, module in enumerate(modules):
            if isinstance(module, block_types):
                # As an optimization, we do not reshard after forward for the last
                # transformer block since FSDP would prefetch it immediately.
                fully_shard(
                    module,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward and int(module_id) < len(modules) - 1,
                )
        # finally, we shard the entire model
        fully_shard(model, **fsdp_config, reshard_after_forward=reshard_after_forward)

        return model

    @staticmethod
    def get_weight_initalized_model(model: nn.Module, model_initializer: ModelInitializationIF) -> nn.Module:
        # initialize the weights if they are on a meta device
        def is_model_on_meta_device(model: nn.Module) -> bool:
            meta_counter = 0
            param_counter = 0
            for tensor in itertools.chain(model.parameters(), model.buffers()):
                if tensor.device == torch.device("meta"):
                    meta_counter += 1
                param_counter += 1

            if meta_counter > 0 and meta_counter < param_counter:
                raise ModelStateError("Either all or none of the parameters and buffers must be on meta device!")
            return meta_counter > 0

        if is_model_on_meta_device(model=model):
            # Allocate buffers and sharded parameters on GPU
            model = model.to_empty(device="cuda")

        model_initializer.initialize_in_place(model)
        return model

    @staticmethod
    def get_compiled_model(model: nn.Module, block_names: List[str]) -> nn.Module:
        """
        Apply torch.compile to each transformer block, which makes compilation efficient due to
        repeated structure. Alternatively one can compile the whole model (after applying DP).

        Inspired by: https://github.com/pytorch/torchtitan/blob/6b2912a9b53464bfef744e62100716271b2b248f/torchtitan/parallelisms/parallelize_llama.py#L275
        """

        def get_parent_module_and_child_name(child_module: nn.Module, model: nn.Module) -> nn.Module:
            for _, parent_candidate in model.named_modules():
                for child_name, child_candidate in parent_candidate.named_children():
                    if child_candidate is child_module:
                        return parent_candidate, child_name
            raise ModelStateError("No valid parent candidate")

        block_types = tuple([get_module_class_from_name(model, b) for b in block_names])

        for _, module in model.named_modules():
            if isinstance(module, block_types):
                compiled_module = torch.compile(module, fullgraph=True)
                parent_module, child_name = get_parent_module_and_child_name(child_module=module, model=model)
                parent_module.register_module(name=child_name, module=compiled_module)

        return model

    @staticmethod
    def get_tensor_parallelized_model(
        model: nn.Module,
        device_mesh: DeviceMesh,
    ) -> nn.Module:
        # TODO: this is gpt-2 specific and should be part of the configuration
        attention_block_tp_plan = {
            # by default ColwiseParallel input layouts is replicated
            # and RowwiseParallel output layouts is replicated
            # SwiGLU parallelization
            "mlp.W": ColwiseParallel(),
            "mlp.W_2": RowwiseParallel(),
            "mlp.V": ColwiseParallel(),
            # attention matrices parallelization
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attn.q_attn": ColwiseParallel(),
            "attn.k_attn": ColwiseParallel(),
            "attn.v_attn": ColwiseParallel(),  # default: input replicated, output sharded on dim -1
            "attn.c_proj": RowwiseParallel(output_layouts=Shard(1)),  # input sharded on dim -1, output sharded on dim 1
            # norms
            "attention_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
        }

        outer_model_tp_plan = {
            "transformer.wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "transformer.wpe": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "lm_head_norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),  # Shard(-1) if loss_parallel else Replicate(),
                use_local_output=True,  # not loss_parallel,
            ),
        }

        model = parallelize_module(model, device_mesh["tp"], outer_model_tp_plan)
        print(outer_model_tp_plan, attention_block_tp_plan)

        return model

    @staticmethod
    def get_gpt2_model(
        use_meta_device: bool,
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
        attention_norm_config: LayerNormConfig,
        ffn_norm_config: LayerNormConfig,
        lm_head_norm_config: LayerNormConfig,
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
        )

        if use_meta_device:
            with torch.device("meta"):
                model = GPT2LLM(**config)
        else:
            model = GPT2LLM(**config)
        return model
