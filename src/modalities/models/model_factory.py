import itertools
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from typing_extensions import deprecated

from modalities.checkpointing.checkpoint_loading import FSDP1CheckpointLoadingIF
from modalities.config.config import ActivationCheckpointedModelConfig
from modalities.exceptions import ModelStateError
from modalities.models.gpt2.gpt2_model import (
    GPT2LLM,
    AttentionConfig,
    AttentionImplementation,
    LayerNormWrapperConfig,
    PositionTypes,
    SwiGLU,
    TransformerMLP,
)
from modalities.models.model import ActivationType
from modalities.nn.model_initialization.initialization_if import ModelInitializationIF
from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings, MixedPrecisionSettings
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.running_env.fsdp.fsdp_auto_wrapper import FSDPTransformerAutoWrapPolicyFactory
from modalities.training.activation_checkpointing.activation_checkpointing import (
    ActivationCheckpointing,
    apply_activation_checkpointing_inplace,
)
from modalities.training.activation_checkpointing.activation_checkpointing_variants import (
    ActivationCheckpointingVariants,
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
    ) -> FSDP1:
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
        # if DP_REPLICATE is not in the mesh, we apply full sharding and hybrid sharding otherwise
        fsdp2_degrees = (
            (ParallelismDegrees.DP_REPLICATE.value, ParallelismDegrees.DP_SHARD.value)
            if ParallelismDegrees.DP_REPLICATE in device_mesh.mesh_dim_names
            else (ParallelismDegrees.DP_SHARD.value,)
        )
        fsdp_config = {"mesh": device_mesh[fsdp2_degrees], "mp_policy": mp_policy}

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
    def get_weight_initialized_model(model: nn.Module, model_initializer: ModelInitializationIF) -> nn.Module:
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
    def get_activation_checkpointed_fsdp1_model_(model: FSDP1, activation_checkpointing_modules: list[str]) -> FSDP1:
        """Apply activation checkpointing to the given FSDP1-wrapped model (in-place operation).

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
    def get_activation_checkpointed_model_(
        ac_variant: ActivationCheckpointingVariants,
        layers_fqn: str,
        model: nn.Module,
        ac_fun_params: (
            ActivationCheckpointedModelConfig.FullACParams
            | ActivationCheckpointedModelConfig.SelectiveLayerACParams
            | ActivationCheckpointedModelConfig.SelectiveOpACParams
        ),
    ) -> nn.Module:
        """FSDP2 variant for applying activation checkpointing to the given model (in-place operation).
        When using FSDP2, we always first apply activation checkpointing to the model and then wrap it with FSDP2.

        Args:
            ac_variant (ActivationCheckpointingVariants): The activation checkpointing variant to use.
            layers_fqn (str): Fully qualified name (FQN) of the layers to apply activation checkpointing to.
            model (nn.Module): The (unwrapped) model to apply activation checkpointing to.
            ac_fun_params (ACM.FullACParams  |  ACM.SelectiveLayerACParams  |  ACM.SelectiveOpACParams):
                The parameters for the activation checkpointing function, depending on the variant.

        Raises:
            ValueError: Activation checkpointing can only be applied to unwrapped nn.Module models

        Returns:
            nn.Module: The model with activation checkpointing applied.
        """
        if not isinstance(model, nn.Module):
            raise ValueError(
                "Activation checkpointing can only be applied to unwrapped nn.Module model! "
                f"Current model type: {type(model)}"
            )

        ActivationCheckpointing.apply_activation_checkpointing_(
            model=model,
            layers_fqn=layers_fqn,
            ac_variant=ac_variant,
            ac_fun_params=ac_fun_params,
        )
        return model

    @staticmethod
    def get_compiled_model(
        model: nn.Module, block_names: list[str], fullgraph: bool, debug: Optional[bool] = False
    ) -> nn.Module:
        """Apply torch.compile to each transformer block, which makes compilation efficient due to
        repeated structure. Alternatively one can compile the whole model (after applying DP).
        Inspired by: https://github.com/pytorch/torchtitan/blob/6b2912a9b53464bfef744e62100716271b2b248f/torchtitan/parallelisms/parallelize_llama.py#L275

        Note: With fullgraph=True, we enforce the block to be compiled as a whole, which raises an error on
              graph breaks and maximizes speedup.

        Args:
            model (nn.Module): The model to be compiled.
            block_names (list[str]): List of block names to be compiled individually.
            fullgraph (bool): Flag enforcing the block to be compiled without graph breaks.
            debug (Optional[bool]): Flag to enable debug mode. Default is False.

        Returns:
            nn.Module: The compiled model.
        """

        def get_parent_module_and_child_name(child_module: nn.Module, model: nn.Module) -> tuple[nn.Module, str]:
            # returns the parent module and the child name of the given child module
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
                raise ValueError(f"The block name {name} does not match any modules in the model")

        block_types = tuple(block_types)

        for _, module in model.named_modules():
            if isinstance(module, block_types):
                options = {"trace.enabled": True} if debug else {}
                compiled_module = torch.compile(module, fullgraph=fullgraph, options=options)
                parent_module, child_name = get_parent_module_and_child_name(child_module=module, model=model)
                parent_module.register_module(name=child_name, module=compiled_module)
        return model

    @staticmethod
    def get_debugging_enriched_model(
        model: nn.Module, logging_dir_path: Path, tracked_ranks: Optional[Set[int]] = None, log_interval_steps: int = 1
    ) -> nn.Module:
        """
        Enriches the model with debugging hooks to log tensor statistics during forward and backward passes.
        During the forward pass, it logs the input and output tensors of each module, as well as the parameters.
        Similarly, during the backward pass, it logs the gradients of the output tensors.

        The following tensor statistics are logged:
            - global shape
            - local shape
            - is_dtensor (whether the tensor is a DTensor)
            - nan count
            - inf count
            - mean
            - std
            - min
            - max
        The statistics are written to a JSONL file in the specified logging directory.

        Args:
            model (nn.Module): The model to be enriched with debugging hooks.
            logging_dir_path (Path): The directory path where the tensor statistics will be logged.
            tracked_ranks (Optional[Set[int]]): A set of ranks to track. If provided, only these ranks
                will log the statistics. If None, all ranks will log the statistics.
            log_interval_steps (int): The interval in steps at which to log the tensor statistics. Default is 1.
        """

        @dataclass
        class TensorStats:
            """Dataclass to hold tensor statistics."""

            global_shape: list[int]
            local_shape: list[int]
            is_dtensor: bool
            nan_count: int
            inf_count: int
            mean: float
            std: float
            min: float
            max: float

        @dataclass
        class CounterRef:
            """Dataclass to hold a counter reference for tracking the number of hooks called.
            This is used as a closure to keep track of the number of hooks called."""

            value: int = 0

        rank = dist.get_rank() if dist.is_initialized() else 0

        if tracked_ranks is not None and rank not in tracked_ranks:
            return model
        if rank == 0:
            logging_dir_path.mkdir(parents=True, exist_ok=True)
        logging_file_path = logging_dir_path / f"tensor_stats_rank_{rank}.jsonl"

        def get_tensor_stats(tensor: torch.Tensor) -> TensorStats:
            """Get statistics of a tensor."""
            local_tensor = tensor.to_local() if isinstance(tensor, dist.tensor.DTensor) else tensor
            float_dtypes = {torch.float, torch.bfloat16}
            numeric_dtypes = float_dtypes | {torch.int, torch.long}

            dtype = local_tensor.dtype
            is_float = dtype in float_dtypes
            is_numeric = dtype in numeric_dtypes

            tensor_stats = TensorStats(
                global_shape=list(tensor.shape),
                local_shape=list(local_tensor.shape),
                is_dtensor=isinstance(tensor, dist.tensor.DTensor),
                nan_count=torch.isnan(local_tensor).sum().item(),
                inf_count=torch.isinf(local_tensor).sum().item(),
                mean=local_tensor.mean().item() if is_float else -1,
                std=local_tensor.std().item() if is_float else -1,
                min=local_tensor.min().item() if is_numeric else -1,
                max=local_tensor.max().item() if is_numeric else -1,
            )
            return tensor_stats

        def write_out_tensor_stats(tensor_stats: TensorStats, counter: int, hook_type: str, tensor_tag: str, rank: int):
            """Write out tensor statistics to a file."""
            with open(logging_file_path, "a", encoding="utf-8") as f:
                tensor_stats_dict = asdict(tensor_stats)
                tensor_stats_dict = {
                    "tensor_tag": tensor_tag,
                    "hook_type": hook_type,
                    **tensor_stats_dict,
                    "counter": counter,
                    "rank": rank,
                }

                f.write(json.dumps(tensor_stats_dict) + "\n")

        def pre_forward_hook(module: nn.Module, forward_input, counter: CounterRef, log_interval_steps: int):
            if log_interval_steps > 0 and counter.value % log_interval_steps != 0:
                counter.value += 1
                return

            if isinstance(forward_input, tuple):
                forward_inputs = forward_input
            else:
                forward_inputs = (forward_input,)

            for forward_input in forward_inputs:
                tensor_stats = get_tensor_stats(forward_input)
                write_out_tensor_stats(tensor_stats, counter.value, "forward_input", module._debug_name, rank)

            # Retrieves statistics of the module's parameters before forward pass.
            for name, param in module.named_parameters(recurse=False):
                tensor_stats = get_tensor_stats(param)
                full_name = f"{module._debug_name}.{name}"
                write_out_tensor_stats(
                    tensor_stats=tensor_stats,
                    counter=counter.value,
                    hook_type="forward_weights",
                    tensor_tag=full_name,
                    rank=rank,
                )
            counter.value += 1

        def forward_hook(module: nn.Module, foward_input, forward_output, counter: CounterRef, log_interval_steps: int):
            if log_interval_steps > 0 and counter.value % log_interval_steps != 0:
                counter.value += 1
                return

            if isinstance(forward_output, tuple):
                forward_outputs = forward_output
            else:
                forward_outputs = (forward_output,)

            for out in forward_outputs:
                tensor_stats = get_tensor_stats(out)
                write_out_tensor_stats(tensor_stats, counter.value, "forward_output", module._debug_name, rank)
            counter.value += 1

        def backward_hook(module, grad_input, grad_output, counter: CounterRef, log_interval_steps: int):
            if log_interval_steps > 0 and counter.value % log_interval_steps != 0:
                counter.value += 1
                return

            for grad_out in grad_output:
                tensor_stats = get_tensor_stats(grad_out)
                write_out_tensor_stats(tensor_stats, counter.value, "backward_output", module._debug_name, rank)
            counter.value += 1

        def register_hooks_recursively(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                child._debug_name = full_name

                child.register_forward_pre_hook(
                    partial(pre_forward_hook, counter=CounterRef(), log_interval_steps=log_interval_steps)
                )
                child.register_forward_hook(
                    partial(forward_hook, counter=CounterRef(), log_interval_steps=log_interval_steps)
                )
                child.register_full_backward_hook(
                    partial(backward_hook, counter=CounterRef(), log_interval_steps=log_interval_steps)
                )
                register_hooks_recursively(child, full_name)

        register_hooks_recursively(model)

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
        if use_meta_device and use_weight_tying:
            raise ValueError(
                "Weight tying is not supported on meta device. "
                "Please set at least use_meta_device=False or use_weight_tying=False."
                "https://github.com/Modalities/modalities/issues/357"
            )
        if use_meta_device:
            with torch.device("meta"):
                model = GPT2LLM(**config)
        else:
            model = GPT2LLM(**config)
        return model

    @staticmethod
    def get_gpt2_tensor_parallelized_model(model: GPT2LLM, device_mesh: DeviceMesh) -> nn.Module:
        tp_mesh = device_mesh[ParallelismDegrees.TP.value]
        model_tp_plan = {
            # Row-wise parallelism might seem counterintuitive here,
            # but the embedding layer has weight shape (vocab_size, n_embd).
            # Row-wise sharding allows each rank to store a slice of the vocabulary
            # and perform lookups only for the tokens it owns.
            # The input token IDs are replicated across all ranks so that each rank
            # can identify which tokens it is responsible for.
            # Each rank produces a partial embedding output, and an all-reduce is performed
            # in the background to obtain the full embedding vectors of shape
            # (batch_size, sequence_length, n_embd).
            # Finally, we shard the output on the sequence dimension to enable sequence parallelism
            # in the downstream transformer blocks.
            "transformer.wte": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "transformer.lm_head_norm": SequenceParallel(),
            "transformer.lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),  # Shard(-1) if loss parallelism is used
                use_local_output=True,  # default, should be not loss_parallel if loss parallelism is used
            ),
        }

        if isinstance(model.transformer.wpe, nn.Embedding):
            # If the position embedding is an nn.Embedding, we can shard it on the sequence dimension
            # to enable sequence parallelism in the downstream transformer blocks.
            # Note, for RoPE the wpe layer is an identity operation, which cannnot be sharded.
            model_tp_plan["transformer.wpe"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(0),
            )

        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=model_tp_plan,
        )

        transformer_block_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attn": PrepareModuleInput(
                # here we prepare the actual input of the attention module
                # (i.e., the arguements to the forward method)
                # The incomming inputs are sharded on the sequence dimension
                # due to the pre-layer attention norm running sequence parallelism.
                # The inputs are transformed into the desired format by replicating
                # them across all ranks.
                # In the pytorch tutorial and torch titan we pass in an additional None argument
                # for freqs_cis (i.e., precomputed cosine and sine frequencies.), which is not
                # needed here due to implementation differences.
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.q_attn": ColwiseParallel(),
            "attn.k_attn": ColwiseParallel(),
            "attn.v_attn": ColwiseParallel(),
            "attn.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
        }
        if isinstance(model.transformer.h[0].mlp, SwiGLU):
            mlp_plan = {
                "mlp.W": ColwiseParallel(),
                "mlp.W_2": RowwiseParallel(output_layouts=Shard(1)),
                "mlp.V": ColwiseParallel(),
            }
        elif isinstance(model.transformer.h[0].mlp, TransformerMLP):
            mlp_plan = {
                "mlp.c_fc": ColwiseParallel(),
                "mlp.c_proj": RowwiseParallel(output_layouts=Shard(1)),
            }
        else:
            raise NotImplementedError(
                "Only SwiGLU and GELU (used in TransformersMLP) are supported for the MLP in GPT2. "
                "Please implement the tensor parallelization for other MLP types."
            )
        transformer_block_tp_plan.update(mlp_plan)

        for transformer_block in model.transformer.h:
            # override the number of q and kv heads
            if transformer_block.attn.n_head_q % tp_mesh.size() != 0:
                raise ValueError(
                    f"Number of query heads {transformer_block.attn.n_head_q} must be divisible by "
                    f"the number of tensor parallel devices {tp_mesh.size()}."
                )
            if transformer_block.attn.n_head_kv % tp_mesh.size() != 0:
                raise ValueError(
                    f"Number of key-value heads {transformer_block.attn.n_head_kv} must be divisible by "
                    f"the number of tensor parallel devices {tp_mesh.size()}."
                )
            transformer_block.attn.n_head_q = transformer_block.attn.n_head_q // tp_mesh.size()
            transformer_block.attn.n_head_kv = transformer_block.attn.n_head_kv // tp_mesh.size()
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=transformer_block_tp_plan,
            )

        return model
