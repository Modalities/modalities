from einsum_transformer import EinsumTransformer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule as FSDP2

from modalities.running_env.env_utils import FSDP2MixedPrecisionSettings
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.util import get_module_class_from_name


def get_fsdp2_wrapped_model(
    model: EinsumTransformer,
    block_names: list[str],
    device_mesh: DeviceMesh,
    mixed_precision_settings: FSDP2MixedPrecisionSettings,
    reshard_after_forward: bool,
) -> FSDP2:
    # map the block names to the actual block class (e.b., GPT2Block)
    block_types = tuple([get_module_class_from_name(model, b) for b in block_names])

    mp_policy = MixedPrecisionPolicy(
        param_dtype=mixed_precision_settings.param_dtype.value,
        reduce_dtype=mixed_precision_settings.reduce_dtype.value,
    )
    # if DP_REPLICATE is not in the mesh, we apply full sharding and hybrid sharding otherwise
    fsdp2_degrees = (
        (ParallelismDegrees.DP_REPLICATE.value, ParallelismDegrees.DP_SHARD.value)
        if ParallelismDegrees.DP_REPLICATE.value in device_mesh.mesh_dim_names
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
    return model
