#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading
from modalities.checkpointing.stateful.app_state import AppState
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticAppStateType, PydanticDeviceMeshIFType
from modalities.main import Main
from modalities.running_env.cuda_env import CudaEnv
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_mesh_for_parallelism_method


class ComponentsInstantiationModel(BaseModel):
    app_state: PydanticAppStateType
    device_mesh: PydanticDeviceMeshIFType | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one or more Modalities DCP checkpoints into an app state.")
    parser.add_argument("--config-file-path", type=Path, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--experiments-root-path",
        type=Path,
        required=True,
        help="Path passed to Main for resolver/context setup.",
    )
    parser.add_argument(
        "--checkpoint-dir-paths",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to multiple checkpoint directories containing *.distcp files.",
    )
    return parser.parse_args()


def _resolve_checkpoint_dir_paths(args: argparse.Namespace) -> list[Path]:
    return list(args.checkpoint_dir_paths)


def _get_layer_key(parameter_name: str) -> str:
    # Strip common wrapping prefixes that appear for wrapped modules.
    name = parameter_name
    for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    tokens = name.split(".")
    for i in range(len(tokens) - 1):
        if tokens[i] in {"h", "layers", "blocks"} and tokens[i + 1].isdigit():
            if i > 0:
                return ".".join(tokens[i - 1 : i + 2])
            return ".".join(tokens[i : i + 2])

    # Fallback: group by parent module path if no canonical layer index token exists.
    return ".".join(tokens[:-1]) if len(tokens) > 1 else name


def _get_dp_shard_group(device_mesh: DeviceMesh | None):
    if device_mesh is None:
        return None
    try:
        return get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.DP_SHARD).get_group()
    except Exception:
        # Fallback to the default process group if a dedicated DP-shard group is unavailable.
        return None


def _compute_and_print_layer_norms(app_state: AppState, dp_shard_group) -> None:
    layer_sq_sums: dict[str, torch.Tensor] = {}

    for model_part_idx, model_part in enumerate(app_state.model_parts):
        for name, parameter in model_part.named_parameters():
            if not parameter.requires_grad:
                continue
            full_name = f"model_part_{model_part_idx}.{name}" if len(app_state.model_parts) > 1 else name
            layer_key = _get_layer_key(full_name)

            # FSDP2 parameters can be DTensors. Convert to local shard first so c10d all_reduce
            # operates on plain tensors instead of DTensors.
            local_param = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            local_sq_sum = local_param.detach().float().pow(2).sum()
            layer_sq_sums[layer_key] = layer_sq_sums.get(layer_key, torch.zeros_like(local_sq_sum)) + local_sq_sum

    # Aggregate over the DP-shard group to reconstruct global norms for sharded parameters.
    for layer_key, sq_sum in layer_sq_sums.items():
        dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM, group=dp_shard_group)
        layer_sq_sums[layer_key] = sq_sum

    if dist.get_rank() == 0:
        print("Per-layer parameter L2 norms (global across DP-shards):")
        for layer_key in sorted(layer_sq_sums):
            norm = torch.sqrt(layer_sq_sums[layer_key]).item()
            print(f"{layer_key}: {norm:.6f}")


def main() -> None:
    args = _parse_args()
    checkpoint_dir_paths = _resolve_checkpoint_dir_paths(args)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        rank = dist.get_rank()

        for checkpoint_dir_path in checkpoint_dir_paths:
            # Rebuild components per checkpoint because AppState only supports one load call.
            main_obj = Main(
                config_path=args.config_file_path,
                experiments_root_path=args.experiments_root_path,
            )
            components = cast(
                ComponentsInstantiationModel,
                main_obj.build_components(components_model_type=ComponentsInstantiationModel),
            )

            app_state = cast(AppState, getattr(components, "app_state"))
            device_mesh = cast(DeviceMesh | None, getattr(components, "device_mesh", None))

            loader = DCPCheckpointLoading(global_rank=rank)
            loader.load_checkpoint_(app_state=app_state, checkpoint_dir_path=checkpoint_dir_path)

            dp_shard_group = _get_dp_shard_group(device_mesh)
            if rank == 0:
                print(f"\n=== {checkpoint_dir_path} ===")
            _compute_and_print_layer_norms(app_state, dp_shard_group)

            if rank == 0:
                print(
                    f"Loaded checkpoint from {checkpoint_dir_path} on world size {dist.get_world_size()} "
                    f"(pid={os.getpid()})."
                )


if __name__ == "__main__":
    main()
