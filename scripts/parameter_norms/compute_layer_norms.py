#!/usr/bin/env python3

import argparse
import json
import os
import re
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
    parser.add_argument(
        "--json-output-path",
        type=Path,
        default=Path("layer_norms_across_checkpoints.json"),
        help="Output path for raw per-checkpoint norms as JSON.",
    )
    return parser.parse_args()


def _resolve_checkpoint_dir_paths(args: argparse.Namespace) -> list[Path]:
    return list(args.checkpoint_dir_paths)


def _normalize_parameter_name(parameter_name: str) -> str:
    name = parameter_name
    for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _get_dp_shard_group(device_mesh: DeviceMesh | None):
    if device_mesh is None:
        return None
    try:
        return get_mesh_for_parallelism_method(device_mesh, ParallelismDegrees.DP_SHARD).get_group()
    except Exception:
        # Fallback to the default process group if a dedicated DP-shard group is unavailable.
        return None


def _compute_and_print_parameter_norms(app_state: AppState, dp_shard_group) -> dict[str, float]:
    parameter_sq_sums: dict[str, torch.Tensor] = {}

    for model_part_idx, model_part in enumerate(app_state.model_parts):
        for name, parameter in model_part.named_parameters():
            if not parameter.requires_grad:
                continue
            raw_name = f"model_part_{model_part_idx}.{name}" if len(app_state.model_parts) > 1 else name
            parameter_name = _normalize_parameter_name(raw_name)

            # FSDP2 parameters can be DTensors. Convert to local shard first so c10d all_reduce
            # operates on plain tensors instead of DTensors.
            local_param = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            local_sq_sum = local_param.detach().float().pow(2).sum()
            parameter_sq_sums[parameter_name] = local_sq_sum

    # Aggregate over the DP-shard group to reconstruct global norms for sharded parameters.
    for parameter_name, sq_sum in parameter_sq_sums.items():
        dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM, group=dp_shard_group)
        parameter_sq_sums[parameter_name] = sq_sum

    parameter_norms = {name: torch.sqrt(sq_sum).item() for name, sq_sum in parameter_sq_sums.items()}

    if dist.get_rank() == 0:
        print("Per-parameter L2 norms (global across DP-shards):")
        for parameter_name in sorted(parameter_norms):
            print(f"{parameter_name}: {parameter_norms[parameter_name]:.6f}")

    return parameter_norms


def _extract_checkpoint_label(checkpoint_dir_path: Path) -> str:
    match = re.search(r"seen_steps_(\d+)", checkpoint_dir_path.name)
    if match:
        return f"steps_{match.group(1)}"
    return checkpoint_dir_path.name


def _save_json_results(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    args = _parse_args()
    checkpoint_dir_paths = _resolve_checkpoint_dir_paths(args)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        rank = dist.get_rank()
        collected_results: list[dict] = []

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
            parameter_norms = _compute_and_print_parameter_norms(app_state, dp_shard_group)

            if rank == 0:
                collected_results.append(
                    {
                        "checkpoint_path": str(checkpoint_dir_path),
                        "checkpoint_label": _extract_checkpoint_label(checkpoint_dir_path),
                        "parameter_norms": parameter_norms,
                    }
                )
                print(
                    f"Loaded checkpoint from {checkpoint_dir_path} on world size {dist.get_world_size()} "
                    f"(pid={os.getpid()})."
                )

        if rank == 0:
            _save_json_results(collected_results, args.json_output_path)
            print(f"Saved raw parameter norms JSON to {args.json_output_path}")


if __name__ == "__main__":
    main()
