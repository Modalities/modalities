"""Helpers for asserting that a training run's checkpoint can actually be resumed.

Used by the MoE expert-parallelism (EP) end-to-end tests. Their optimizers take the custom,
non-torch-standard checkpoint path -- ``EPAdamW`` directly, or several of them wrapped in an
``OptimizersList`` under pipeline parallelism -- so asserting that a checkpoint file appeared says
nothing about whether the optimizer state can be read back into a fresh optimizer. These helpers
resume the checkpoint into freshly built model/optimizer instances and compare the restored Adam
moments and scheduler state against the trained ones.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from modalities.checkpointing.fsdp.fsdp_checkpoint_loading import DCPCheckpointLoading
from modalities.checkpointing.stateful.app_state import AppState


def get_last_checkpoint_dir_path(checkpoint_root_path: Path) -> Path:
    """Returns the checkpoint folder that the last DCP save wrote, as recorded by rank 0.

    Barriers first, because only rank 0 writes ``last_checkpoint_info.json`` while every rank needs
    the path (the subsequent load is collective).
    """
    dist.barrier()
    checkpoint_info_file_path = checkpoint_root_path / "last_checkpoint_info.json"
    assert checkpoint_info_file_path.exists(), "Expected checkpoint info file from DCP save."
    with open(checkpoint_info_file_path, "r", encoding="utf-8") as f:
        checkpoint_info = json.load(f)
    checkpoint_dir_path = Path(checkpoint_info["checkpoint_folder_path"])
    assert checkpoint_dir_path.exists(), f"Checkpoint folder {checkpoint_dir_path} does not exist."
    return checkpoint_dir_path


def _collect_tensor_leaves(state: Any, path: str = "") -> dict[str, torch.Tensor]:
    """Flattens an arbitrarily nested state dict into {path: cpu tensor}.

    Works for both optimizer state-dict layouts in use here (the standard one, EPAdamW's nested one
    and the per-stage-indexed OptimizersList one) without hard-coding any of them. DTensors are
    reduced to their local shard, which is what each rank owns and checkpoints.
    """
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(state, dict):
        for key, value in state.items():
            leaves.update(_collect_tensor_leaves(value, f"{path}/{key}"))
    elif isinstance(state, (list, tuple)):
        for idx, value in enumerate(state):
            leaves.update(_collect_tensor_leaves(value, f"{path}/{idx}"))
    elif isinstance(state, DTensor):
        leaves[path] = state.to_local().detach().cpu()
    elif isinstance(state, torch.Tensor):
        leaves[path] = state.detach().cpu()
    return leaves


def assert_checkpoint_resumes_optimizer_and_scheduler_state(
    trained_app_state: AppState,
    resumed_app_state: AppState,
    checkpoint_dir_path: Path,
    global_rank: int,
) -> None:
    """Loads the checkpoint into ``resumed_app_state`` (fresh model/optimizer/scheduler instances)
    and asserts that its optimizer moments and scheduler position match the trained run."""
    trained_optimizer_state = _collect_tensor_leaves(trained_app_state.optimizer.state_dict())
    assert trained_optimizer_state, "Expected non-empty optimizer state (Adam moments) after training."
    trained_scheduler_state = dict(trained_app_state.lr_scheduler.state_dict())
    assert trained_scheduler_state["last_epoch"] > 0, "Expected the lr scheduler to have advanced during training."

    resumed_optimizer_state_before_load = _collect_tensor_leaves(resumed_app_state.optimizer.state_dict())
    assert (
        resumed_app_state.lr_scheduler.state_dict()["last_epoch"] != trained_scheduler_state["last_epoch"]
    ), "Expected the freshly built lr scheduler to differ from the trained one, otherwise the check below is vacuous."

    DCPCheckpointLoading(global_rank=global_rank).load_checkpoint_(
        app_state=resumed_app_state, checkpoint_dir_path=checkpoint_dir_path
    )

    resumed_optimizer_state = _collect_tensor_leaves(resumed_app_state.optimizer.state_dict())
    assert set(resumed_optimizer_state.keys()) == set(trained_optimizer_state.keys()), (
        "Restored optimizer state has different entries than the trained one: "
        f"missing={sorted(set(trained_optimizer_state) - set(resumed_optimizer_state))}, "
        f"unexpected={sorted(set(resumed_optimizer_state) - set(trained_optimizer_state))}."
    )
    for key, trained_value in trained_optimizer_state.items():
        torch.testing.assert_close(
            resumed_optimizer_state[key], trained_value, msg=f"Optimizer state entry {key} was not restored."
        )
    # Guard against a vacuous comparison: at least one entry must have actually changed by loading.
    assert any(
        not torch.equal(resumed_optimizer_state[key], resumed_optimizer_state_before_load[key])
        for key in resumed_optimizer_state
        if key in resumed_optimizer_state_before_load
        and resumed_optimizer_state[key].shape == resumed_optimizer_state_before_load[key].shape
    ), "Loading the checkpoint did not change any optimizer state entry."

    assert resumed_app_state.lr_scheduler.state_dict()["last_epoch"] == trained_scheduler_state["last_epoch"]
