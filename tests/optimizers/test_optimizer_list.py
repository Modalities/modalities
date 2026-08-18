from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.optim import AdamW

from modalities.checkpointing.stateful.app_state import AppState
from modalities.models.model import NNModel
from modalities.optimizers.ep_adamw import EPAdamW, _init_optimizer_state_
from modalities.optimizers.optimizer_list import OptimizersList


class DummyDPShardMesh:
    def __init__(self):
        self._group = object()

    def get_group(self):
        return self._group


class EPSubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ep_weight = nn.Parameter(torch.tensor([1.0, -1.0]))
        self._ep_enabled = True


class TinyStageModel(NNModel):
    """Stands in for one pipeline stage. The stage index is part of every parameter name so that
    the model state dict keys stay unique across the stages held by one rank (AppState asserts this).
    """

    def __init__(self, stage_idx: int):
        super().__init__(weight_decay_groups={"linear": ["linear"], "layernorm": ["norm"]})
        self.blocks = nn.ModuleDict({f"stage_{stage_idx}": nn.Module()})
        block = self.blocks[f"stage_{stage_idx}"]
        block.linear = nn.Linear(2, 2, bias=False)
        block.norm = nn.LayerNorm(2)
        block.experts = EPSubmodule()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        block = next(iter(self.blocks.values()))
        return {"y": block.linear(inputs["x"])}


def _patch_distributed_ops(monkeypatch):
    from modalities.optimizers import ep_adamw as ep_adamw_module

    monkeypatch.setattr(ep_adamw_module.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(ep_adamw_module.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(ep_adamw_module.dist, "all_reduce", lambda tensor, op=None, group=None: tensor)
    monkeypatch.setattr(ep_adamw_module.dist, "broadcast", lambda tensor, src=0, group=None: tensor)
    monkeypatch.setattr(ep_adamw_module.dist, "get_global_rank", lambda group, group_rank: group_rank)


def _build_app_state(num_stages: int = 2) -> AppState:
    """Builds an AppState with one EPAdamW per stage, wrapped in an OptimizersList -- i.e. the
    EP + pipeline-parallel setup, where OptimizersList takes its native state-dict branch."""
    torch.manual_seed(0)
    model_parts = [TinyStageModel(stage_idx=stage_idx) for stage_idx in range(num_stages)]
    optimizers = [
        EPAdamW(
            model=model_part,
            device_mesh={"dp_shard": DummyDPShardMesh()},
            lr=1e-2,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            weight_decay_groups_excluded=["layernorm"],
        )
        for model_part in model_parts
    ]
    optimizer_list = OptimizersList(model_parts=model_parts, optimizers=optimizers)
    return AppState(model=model_parts, optimizer=optimizer_list)


def _get_moments(state_dict: dict[str, Any], stage_idx: int, sub_optimizer: str) -> dict[Any, dict[str, torch.Tensor]]:
    return {
        param_id: {key: value.clone() for key, value in param_state.items()}
        for param_id, param_state in state_dict[str(stage_idx)][sub_optimizer]["state"].items()
    }


def test_optimizers_list_state_dict_namespaces_ep_optimizers_per_stage(monkeypatch):
    """The native (EP) branch keys each stage's optimizer state by its position, so that the states
    of multiple stages held by one rank (interleaved schedules) cannot collide."""
    _patch_distributed_ops(monkeypatch)
    app_state = _build_app_state(num_stages=2)

    state_dict = app_state.optimizer.state_dict()

    assert set(state_dict.keys()) == {"0", "1"}
    for stage_state_dict in state_dict.values():
        assert set(stage_state_dict.keys()) == {"ep_adamw", "dense_adamw"}


def test_optimizers_list_ep_optimizer_state_survives_dcp_round_trip(monkeypatch, tmp_path: Path):
    """Trains one step, checkpoints via DCP and resumes into fresh model/optimizer instances,
    asserting that the Adam moments of every stage are restored.

    This covers the native state-dict branch of OptimizersList end to end. It is a regression test
    for the load direction in particular: DCP derives its read plan from the local state dict, so a
    freshly built optimizer that reports an empty state dict resumes with zeroed moments instead of
    the checkpointed ones.
    """
    _patch_distributed_ops(monkeypatch)
    trained_app_state = _build_app_state(num_stages=2)

    # Distinct gradients per stage, so that a mix-up between the stages' states would be visible.
    for stage_idx, model_part in enumerate(trained_app_state.model_parts, start=1):
        for parameter in model_part.parameters():
            parameter.grad = torch.full_like(parameter, float(stage_idx))
    trained_app_state.optimizer.step()
    trained_app_state.optimizer.zero_grad()

    trained_moments = {
        (stage_idx, sub_optimizer): _get_moments(trained_app_state.optimizer.state_dict(), stage_idx, sub_optimizer)
        for stage_idx in range(2)
        for sub_optimizer in ("ep_adamw", "dense_adamw")
    }
    assert all(moments for moments in trained_moments.values()), "Expected optimizer state after a step."

    checkpoint_path = tmp_path / "checkpoint"
    dcp.save(state_dict={"app": trained_app_state}, checkpoint_id=checkpoint_path)

    resumed_app_state = _build_app_state(num_stages=2)
    dcp.load(state_dict={"app": resumed_app_state}, checkpoint_id=checkpoint_path)
    resumed_state_dict = resumed_app_state.optimizer.state_dict()

    for (stage_idx, sub_optimizer), expected_moments in trained_moments.items():
        restored_moments = _get_moments(resumed_state_dict, stage_idx, sub_optimizer)
        assert set(restored_moments.keys()) == set(expected_moments.keys())
        for param_id, expected_param_state in expected_moments.items():
            for key, expected_value in expected_param_state.items():
                torch.testing.assert_close(
                    restored_moments[param_id][key],
                    expected_value,
                    msg=f"Mismatch for stage {stage_idx}, {sub_optimizer}, param {param_id}, {key}.",
                )

    # The model parameters must be restored as well (they are part of the same checkpoint).
    for trained_part, resumed_part in zip(trained_app_state.model_parts, resumed_app_state.model_parts):
        for trained_param, resumed_param in zip(trained_part.parameters(), resumed_part.parameters()):
            torch.testing.assert_close(resumed_param, trained_param)


def test_init_optimizer_state_materializes_state_without_changing_parameters():
    """The state materialization must be a no-op on the parameters themselves: it steps with a zero
    learning rate so that AdamW's decoupled weight decay cannot shrink them."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    assert not optimizer.state
    _init_optimizer_state_(optimizer)

    assert optimizer.state, "Expected materialized (zero) optimizer state."
    for parameter_before, parameter in zip(parameters_before, model.parameters()):
        torch.testing.assert_close(parameter, parameter_before)
        assert parameter.grad is None
    for param_state in optimizer.state.values():
        torch.testing.assert_close(param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"]))


def test_init_optimizer_state_skips_when_gradients_are_present():
    """Mid-training (gradients present) the zero-gradient step would not be a no-op, so it is skipped."""
    torch.manual_seed(0)
    model = nn.Linear(4, 4, bias=False)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    _init_optimizer_state_(optimizer)

    assert not optimizer.state
    for parameter in model.parameters():
        assert parameter.grad is not None
