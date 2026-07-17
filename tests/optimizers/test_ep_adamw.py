import torch
import torch.nn as nn

from modalities.models.model import NNModel
from modalities.optimizers.ep_adamw import EPAdamW


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


class TinyModel(NNModel):
    def __init__(self):
        super().__init__(weight_decay_groups={"linear": ["linear"], "embedding": [], "layernorm": ["norm"]})
        self.linear = nn.Linear(2, 2, bias=False)
        self.norm = nn.LayerNorm(2)
        self.experts = EPSubmodule()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = inputs["x"]
        return {"y": self.linear(x)}


def _patch_distributed_ops(monkeypatch):
    from modalities.optimizers import ep_adamw as ep_adamw_module

    monkeypatch.setattr(ep_adamw_module.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(ep_adamw_module.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(ep_adamw_module.dist, "all_reduce", lambda tensor, op=None, group=None: tensor)
    monkeypatch.setattr(ep_adamw_module.dist, "broadcast", lambda tensor, src=0, group=None: tensor)
    monkeypatch.setattr(ep_adamw_module.dist, "get_global_rank", lambda group, group_rank: group_rank)


def test_ep_adamw_state_dict_and_load_state_dict(monkeypatch):
    _patch_distributed_ops(monkeypatch)

    model = TinyModel()
    optimizer = EPAdamW(
        model=model,
        device_mesh={"dp_shard": DummyDPShardMesh()},
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        weight_decay_groups_excluded=["layernorm"],
    )

    state = optimizer.state_dict()
    assert "ep_adamw" in state
    assert "dense_adamw" in state

    optimizer.load_state_dict(state)


def test_ep_adamw_step_updates_parameters_and_zero_grad(monkeypatch):
    _patch_distributed_ops(monkeypatch)

    model = TinyModel()
    optimizer = EPAdamW(
        model=model,
        device_mesh={"dp_shard": DummyDPShardMesh()},
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        weight_decay_groups_excluded=["layernorm"],
    )

    before = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    optimizer.step()
    after = list(model.parameters())

    for p_before, p_after in zip(before, after):
        assert not torch.allclose(p_before, p_after)

    optimizer.zero_grad(set_to_none=True)
    for p in model.parameters():
        assert p.grad is None
