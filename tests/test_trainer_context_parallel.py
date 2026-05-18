from types import SimpleNamespace

import pytest
import torch

from modalities.batch import DatasetBatch
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.trainer import Trainer


class _DummyPublisher:
    def publish_message(self, payload):
        return None


class _DummyGradientClipper:
    def clip_gradients(self):
        return torch.tensor(0.0)


class _DummyProfiler:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _MeshEntry:
    def __init__(self, degree: int):
        self._degree = degree

    def size(self):
        return self._degree

    def get_coordinate(self):
        return [0]


class _DummyDeviceMesh:
    mesh_dim_names = [ParallelismDegrees.DP_SHARD.value, ParallelismDegrees.CP.value]

    def __getitem__(self, key: str):
        if key in (ParallelismDegrees.DP_SHARD.value, ParallelismDegrees.CP.value):
            return _MeshEntry(2)
        raise KeyError(key)

    def size(self, index: int):
        return 2


def test_trainer_passes_model_cp_load_balancer(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for context-parallel trainer sharding test")

    captured = {}

    def _fake_shard(*, cp_mesh, buffers, seq_dims, load_balancer_type, shard_impl=None):
        captured["load_balancer_type"] = load_balancer_type
        return buffers

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = Trainer(
        global_rank=0,
        progress_publisher=_DummyPublisher(),
        evaluation_result_publisher=_DummyPublisher(),
        gradient_acc_steps=1,
        global_num_tokens_per_train_step=1,
        device_mesh=_DummyDeviceMesh(),
        num_seen_train_steps=0,
        global_num_seen_tokens=0,
        num_target_steps=1,
        num_target_tokens=1,
        gradient_clipper=_DummyGradientClipper(),
        profiler=_DummyProfiler(),
    )

    model = SimpleNamespace(sample_key="input_ids", _context_parallel_load_balancer=None)
    loss_fun = SimpleNamespace(target_key="target_ids")

    batch = DatasetBatch(
        samples={"input_ids": torch.ones(1, 8, device=torch.cuda.current_device(), dtype=torch.long)},
        targets={"target_ids": torch.ones(1, 8, device=torch.cuda.current_device(), dtype=torch.long)},
    )

    trainer._apply_context_parallel_sharding_to_batch_(
        batch=batch,
        sample_key=model.sample_key,
        target_key=loss_fun.target_key,
        context_parallel_load_balancer=model._context_parallel_load_balancer,
    )

    assert "load_balancer_type" in captured
    assert captured["load_balancer_type"] is None
