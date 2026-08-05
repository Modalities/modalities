import pytest
import torch

from modalities.batch import DatasetBatch
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees
from modalities.trainer import Trainer


class _DummyPublisher:
    def publish_message(self, **_kwargs):
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


class _DummyDeviceMeshCPDegreeOne:
    """Device mesh that has a CP dimension, but with degree 1 (effectively disabled)."""

    mesh_dim_names = [ParallelismDegrees.DP_SHARD.value, ParallelismDegrees.CP.value]

    def __getitem__(self, key: str):
        if key == ParallelismDegrees.DP_SHARD.value:
            return _MeshEntry(2)
        if key == ParallelismDegrees.CP.value:
            return _MeshEntry(1)
        raise KeyError(key)

    def size(self, index: int):
        return 2


class _DummyDeviceMeshNoCP:
    """Device mesh that has no CP dimension at all."""

    mesh_dim_names = [ParallelismDegrees.DP_SHARD.value]

    def __getitem__(self, key: str):
        if key == ParallelismDegrees.DP_SHARD.value:
            return _MeshEntry(2)
        raise KeyError(key)

    def size(self, index: int):
        return 2


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_trainer(device_mesh) -> Trainer:
    return Trainer(
        global_rank=0,
        progress_publisher=_DummyPublisher(),
        evaluation_result_publisher=_DummyPublisher(),
        gradient_acc_steps=1,
        global_num_tokens_per_train_step=1,
        device_mesh=device_mesh,
        num_seen_train_steps=0,
        global_num_seen_tokens=0,
        num_target_steps=1,
        num_target_tokens=1,
        gradient_clipper=_DummyGradientClipper(),
        profiler=_DummyProfiler(),
    )


def _cuda_batch(seq_len: int = 8) -> DatasetBatch:
    dev = torch.cuda.current_device()
    return DatasetBatch(
        samples={"input_ids": torch.ones(1, seq_len, device=dev, dtype=torch.long)},
        targets={"target_ids": torch.ones(1, seq_len, device=dev, dtype=torch.long)},
    )


# ── CP sharding: load-balancer forwarding ────────────────────────────────────


def test_trainer_passes_model_cp_load_balancer_none(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for context-parallel trainer sharding test")

    captured = {}

    def _fake_shard(*, cp_mesh, buffers, seq_dims, load_balancer_type, shard_impl=None):
        captured["load_balancer_type"] = load_balancer_type
        return buffers

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMesh())
    batch = _cuda_batch()
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=batch, sample_key="input_ids", target_key="target_ids", context_parallel_load_balancer=None
    )

    assert captured.get("load_balancer_type") is None


def test_trainer_passes_model_cp_load_balancer_headtail(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for context-parallel trainer sharding test")

    captured = {}

    def _fake_shard(*, cp_mesh, buffers, seq_dims, load_balancer_type, shard_impl=None):
        captured["load_balancer_type"] = load_balancer_type
        return buffers

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMesh())
    batch = _cuda_batch()
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=batch, sample_key="input_ids", target_key="target_ids", context_parallel_load_balancer="headtail"
    )

    assert captured.get("load_balancer_type") == "headtail"


# ── CP sharding: selective buffer inclusion ───────────────────────────────────


def test_cp_sharding_with_sample_key_none_only_shards_target(monkeypatch):
    """When sample_key is None (e.g. non-first PP stage), only the target enters the shard call."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    captured = {}

    def _fake_shard(*, cp_mesh, buffers, seq_dims, load_balancer_type, shard_impl=None):
        captured["num_buffers"] = len(buffers)
        return buffers

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMesh())
    batch = _cuda_batch()
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=batch, sample_key=None, target_key="target_ids", context_parallel_load_balancer=None
    )

    assert captured.get("num_buffers") == 1


def test_cp_sharding_with_sample_key_includes_position_ids(monkeypatch):
    """When sample_key is provided, position_ids and the sample both enter the shard call,
    and position_ids is written back to batch.samples."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    captured = {}

    def _fake_shard(*, cp_mesh, buffers, seq_dims, load_balancer_type, shard_impl=None):
        captured["num_buffers"] = len(buffers)
        return buffers

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMesh())
    batch = _cuda_batch(seq_len=8)
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=batch, sample_key="input_ids", target_key="target_ids", context_parallel_load_balancer=None
    )

    # position_ids + input_ids + target_ids
    assert captured.get("num_buffers") == 3
    assert "position_ids" in batch.samples
    assert batch.samples["position_ids"].shape == (1, 8)


# ── CP sharding: early-exit conditions ───────────────────────────────────────


def test_cp_sharding_skipped_when_cp_mesh_degree_one(monkeypatch):
    """cp_mesh.size() == 1 must result in no sharding call."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    called = {"count": 0}

    def _fake_shard(**_kwargs):
        called["count"] += 1
        return _kwargs["buffers"]

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMeshCPDegreeOne())
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=_cuda_batch(), sample_key="input_ids", target_key="target_ids", context_parallel_load_balancer=None
    )

    assert called["count"] == 0


def test_cp_sharding_skipped_when_no_cp_in_mesh(monkeypatch):
    """A mesh with no CP dimension must skip sharding entirely."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    called = {"count": 0}

    def _fake_shard(**_kwargs):
        called["count"] += 1
        return _kwargs["buffers"]

    monkeypatch.setattr("modalities.trainer.shard_tensor_buffers_for_context_parallel", _fake_shard)

    trainer = _make_trainer(_DummyDeviceMeshNoCP())
    trainer._apply_context_parallel_sharding_to_batch_(
        batch=_cuda_batch(), sample_key="input_ids", target_key="target_ids", context_parallel_load_balancer=None
    )

    assert called["count"] == 0
