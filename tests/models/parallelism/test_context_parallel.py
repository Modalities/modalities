import pytest
import torch

from modalities.models.parallelism.context_parallel import shard_tensor_buffers_for_context_parallel


class _DummyMesh:
    device_type = "cuda"

    def size(self) -> int:
        return 2


def test_shard_tensor_buffers_len_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="len\(buffers\) == len\(seq_dims\)"):
        shard_tensor_buffers_for_context_parallel(
            cp_mesh=_DummyMesh(),
            buffers=(torch.ones(2, 8),),
            seq_dims=(1, 1),
            shard_impl=lambda **_: (),
        )


def test_shard_tensor_buffers_ptrr_requires_masks_raises() -> None:
    with pytest.raises(ValueError, match="PTRR load balancing is not supported"):
        shard_tensor_buffers_for_context_parallel(
            cp_mesh=_DummyMesh(),
            buffers=(torch.ones(2, 8),),
            seq_dims=(1,),
            load_balancer_type="ptrr",
        )


def test_shard_tensor_buffers_uses_injected_impl() -> None:
    called = {}

    def fake_shard_impl(*, mesh, buffers, seq_dims, load_balancer):
        called["mesh"] = mesh
        called["seq_dims"] = seq_dims
        called["load_balancer"] = load_balancer
        return tuple(t[:, : t.shape[1] // 2].contiguous() for t in buffers)

    sample = torch.arange(16, dtype=torch.int64).view(2, 8)
    target = torch.arange(16, dtype=torch.int64).view(2, 8)

    sharded_sample, sharded_target = shard_tensor_buffers_for_context_parallel(
        cp_mesh=_DummyMesh(),
        buffers=(sample, target),
        seq_dims=(1, 1),
        load_balancer_type=None,
        shard_impl=fake_shard_impl,
    )

    assert called["seq_dims"] == (1, 1)
    assert called["load_balancer"] is None
    assert sharded_sample.shape == (2, 4)
    assert sharded_target.shape == (2, 4)
