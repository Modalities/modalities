import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from modalities.models.gpt2.gpt2_model import CausalSelfAttention
from modalities.models.model_factory import GPT2ModelFactory, ModelFactory
from modalities.models.parallelism.context_parallel import (
    apply_cp_to_sdpa_attention_forward,
    shard_tensor_buffers_for_context_parallel,
)


class _DummyMesh:
    device_type = "cuda"

    def size(self) -> int:
        return 2


# ── shard_tensor_buffers_for_context_parallel ────────────────────────────────


def test_shard_tensor_buffers_len_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"len\(buffers\) == len\(seq_dims\)"):
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


# ── apply_cp_to_sdpa_attention_forward ───────────────────────────────────────


def _get_execute_attention_fn() -> object:
    """Return the raw function behind the execute_attention classmethod descriptor.

    Classmethod descriptors return a new bound-method wrapper on every attribute
    access, so `is` comparisons on `CausalSelfAttention.execute_attention` always
    fail. Comparing the underlying `__func__` (or the descriptor stored in
    `__dict__`) gives a stable identity.
    """
    descriptor = CausalSelfAttention.__dict__["execute_attention"]
    return getattr(descriptor, "__func__", descriptor)


def _attention_module() -> CausalSelfAttention:
    module = object.__new__(CausalSelfAttention)
    nn.Module.__init__(module)
    return module


def test_apply_cp_empty_modules_does_not_patch() -> None:
    original_fn = _get_execute_attention_fn()
    apply_cp_to_sdpa_attention_forward(attention_modules=[], cp_mesh=object())
    assert _get_execute_attention_fn() is original_fn


def test_apply_cp_reentry_same_mesh_is_noop() -> None:
    mesh = object()
    module = _attention_module()
    setattr(module, "_cp_execute_attention_wrapped", True)
    setattr(module, "_cp_mesh", mesh)

    apply_cp_to_sdpa_attention_forward(attention_modules=[module], cp_mesh=mesh)

    assert "execute_attention" not in module.__dict__


def test_apply_cp_reentry_different_mesh_raises() -> None:
    mesh_a = object()
    mesh_b = object()
    module = _attention_module()
    setattr(module, "_cp_execute_attention_wrapped", True)
    setattr(module, "_cp_mesh", mesh_a)

    with pytest.raises(RuntimeError, match="already configured with a different cp_mesh"):
        apply_cp_to_sdpa_attention_forward(attention_modules=[module], cp_mesh=mesh_b)


def test_apply_cp_patches_only_supplied_instance() -> None:
    mock_dtensor_module = MagicMock()
    mock_attn_module = MagicMock()
    cp_module = _attention_module()
    non_cp_module = _attention_module()

    with patch.dict(
        sys.modules,
        {
            "torch.distributed.tensor": mock_dtensor_module,
            "torch.distributed.tensor.experimental": MagicMock(),
            "torch.distributed.tensor.experimental._attention": mock_attn_module,
        },
    ):
        original_fn = _get_execute_attention_fn()
        mesh = object()
        apply_cp_to_sdpa_attention_forward(attention_modules=[cp_module], cp_mesh=mesh)

    assert getattr(cp_module, "_cp_execute_attention_wrapped", False) is True
    assert getattr(cp_module, "_cp_mesh", None) is mesh
    assert "execute_attention" in cp_module.__dict__
    assert "execute_attention" not in non_cp_module.__dict__
    assert _get_execute_attention_fn() is original_fn
    mock_attn_module._enable_context_parallel_dispatcher.assert_called_once()


# ── GPT2ModelFactory._validate_context_parallel_seq_len ──────────────────────


def _model(seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(sequence_length=seq_len)


def test_validate_seq_len_headtail_valid() -> None:
    # 256 divisible by cp=2 * tp=1 * 2 = 4 ✓
    GPT2ModelFactory._validate_context_parallel_seq_len(_model(256), cp_degree=2, load_balancer_type="headtail")


def test_fsdp2_mesh_includes_cp_as_replication_dimension() -> None:
    mesh = SimpleNamespace(mesh_dim_names=("dp_shard", "cp"))

    assert ModelFactory._get_fsdp2_mesh_degrees(mesh) == ("cp", "dp_shard")


def test_ptrr_is_rejected_during_cp_model_construction() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        GPT2ModelFactory._apply_context_parallel_to_gpt2_attention(
            model=_model(256), cp_mesh=_DummyMesh(), context_parallel_load_balancer="ptrr"
        )


def test_validate_seq_len_headtail_invalid() -> None:
    # 100 % (3 * 2) = 100 % 6 ≠ 0 → should raise
    with pytest.raises(ValueError, match="divisible"):
        GPT2ModelFactory._validate_context_parallel_seq_len(_model(100), cp_degree=3, load_balancer_type="headtail")


def test_validate_seq_len_none_valid_odd_multiple() -> None:
    # 6 % (cp=2 * tp=1 * 1) = 0 → passes with None.
    # Would fail with headtail (6 % 4 ≠ 0) — this is the bug-fix regression case.
    GPT2ModelFactory._validate_context_parallel_seq_len(_model(6), cp_degree=2, load_balancer_type=None)


def test_validate_seq_len_none_invalid() -> None:
    # 9 % (cp=2) ≠ 0 → raises even without headtail factor
    with pytest.raises(ValueError, match="divisible"):
        GPT2ModelFactory._validate_context_parallel_seq_len(_model(9), cp_degree=2, load_balancer_type=None)


def test_validate_seq_len_headtail_with_tp_valid() -> None:
    # 256 % (tp=2 * cp=2 * 2) = 256 % 8 = 0 ✓
    GPT2ModelFactory._validate_context_parallel_seq_len(
        _model(256), cp_degree=2, tp_degree=2, load_balancer_type="headtail"
    )


def test_validate_seq_len_headtail_with_tp_invalid() -> None:
    # 256 % (tp=3 * cp=2 * 2) = 256 % 12 ≠ 0
    with pytest.raises(ValueError, match="divisible"):
        GPT2ModelFactory._validate_context_parallel_seq_len(
            _model(256), cp_degree=2, tp_degree=3, load_balancer_type="headtail"
        )


def test_validate_seq_len_none_with_tp_valid_headtail_would_fail() -> None:
    # 9 % (tp=3 * cp=3 * 1) = 0 → passes with None.
    # headtail would need 9 % (3*3*2=18) = 9 % 18 ≠ 0 → fails.
    GPT2ModelFactory._validate_context_parallel_seq_len(_model(9), cp_degree=3, tp_degree=1, load_balancer_type=None)
