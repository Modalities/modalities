import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from modalities.models.gpt2.gpt2_model import CausalSelfAttention
from modalities.models.model_factory import GPT2ModelFactory
from modalities.models.parallelism.context_parallel import (
    apply_cp_to_sdpa_attention_forward,
    shard_tensor_buffers_for_context_parallel,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_UNSET = object()


class _DummyMesh:
    device_type = "cuda"

    def size(self) -> int:
        return 2


# ── fixture: isolate class-level CP patch across tests ───────────────────────


@pytest.fixture()
def cp_class_patch_isolated():
    """Restore CausalSelfAttention to its exact pre-test state after any test
    that touches (or simulates) the class-level CP patch."""
    original_method = CausalSelfAttention.execute_attention
    saved_wrapped = getattr(CausalSelfAttention, "_cp_execute_attention_wrapped", _UNSET)
    saved_mesh = getattr(CausalSelfAttention, "_cp_mesh", _UNSET)
    yield
    CausalSelfAttention.execute_attention = original_method
    for attr, saved in [("_cp_execute_attention_wrapped", saved_wrapped), ("_cp_mesh", saved_mesh)]:
        if saved is _UNSET:
            if hasattr(CausalSelfAttention, attr):
                delattr(CausalSelfAttention, attr)
        else:
            setattr(CausalSelfAttention, attr, saved)


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


def test_apply_cp_empty_modules_does_not_patch(cp_class_patch_isolated) -> None:
    """An empty attention_modules list must leave the class untouched."""
    original_fn = _get_execute_attention_fn()
    apply_cp_to_sdpa_attention_forward(attention_modules=[], cp_mesh=object())
    assert _get_execute_attention_fn() is original_fn
    assert not getattr(CausalSelfAttention, "_cp_execute_attention_wrapped", False)


def test_apply_cp_reentry_same_mesh_is_noop(cp_class_patch_isolated) -> None:
    """A second call with the same mesh object must return silently."""
    mesh = object()
    setattr(CausalSelfAttention, "_cp_execute_attention_wrapped", True)
    setattr(CausalSelfAttention, "_cp_mesh", mesh)
    original_fn = _get_execute_attention_fn()

    apply_cp_to_sdpa_attention_forward(attention_modules=[MagicMock(spec=nn.Module)], cp_mesh=mesh)

    assert _get_execute_attention_fn() is original_fn


def test_apply_cp_reentry_different_mesh_raises(cp_class_patch_isolated) -> None:
    """A second call with a different mesh must raise RuntimeError."""
    mesh_a = object()
    mesh_b = object()
    setattr(CausalSelfAttention, "_cp_execute_attention_wrapped", True)
    setattr(CausalSelfAttention, "_cp_mesh", mesh_a)

    with pytest.raises(RuntimeError, match="already patched.*different cp_mesh"):
        apply_cp_to_sdpa_attention_forward(attention_modules=[MagicMock(spec=nn.Module)], cp_mesh=mesh_b)


def test_apply_cp_patches_classmethod_and_sets_flags(cp_class_patch_isolated) -> None:
    """First call must replace execute_attention and set both guard flags."""
    mock_dtensor_module = MagicMock()
    mock_attn_module = MagicMock()

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
        apply_cp_to_sdpa_attention_forward(attention_modules=[MagicMock(spec=nn.Module)], cp_mesh=mesh)

    assert getattr(CausalSelfAttention, "_cp_execute_attention_wrapped", False) is True
    assert getattr(CausalSelfAttention, "_cp_mesh", None) is mesh
    assert _get_execute_attention_fn() is not original_fn
    mock_attn_module._enable_context_parallel_dispatcher.assert_called_once()


# ── GPT2ModelFactory._validate_context_parallel_seq_len ──────────────────────


def _model(seq_len: int) -> SimpleNamespace:
    return SimpleNamespace(sequence_length=seq_len)


def test_validate_seq_len_headtail_valid() -> None:
    # 256 divisible by cp=2 * tp=1 * 2 = 4 ✓
    GPT2ModelFactory._validate_context_parallel_seq_len(_model(256), cp_degree=2, load_balancer_type="headtail")


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
