# Portions of this file are adapted from the Mamba-2 reference implementation in
# state-spaces/mamba (https://github.com/state-spaces/mamba): the chunk-parallel SSD block
# decomposition follows `ssd_minimal_discrete` / `segsum`, and GatedRMSNorm reproduces the
# semantics of `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn` with norm_before_gate=False.
# Copyright (c) 2024, Tri Dao, Albert Gu. Licensed under the Apache License, Version 2.0.

"""Pure-PyTorch Mamba-2 primitives (state space dual / SSD).

This module provides a dependency-free, CPU-runnable and ``torch.compile``-friendly
implementation of the three building blocks a Mamba-2 mixer needs:

1. :func:`ssd_chunked_scan` - the chunk-parallel selective state space scan.
2. :func:`causal_depthwise_conv1d` - the short causal depthwise convolution.
3. :class:`GatedRMSNorm` - the grouped, gated RMS normalization applied to the SSM output.

The chunked scan follows the block-decomposition of the Mamba-2 paper (Dao & Gu, 2024) and is
numerically equivalent (up to floating point accumulation order) to the fused Triton kernel
``mamba_split_conv1d_scan_combined`` used by the Megatron-LM reference implementation. It is the
default backend so that Modalities does not take a hard dependency on ``mamba-ssm``; see
:class:`modalities.models.components.mamba2.mamba2_mixer.SSDBackend` for the fused alternative.

Notation used throughout:
    B: batch size, L: sequence length, H: number of SSM heads, P: SSM head dimension,
    G: number of B/C groups, N: SSM state dimension, C: number of chunks, S: chunk length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _segment_sum(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise segment sums of a sequence of log-decay values.

    For an input of shape ``(..., T)`` this returns a tensor of shape ``(..., T, T)`` where
    entry ``[..., i, j]`` holds ``sum(x[..., j + 1 : i + 1])`` for ``i >= j`` and ``-inf``
    otherwise. Exponentiating the result yields the causal decay matrix of the SSM, and doing
    the accumulation in log space is what keeps long chunks numerically stable.

    Args:
        x (torch.Tensor): Log-decay values of shape ``(..., T)``.

    Returns:
        torch.Tensor: Segment sums of shape ``(..., T, T)``.
    """
    seq_len = x.size(-1)
    # Replicate along a new trailing axis so that entry [..., i, j] starts out as x[..., i].
    x_expanded = x.unsqueeze(-1).expand(*x.shape, seq_len)
    strictly_lower = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=-1)
    x_expanded = x_expanded.masked_fill(~strictly_lower, 0)
    # Accumulating over i turns entry [..., i, j] into sum(x[..., j + 1 : i + 1]).
    segment_sums = torch.cumsum(x_expanded, dim=-2)
    lower = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)
    return segment_sums.masked_fill(~lower, -torch.inf)


def _expand_groups_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Expands a per-group tensor to a per-head tensor.

    Mamba-2 shares the ``B`` and ``C`` projections across all heads of a group (the SSM analogue
    of grouped-query attention). Every group covers ``num_heads // num_groups`` consecutive heads.

    Args:
        x (torch.Tensor): Per-group tensor of shape ``(B, L, G, N)``.
        num_heads (int): The number of heads to expand to.

    Raises:
        ValueError: If the number of heads is not divisible by the number of groups.

    Returns:
        torch.Tensor: Per-head tensor of shape ``(B, L, H, N)``.
    """
    num_groups = x.size(2)
    if num_heads % num_groups != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_groups ({num_groups}).")
    heads_per_group = num_heads // num_groups
    if heads_per_group == 1:
        return x
    return x.repeat_interleave(heads_per_group, dim=2)


def ssd_chunked_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    chunk_size: int = 128,
    initial_states: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the chunk-parallel Mamba-2 selective state space scan.

    Implements the recurrence, for head ``h`` with associated group ``g``::

        state_t = exp(dt_t * A_h) * state_{t-1} + dt_t * outer(B_t^g, x_t)
        y_t     = C_t^g @ state_t + D_h * x_t

    The sequence is split into chunks of ``chunk_size``. Within a chunk the recurrence is
    evaluated as a masked matrix multiplication (the "quadratic"/attention-like form), while
    across chunks only the compact ``(P, N)`` chunk states are passed. That is the duality the
    Mamba-2 paper exploits, and it is what makes the scan efficient without a custom kernel.

    The scan is internally computed in float32 regardless of the input dtype, because the
    ``cumsum`` over log-decays and the state accumulation are the numerically sensitive parts.
    The result is cast back to the dtype of ``x``.

    Args:
        x (torch.Tensor): SSM input of shape ``(B, L, H, P)``.
        dt (torch.Tensor): Positive discretization step of shape ``(B, L, H)``, i.e. the value
            after applying softplus to the projected ``dt`` plus its bias.
        A (torch.Tensor): Negative per-head decay rate of shape ``(H,)``, i.e. ``-exp(A_log)``.
        B (torch.Tensor): Input projection of shape ``(B, L, G, N)``.
        C (torch.Tensor): Output projection of shape ``(B, L, G, N)``.
        D (torch.Tensor | None): Optional per-head skip connection of shape ``(H,)``.
        chunk_size (int): Chunk length used by the block decomposition. Must be positive. The
            result is mathematically independent of this value; it only trades off compute
            against memory.
        initial_states (torch.Tensor | None): Optional initial SSM state of shape
            ``(B, H, P, N)``. Defaults to zeros.
        return_final_state (bool): Whether to additionally return the SSM state after the last
            token, of shape ``(B, H, P, N)``.

    Raises:
        ValueError: If ``chunk_size`` is not positive or the input shapes are inconsistent.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The output of shape ``(B, L, H, P)``,
            and the final state if ``return_final_state`` is set.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B, L, H, P), got {tuple(x.shape)}.")
    if B.shape != C.shape:
        raise ValueError(f"B and C must have the same shape, got {tuple(B.shape)} and {tuple(C.shape)}.")

    batch_size, seq_len, num_heads, head_dim = x.shape
    out_dtype = x.dtype

    # The scan is accumulation-heavy; run it in fp32 and cast back at the end.
    x_f = x.float()
    dt_f = dt.float()
    A_f = A.float()
    B_f = _expand_groups_to_heads(B.float(), num_heads=num_heads)
    C_f = _expand_groups_to_heads(C.float(), num_heads=num_heads)

    # Zero-pad the sequence up to a multiple of chunk_size. Appending zeros at the end is safe
    # for a causal scan: padded tokens cannot influence the outputs of real tokens, and their
    # own outputs are discarded. dt = 0 additionally makes the padded steps state-preserving.
    pad_len = (-seq_len) % chunk_size
    if pad_len > 0:
        x_f = F.pad(x_f, (0, 0, 0, 0, 0, pad_len))
        dt_f = F.pad(dt_f, (0, 0, 0, pad_len))
        B_f = F.pad(B_f, (0, 0, 0, 0, 0, pad_len))
        C_f = F.pad(C_f, (0, 0, 0, 0, 0, pad_len))
    padded_len = seq_len + pad_len
    num_chunks = padded_len // chunk_size

    # Fold dt into the input and the decay rate, matching the discretized formulation.
    x_scaled = x_f * dt_f.unsqueeze(-1)  # (B, L, H, P)
    log_decay = A_f * dt_f  # (B, L, H), non-positive

    # Reshape into chunks: (B, C, S, H, ...)
    def to_chunks(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch_size, num_chunks, chunk_size, *tensor.shape[2:])

    x_chunked = to_chunks(x_scaled)  # (B, C, S, H, P)
    B_chunked = to_chunks(B_f)  # (B, C, S, H, N)
    C_chunked = to_chunks(C_f)  # (B, C, S, H, N)
    # (B, C, S, H) -> (B, H, C, S) so that the decay cumsum runs over the chunk axis.
    log_decay_chunked = to_chunks(log_decay).permute(0, 3, 1, 2)  # (B, H, C, S)
    log_decay_cumsum = torch.cumsum(log_decay_chunked, dim=-1)  # (B, H, C, S)

    # 1. Intra-chunk (diagonal blocks): evaluate the recurrence in its quadratic form.
    intra_chunk_decay = torch.exp(_segment_sum(log_decay_chunked))  # (B, H, C, S, S)
    y_intra = torch.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp", C_chunked, B_chunked, intra_chunk_decay, x_chunked
    )  # (B, C, S, H, P)

    # 2. Per-chunk end states: decay each position forward to the chunk boundary.
    decay_to_chunk_end = torch.exp(log_decay_cumsum[..., -1:] - log_decay_cumsum)  # (B, H, C, S)
    chunk_states = torch.einsum("bcshn,bhcs,bcshp->bchpn", B_chunked, decay_to_chunk_end, x_chunked)  # (B, C, H, P, N)

    # 3. Inter-chunk recurrence over the compact chunk states.
    if initial_states is None:
        leading_state = torch.zeros_like(chunk_states[:, :1])  # (B, 1, H, P, N)
    else:
        leading_state = initial_states.float().unsqueeze(1)  # (B, 1, H, P, N)
    all_states = torch.cat([leading_state, chunk_states], dim=1)  # (B, C+1, H, P, N)
    # Pad with a leading zero so that chunk 0 receives the (undecayed) initial state.
    chunk_boundary_decay = torch.exp(_segment_sum(F.pad(log_decay_cumsum[..., -1], (1, 0))))  # (B, H, C+1, C+1)
    propagated_states = torch.einsum("bhzc,bchpn->bzhpn", chunk_boundary_decay, all_states)  # (B, C+1, H, P, N)
    chunk_start_states = propagated_states[:, :-1]  # (B, C, H, P, N)

    # 4. Inter-chunk (off-diagonal blocks): read the incoming chunk state out at every position.
    decay_from_chunk_start = torch.exp(log_decay_cumsum)  # (B, H, C, S)
    y_inter = torch.einsum(
        "bclhn,bchpn,bhcl->bclhp", C_chunked, chunk_start_states, decay_from_chunk_start
    )  # (B, C, S, H, P)

    y = (y_intra + y_inter).reshape(batch_size, padded_len, num_heads, head_dim)
    if pad_len > 0:
        y = y[:, :seq_len]

    if D is not None:
        y = y + D.float().view(1, 1, num_heads, 1) * x_f[:, :seq_len]

    y = y.to(out_dtype)
    if return_final_state:
        return y, propagated_states[:, -1].to(out_dtype)
    return y


def ssd_recurrent_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Evaluates the Mamba-2 recurrence step by step, as a correctness reference.

    This is the literal transcription of the SSM recurrence and is intentionally slow. It exists
    so that :func:`ssd_chunked_scan` can be validated against a definition that involves no
    block decomposition, no chunking and no log-space tricks.

    Args:
        x (torch.Tensor): SSM input of shape ``(B, L, H, P)``.
        dt (torch.Tensor): Positive discretization step of shape ``(B, L, H)``.
        A (torch.Tensor): Negative per-head decay rate of shape ``(H,)``.
        B (torch.Tensor): Input projection of shape ``(B, L, G, N)``.
        C (torch.Tensor): Output projection of shape ``(B, L, G, N)``.
        D (torch.Tensor | None): Optional per-head skip connection of shape ``(H,)``.

    Returns:
        torch.Tensor: The output of shape ``(B, L, H, P)``.
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    x_f, dt_f, A_f = x.float(), dt.float(), A.float()
    B_f = _expand_groups_to_heads(B.float(), num_heads=num_heads)
    C_f = _expand_groups_to_heads(C.float(), num_heads=num_heads)
    state_dim = B_f.size(-1)

    state = torch.zeros(batch_size, num_heads, head_dim, state_dim, device=x.device, dtype=torch.float32)
    outputs = []
    for t in range(seq_len):
        decay = torch.exp(dt_f[:, t] * A_f).view(batch_size, num_heads, 1, 1)  # (B, H, 1, 1)
        update = torch.einsum("bhp,bhn->bhpn", x_f[:, t] * dt_f[:, t].unsqueeze(-1), B_f[:, t])
        state = decay * state + update
        outputs.append(torch.einsum("bhpn,bhn->bhp", state, C_f[:, t]))
    y = torch.stack(outputs, dim=1)  # (B, L, H, P)

    if D is not None:
        y = y + D.float().view(1, 1, num_heads, 1) * x_f
    return y.to(x.dtype)


def causal_depthwise_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """
    Applies a causal depthwise 1D convolution along the sequence dimension.

    Causality is achieved by left-padding with ``kernel_size - 1`` zeros so that position ``t``
    only ever sees positions ``<= t``.

    Args:
        x (torch.Tensor): Input of shape ``(B, L, channels)``.
        weight (torch.Tensor): Depthwise kernel of shape ``(channels, 1, kernel_size)``.
        bias (torch.Tensor | None): Optional bias of shape ``(channels,)``.

    Returns:
        torch.Tensor: Output of shape ``(B, L, channels)``.
    """
    kernel_size = weight.size(-1)
    channels = weight.size(0)
    # conv1d expects (B, channels, L).
    x_channels_first = x.transpose(1, 2)
    x_padded = F.pad(x_channels_first, (kernel_size - 1, 0))
    y = F.conv1d(x_padded, weight=weight, bias=bias, groups=channels)
    return y.transpose(1, 2)


class GatedRMSNorm(nn.Module):
    """
    Grouped RMS normalization with a SiLU gate, as used inside the Mamba-2 mixer.

    The gate is applied *before* normalization (``norm_before_gate=False`` in the reference
    implementation), and the normalization statistics are computed per group rather than over the
    full inner dimension. Grouping matters because the SSM inner dimension is organized into
    ``num_groups`` blocks whose scales can differ substantially.
    """

    def __init__(self, hidden_size: int, num_groups: int = 1, eps: float = 1e-5):
        """
        Initializes the GatedRMSNorm module.

        Args:
            hidden_size (int): The size of the normalized dimension (the SSM inner dimension).
            num_groups (int): The number of groups to normalize independently. Must divide
                ``hidden_size``.
            eps (float): Value added to the mean square for numerical stability.

        Raises:
            ValueError: If ``hidden_size`` is not divisible by ``num_groups``.
        """
        super().__init__()
        if hidden_size % num_groups != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_groups ({num_groups}).")
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def reset_parameters(self) -> None:
        """Resets the learnable scale to one. Called by the Modalities model initialization."""
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Applies the gate and the grouped RMS normalization.

        Args:
            x (torch.Tensor): Input of shape ``(..., hidden_size)``.
            gate (torch.Tensor): Gate of the same shape as ``x``.

        Returns:
            torch.Tensor: Normalized output of the same shape and dtype as ``x``.
        """
        out_dtype = x.dtype
        gated = x.float() * F.silu(gate.float())
        grouped = gated.reshape(*gated.shape[:-1], self.num_groups, self.group_size)
        inv_rms = torch.rsqrt(grouped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (grouped * inv_rms).reshape(*gated.shape)
        return (normalized * self.weight.float()).to(out_dtype)
