# Portions of this file are adapted from NVIDIA's Megatron-LM
# (megatron/core/ssm/mamba_mixer.py): the packed [z, x, B, C, dt] input-projection layout, the
# conv1d / A_log / D / dt_bias parameter shapes, and their initialization distributions.
# Copyright (c) 2024, NVIDIA CORPORATION. Copyright (c) 2024, Tri Dao, Albert Gu.
# Licensed under the Apache License, Version 2.0.

"""Mamba-2 mixer, the sequence-mixing operator of hybrid Mamba-Transformer models.

The layout follows the Megatron-LM reference implementation
(``megatron/core/ssm/mamba_mixer.py``) so that parameter shapes and initialization match, which
keeps checkpoint conversion between the two frameworks tractable.
"""

import logging
import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from modalities.models.components.mamba2.ssd import GatedRMSNorm, causal_depthwise_conv1d, ssd_chunked_scan

logger = logging.getLogger(__name__)

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ModuleNotFoundError:
    mamba_chunk_scan_combined = None

try:
    from causal_conv1d import causal_conv1d_fn
except ModuleNotFoundError:
    causal_conv1d_fn = None


class SSDBackend(str, Enum):
    """
    Enum of the available state space scan implementations.

    Attributes:
        NATIVE (str): The dependency-free pure-PyTorch chunked scan. Runs on CPU and GPU and is
            ``torch.compile``-friendly, but noticeably slower and more memory hungry than the
            fused kernels.
        FUSED (str): The fused Triton kernels from ``mamba-ssm`` and ``causal-conv1d``. Requires
            the optional ``mamba`` extra and a CUDA device.
    """

    NATIVE = "native"
    FUSED = "fused"


# Above this model dimension the native backend becomes the throughput bottleneck and the fused
# kernels should be used instead. Chosen to flag real training runs without noise in unit tests.
_NATIVE_BACKEND_WARN_THRESHOLD_N_EMBD = 1024


def _local_view(parameter: torch.Tensor) -> torch.Tensor:
    """
    Returns the rank-local tensor backing a parameter.

    Under FSDP2 a parameter is a ``DTensor`` whose in-place ``copy_`` rejects a plain-tensor source.
    Writing into the local shard side-steps that while leaving unsharded models unchanged.

    Args:
        parameter (torch.Tensor): A parameter, possibly a ``DTensor``.

    Returns:
        torch.Tensor: The local shard, or the parameter itself if it is not distributed.
    """
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


class Mamba2Mixer(nn.Module):
    """
    Mamba-2 selective state space mixer.

    A single input projection produces five packed tensors ``[z, x, B, C, dt]``. The ``x``, ``B``
    and ``C`` parts pass through a short causal depthwise convolution, the selective scan mixes
    along the sequence, ``z`` gates the result via a grouped RMS norm, and an output projection
    maps back to the model dimension.
    """

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        head_dim: int,
        state_dim: int,
        n_groups: int,
        d_conv: int = 4,
        chunk_size: int = 128,
        ssd_backend: SSDBackend = SSDBackend.NATIVE,
        norm_eps: float = 1e-5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_init_range: tuple[float, float] = (1.0, 16.0),
        bias: bool = False,
        conv_bias: bool = True,
    ):
        """
        Initializes the Mamba2Mixer.

        Args:
            n_embd (int): The model dimension.
            n_heads (int): The number of SSM heads. Together with ``head_dim`` this determines the
                inner dimension ``d_inner = n_heads * head_dim``.
            head_dim (int): The dimension of a single SSM head.
            state_dim (int): The SSM state dimension.
            n_groups (int): The number of ``B``/``C`` groups. Must divide ``n_heads``.
            d_conv (int): The kernel size of the causal depthwise convolution.
            chunk_size (int): The chunk length of the selective scan.
            ssd_backend (SSDBackend): Which scan implementation to use.
            norm_eps (float): Epsilon of the gated RMS norm.
            dt_min (float): Lower bound of the initial ``dt`` range.
            dt_max (float): Upper bound of the initial ``dt`` range.
            dt_init_floor (float): Lower clamp applied to the sampled ``dt`` before inverting
                softplus.
            A_init_range (tuple[float, float]): Range from which the per-head decay rates are
                sampled uniformly before taking the logarithm.
            bias (bool): Whether the input and output projections use a bias.
            conv_bias (bool): Whether the depthwise convolution uses a bias.

        Raises:
            ValueError: If the head/group configuration is inconsistent, or if the fused backend
                is requested but ``mamba-ssm`` / ``causal-conv1d`` are not installed.
        """
        super().__init__()
        if n_heads % n_groups != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups}).")
        if d_conv < 1:
            raise ValueError(f"d_conv must be at least 1, got {d_conv}.")
        if A_init_range[0] <= 0 or A_init_range[1] < A_init_range[0]:
            raise ValueError(f"A_init_range must satisfy 0 < low <= high, got {A_init_range}.")

        ssd_backend = SSDBackend(ssd_backend)
        if ssd_backend == SSDBackend.FUSED:
            missing = [
                name
                for name, module in (("mamba-ssm", mamba_chunk_scan_combined), ("causal-conv1d", causal_conv1d_fn))
                if module is None
            ]
            if missing:
                raise ValueError(
                    f"ssd_backend='fused' requires {' and '.join(missing)} to be installed. "
                    "Install the optional extra via `pip install -e '.[mamba]'`, or use "
                    "ssd_backend='native'."
                )
        elif n_embd > _NATIVE_BACKEND_WARN_THRESHOLD_N_EMBD:
            logger.warning(
                "Mamba2Mixer is using the native (pure-PyTorch) SSD backend with n_embd=%d. "
                "This is correct but significantly slower than the fused kernels. Consider "
                "ssd_backend='fused' for large-scale training runs.",
                n_embd,
            )

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.state_dim = state_dim
        self.n_groups = n_groups
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.ssd_backend = ssd_backend
        self.d_inner = n_heads * head_dim
        self.conv_dim = self.d_inner + 2 * n_groups * state_dim
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.A_init_range = A_init_range

        # in_proj packs [z, x, B, C, dt] into a single matmul.
        self.in_proj = nn.Linear(
            in_features=n_embd,
            out_features=2 * self.d_inner + 2 * n_groups * state_dim + n_heads,
            bias=bias,
        )
        # The depthwise convolution is stored as raw parameters rather than an nn.Conv1d so that
        # the fused and native paths can share the exact same layout.
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, 1, d_conv))
        self.conv1d_bias = nn.Parameter(torch.empty(self.conv_dim)) if conv_bias else None
        # A_log and D are kept in fp32: they enter the scan through exp() and control its decay.
        self.A_log = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))
        self.D = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(n_heads))
        self.norm = GatedRMSNorm(hidden_size=self.d_inner, num_groups=n_groups, eps=norm_eps)
        self.out_proj = nn.Linear(in_features=self.d_inner, out_features=n_embd, bias=bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes the SSM-specific parameters.

        The parameters ``A_log``, ``D``, ``dt_bias`` and the convolution kernel are *not* normally
        distributed and must not be touched by the generic weight initialization. Modalities calls
        ``reset_parameters`` on every submodule before running the configured initializer, and the
        Nemotron parameter-name filters deliberately exclude these names, so this method is the
        single place that defines their distribution.

        The distributions match the Megatron-LM reference implementation:
            - ``A_log = log(U(A_init_range))``
            - ``dt_bias = softplus^-1(clamp(exp(U(log dt_min, log dt_max)), min=dt_init_floor))``
            - ``D = 1``
            - ``conv1d_weight`` / ``conv1d_bias``: the default nn.Conv1d initialization.
        """
        if self.in_proj.weight.device.type == "meta":
            # Nothing to initialize on the meta device; the model factory materializes the model
            # and calls reset_parameters again afterwards.
            return

        with torch.no_grad():
            # kaiming_uniform_/uniform_/ones_ are in-place ops that dispatch per shard and read the
            # global shape, so they are safe on both plain tensors and FSDP2 DTensors.
            nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))
            if self.conv1d_bias is not None:
                fan_in = self.conv1d_weight.size(1) * self.conv1d_weight.size(2)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(self.conv1d_bias, -bound, bound)
            nn.init.ones_(self.D)

            # dt_bias and A_log need values computed from a distribution that no nn.init helper
            # provides, so they are written explicitly. Under FSDP2 the parameters are DTensors and
            # copy_ rejects a plain-tensor source, so write into the local shard instead. The
            # values are i.i.d. per head, so filling each rank's shard independently is equivalent
            # to sampling the full tensor and slicing it.
            dt_bias_local = _local_view(self.dt_bias)
            A_log_local = _local_view(self.A_log)

            # Sample dt log-uniformly in [dt_min, dt_max], then invert softplus so that
            # softplus(dt_bias) reproduces the sampled dt.
            dt = torch.exp(
                torch.rand(dt_bias_local.shape, device=dt_bias_local.device)
                * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=self.dt_init_floor)
            dt_bias_local.copy_(dt + torch.log(-torch.expm1(-dt)))

            A = torch.empty(A_log_local.shape, device=A_log_local.device, dtype=torch.float32).uniform_(
                *self.A_init_range
            )
            A_log_local.copy_(torch.log(A).to(A_log_local.dtype))

    def _split_projection(
        self, zxbcdt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the packed input projection into its five components.

        Args:
            zxbcdt (torch.Tensor): Packed projection of shape ``(B, L, 2 * d_inner + 2 * G * N + H)``.

        Returns:
            tuple: ``(z, x, B, C, dt)`` with shapes ``(B, L, d_inner)``, ``(B, L, d_inner)``,
                ``(B, L, G, N)``, ``(B, L, G, N)`` and ``(B, L, H)``.
        """
        group_state_size = self.n_groups * self.state_dim
        z, xbc, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * group_state_size, self.n_heads],
            dim=-1,
        )
        # The convolution operates on the concatenated [x, B, C] block.
        xbc = self._apply_conv(xbc)
        x, b, c = torch.split(xbc, [self.d_inner, group_state_size, group_state_size], dim=-1)
        batch_size, seq_len = x.shape[:2]
        b = b.view(batch_size, seq_len, self.n_groups, self.state_dim)
        c = c.view(batch_size, seq_len, self.n_groups, self.state_dim)
        return z, x, b, c, dt

    def _apply_conv(self, xbc: torch.Tensor) -> torch.Tensor:
        """
        Applies the causal depthwise convolution followed by a SiLU activation.

        Args:
            xbc (torch.Tensor): The concatenated ``[x, B, C]`` block of shape ``(B, L, conv_dim)``.

        Returns:
            torch.Tensor: The activated convolution output of the same shape.
        """
        if self.ssd_backend == SSDBackend.FUSED and causal_conv1d_fn is not None and xbc.is_cuda:
            # causal_conv1d_fn expects (B, conv_dim, L) and folds the activation in.
            y = causal_conv1d_fn(
                x=xbc.transpose(1, 2).contiguous(),
                weight=self.conv1d_weight.squeeze(1),
                bias=self.conv1d_bias,
                activation="silu",
            )
            return y.transpose(1, 2)
        return F.silu(causal_depthwise_conv1d(xbc, weight=self.conv1d_weight, bias=self.conv1d_bias))

    def _run_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the selective state space scan with the configured backend.

        Args:
            x (torch.Tensor): SSM input of shape ``(B, L, H, P)``.
            dt (torch.Tensor): Positive discretization step of shape ``(B, L, H)``.
            b (torch.Tensor): Input projection of shape ``(B, L, G, N)``.
            c (torch.Tensor): Output projection of shape ``(B, L, G, N)``.

        Returns:
            torch.Tensor: The scan output of shape ``(B, L, H, P)``.
        """
        A = -torch.exp(self.A_log.float())
        if self.ssd_backend == SSDBackend.FUSED and mamba_chunk_scan_combined is not None and x.is_cuda:
            return mamba_chunk_scan_combined(
                x,
                dt,
                A,
                b,
                c,
                chunk_size=self.chunk_size,
                D=self.D.float(),
                z=None,
                dt_bias=None,
                dt_softplus=False,
            )
        return ssd_chunked_scan(
            x=x,
            dt=dt,
            A=A,
            B=b,
            C=c,
            D=self.D,
            chunk_size=self.chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Mamba-2 mixer.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, n_embd)``.

        Returns:
            torch.Tensor: Output of shape ``(B, L, n_embd)``.
        """
        batch_size, seq_len, _ = x.shape
        z, x_ssm, b, c, dt = self._split_projection(self.in_proj(x))

        # softplus keeps dt strictly positive; dt_bias sets its initial magnitude per head.
        dt = F.softplus(dt.float() + self.dt_bias.float())
        x_heads = x_ssm.view(batch_size, seq_len, self.n_heads, self.head_dim)

        y = self._run_scan(x=x_heads, dt=dt, b=b, c=c)
        y = y.reshape(batch_size, seq_len, self.d_inner)
        y = self.norm(y, gate=z)
        return self.out_proj(y)
