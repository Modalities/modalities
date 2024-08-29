# Copyright (c) 2023, Tri Dao.
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.cuda.amp import custom_bwd, custom_fwd


def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    """
    Apply layer normalization to the input tensor `x`.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        weight (torch.Tensor): The weight tensor for the layer normalization.
        bias (torch.Tensor): The bias tensor for the layer normalization.
        residual (torch.Tensor, optional): The residual tensor to be added to the input tensor. Default is None.
        eps (float, optional): A value added to the denominator for numerical stability. Default is 1e-6.
        prenorm (bool, optional): If True, return a tuple of normalized tensor and the input tensor. Default is False.
        upcast (bool, optional): If True, upcast the input tensor, weight, and bias to float. Default is False.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: The normalized tensor if `prenorm` is False,
        otherwise a tuple of normalized tensor and the input tensor.
    """
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)
    return out if not prenorm else (out, x)


def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    """
    Applies RMS normalization to the input tensor `x` using the provided `weight` and `bias` parameters.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        weight (torch.Tensor): The weight tensor to be applied to the normalized input.
        bias (torch.Tensor, optional): The bias tensor to be added to the normalized input. Default is None.
        residual (torch.Tensor, optional): The residual tensor to be added to the input before normalization.
        Default is None.
        eps (float, optional): A small value added to the denominator for numerical stability.
        Default is 1e-6.
        prenorm (bool, optional): If True, returns a tuple containing the normalized output and the input tensor.
        Default is False.
        upcast (bool, optional): If True, upcasts the weight, bias, and input tensors to float before normalization.
        Default is False.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: The normalized output tensor.
        If `prenorm` is True, returns a tuple containing the normalized output and the input tensor.
    """
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    RESIDUAL,
    RESIDUAL_OUT,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    N,
    eps,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Compute layer normalization forward pass using a single kernel.

    Args:
        X: Pointer to the input.
        Y: Pointer to the output.
        W: Pointer to the weights.
        B: Pointer to the biases.
        RESIDUAL: Pointer to the residual.
        RESIDUAL_OUT: Pointer to the residual.
        Mean: Pointer to the mean.
        Rstd: Pointer to the 1/std.
        stride_x_row: Defines, how much to increase the pointer when moving by 1 row.
        stride_y_row: Defines, how much to increase the pointer when moving by 1 row.
        stride_res_row: How much to increase the pointer when moving by 1 row.
        stride_res_out_row: How much to increase the pointer when moving by 1 row.
        N: Number of columns in X.
        eps: Epsilon to avoid division by zero.
        IS_RMS_NORM: Boolean indicating whether it is RMS normalization.
        BLOCK_N: Constant expression indicating the block size.
        HAS_RESIDUAL: Constant expression indicating whether it has residual.
        STORE_RESIDUAL_OUT: Constant expression indicating whether to store residual out.
        HAS_BIAS: Constant expression indicating whether it has bias.

    Returns:
        None
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False):
    """
    Applies layer normalization to the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor.
        bias (torch.Tensor): The bias tensor.
        eps (float): A small value added to the denominator for numerical stability.
        residual (torch.Tensor, optional): The residual tensor. Defaults to None.
        out_dtype (torch.dtype, optional): The output tensor dtype. Defaults to None.
        residual_dtype (torch.dtype, optional): The residual tensor dtype. Defaults to None.
        is_rms_norm (bool, optional): Whether to perform RMS normalization. Defaults to False.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    """

    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            residual_out,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            N,
            eps,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            bias is not None,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,
    W,
    B,
    Y,
    DY,
    DX,
    DW,
    DB,
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dres_in_row,
    M,
    N,
    eps,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    """
    Compute the backward pass of the layer normalization operation.

    Args:
        X: Pointer to the input.
        W: Pointer to the weights.
        B: Pointer to the biases.
        Y: Pointer to the output to be recomputed.
        DY: Pointer to the output gradient.
        DX: Pointer to the input gradient.
        DW: Pointer to the partial sum of weights gradient.
        DB: Pointer to the partial sum of biases gradient.
        DRESIDUAL: -
        DRESIDUAL_IN: -
        Mean: Pointer to the mean.
        Rstd: Pointer to the 1/std.
        stride_x_row: Defines, how much to increase the pointer when moving by 1 row.
        stride_y_row: -
        stride_dy_row: -
        stride_dx_row: -
        stride_dres_row: -
        stride_dres_in_row: -
        M: Number of rows in X.
        N: Number of columns in X.
        eps: Epsilon to avoid division by zero.
        rows_per_program: -
        IS_RMS_NORM: Whether it is RMS normalization or not.
        BLOCK_N: -
        HAS_DRESIDUAL: Whether it has residual or not.
        STORE_DRESIDUAL: Whether to store residual or not.
        HAS_BIAS: Whether it has bias or not.
        RECOMPUTE_OUTPUT: Whether to recompute output or not.

        Returns:
            None
    """
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    dresidual=None,
    has_residual=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    """
    Backward pass for the layer normalization operation.

    Args:
        dy (torch.Tensor): The gradient of the output tensor with respect to the loss.
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor.
        bias (torch.Tensor): The bias tensor.
        eps (float): The epsilon value for numerical stability.
        mean (torch.Tensor): The mean tensor computed during the forward pass.
        rstd (torch.Tensor): The reciprocal standard deviation tensor computed during the forward pass.
        dresidual (torch.Tensor, optional): The gradient of the residual tensor with respect to the loss.
        Defaults to None.
        has_residual (bool, optional): Indicates whether the residual tensor is present.
        Defaults to False.
        is_rms_norm (bool, optional): Indicates whether the operation is RMS normalization.
        Defaults to False.
        x_dtype (torch.dtype, optional): The data type of the input tensor.
        Defaults to None.
        recompute_output (bool, optional): Indicates whether to recompute the output tensor.
        Defaults to False.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((sm_count, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            dresidual_in,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            eps,
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            bias is not None,
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormFn(torch.autograd.Function):
    """LayerNormFn class."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        """
        Forward pass of the LayerNorm module.

        Args:
            ctx: -.
            x (torch.Tensor): Input tensor.
            weight (torch.Tensor): Weight tensor.
            bias (torch.Tensor): Bias tensor.
            residual (torch.Tensor, optional): Residual tensor. Defaults to None.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
            prenorm (bool, optional): Flag indicating whether to apply pre-normalization. Defaults to False.
            residual_in_fp32 (bool, optional): Flag indicating whether the residual tensor is in FP32 format.
            Defaults to False.
            is_rms_norm (bool, optional): Flag indicating whether to apply RMS normalization.
            Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            Output tensor or tuple of output tensor and residual tensor.
        """
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
        y, mean, rstd, residual_out = _layer_norm_fwd(
            x, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm
        )
        ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    def backward(ctx, dy, *args):
        """
        Backward pass of the layer normalization operation.

        Args:
            ctx: -.
            dy (torch.Tensor): -.
            *args: Additional arguments passed to the backward function.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:
            - dx (torch.Tensor): Tensor representing the gradient of the input with respect to the forward output.
            - dw (torch.Tensor): Tensor representing the gradient of the weight with respect to the forward output.
            - db (torch.Tensor): Tensor representing the gradient of the bias with respect to the forward output.
            - dresidual_in (torch.Tensor or None):
            Tensor representing the gradient of the residual input with respect to the forward output
            if residual is present, otherwise None.
            - None.
            - None.
            - None.
            - None.
        """
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = _layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """
    Apply layer normalization to the input tensor `x`.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        weight (torch.Tensor): The weight tensor for affine transformation.
        bias (torch.Tensor): The bias tensor for affine transformation.
        residual (torch.Tensor, optional): The residual tensor to be added after normalization. Default is None.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        prenorm (bool, optional): Whether to apply pre-normalization. Default is False.
        residual_in_fp32 (bool, optional): Whether to convert the residual tensor to float32. Default is False.
        is_rms_norm (bool, optional): Whether to apply root mean square normalization. Default is False.

    Returns:
        torch.Tensor: The normalized tensor.

    """
    return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)


def rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6):
    """
    Apply root mean square normalization to the input tensor.

    Args:
        x (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias (Tensor): The bias tensor.
        residual (Tensor, optional): The residual tensor. Defaults to None.
        prenorm (bool, optional): Whether to apply pre-normalization. Defaults to False.
        residual_in_fp32 (bool, optional): Whether the residual tensor is in FP32 format. Defaults to False.
        eps (float, optional): The epsilon value. Defaults to 1e-6.

    Returns:
        Tensor: The normalized tensor.
    """
    return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, True)


class RMSNorm(torch.nn.Module):
    """RMNNorm class."""

    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        """
        Initializes the RMSNorm object.

        Args:
            hidden_size (int): The size of the hidden state.
            eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.
            device (torch.device, optional): The device on which the parameters are stored. Defaults to None.
            dtype (torch.dtype, optional): The desired data type of the parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters.

        This method resets the weight parameters with ones.

        Args:
            None

        Returns:
            None
        """
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (Tensor): The input tensor.
            residual (Tensor, optional): The residual tensor. Default is None.
            prenorm (bool, optional): Whether to apply pre-normalization. Default is False.
            residual_in_fp32 (bool, optional): Whether the residual tensor is in FP32 format. Default is False.

        Returns:
            Tensor: The output tensor after applying LayerNorm.

        """
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class LayerNormLinearFn(torch.autograd.Function):
    """LayerNormLinearFn class."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        """
        Forward pass of the LayerNorm operation.

        Args:
            ctx: Context object.
            x (torch.Tensor): Input tensor.
            norm_weight (torch.Tensor): Normalization weight tensor.
            norm_bias (torch.Tensor): Normalization bias tensor.
            linear_weight (torch.Tensor): Linear weight tensor.
            linear_bias (torch.Tensor): Linear bias tensor.
            residual (torch.Tensor, optional): Residual tensor. Defaults to None.
            eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-6.
            prenorm (bool, optional): Flag indicating whether to apply pre-normalization. Defaults to False.
            residual_in_fp32 (bool, optional): Flag indicating whether the residual tensor is in FP32 format.
            Defaults to False.
            is_rms_norm (bool, optional): Flag indicating whether to use RMS normalization. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor,torch.Tensor]]

        Raises:
            AssertionError: If the shape of the residual tensor does not match the original shape of the input tensor.

        """
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        norm_weight = norm_weight.contiguous()
        if norm_bias is not None:
            norm_bias = norm_bias.contiguous()
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
        y, mean, rstd, residual_out = _layer_norm_fwd(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        """
        Backward pass of the layer normalization operation.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): Autograd context.
            dout (torch.Tensor): Gradient of the output tensor.
            *args: Additional arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, None, None, None, None]:
            - dx (torch.Tensor): Gradient of the input tensor.
            - dnorm_weight (torch.Tensor): Gradient of the normalization weight.
            - dnorm_bias (torch.Tensor): Gradient of the normalization bias.
            - dlinear_weight (torch.Tensor): Gradient of the linear weight.
            - dlinear_bias (torch.Tensor): Gradient of the linear bias.
            - dresidual_in (torch.Tensor): Gradient of the residual input.
            - None.
            - None.
            - None.
            - None
        """
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = _layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_linear_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    """
    Apply layer normalization followed by a linear transformation to the input tensor `x`.

    Args:
        x (torch.Tensor): The input tensor.
        norm_weight (torch.Tensor): The weight tensor for layer normalization.
        norm_bias (torch.Tensor): The bias tensor for layer normalization.
        linear_weight (torch.Tensor): The weight tensor for the linear transformation.
        linear_bias (torch.Tensor): The bias tensor for the linear transformation.
        residual (torch.Tensor, optional): The residual tensor to be added to the output. Defaults to None.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
        prenorm (bool, optional): Whether to apply pre-normalization. Defaults to False.
        residual_in_fp32 (bool, optional): Whether the residual tensor is in FP32 format. Defaults to False.
        is_rms_norm (bool, optional): Whether to apply RMS normalization. Defaults to False.

    Returns:
        torch.Tensor: The output tensor after applying layer normalization and linear transformation.
    """
    return LayerNormLinearFn.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )
