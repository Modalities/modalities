import logging
import os
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


@contextmanager
def enable_deterministic_cuda():
    """Context manager to enable deterministic CUDA operations and restore previous state."""
    prev_cudnn_deterministic = torch.backends.cudnn.deterministic
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    prev_algos = torch.are_deterministic_algorithms_enabled()
    prev_cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_cudnn_deterministic
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark
        torch.use_deterministic_algorithms(prev_algos)
        if prev_cublas_cfg is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas_cfg


def _detect_nan(
    module: torch.nn.Module,
    module_path: str | None,
    target: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    target_name: str,
    raise_exception: bool,
):
    if isinstance(target, (list, tuple)):
        if any(torch.isnan(o).any() for o in target if isinstance(o, torch.Tensor)):
            logger.error(f"NaN detected in {target_name} {module.__class__.__name__}")
            if module_path:
                logger.error(f"Module path: {module_path}")
            if raise_exception:
                raise ValueError(f"NaN detected in {target_name} of module {module.__class__.__name__}")
    elif isinstance(target, torch.Tensor) and torch.isnan(target).any():
        logger.error(f"NaN detected in {target_name} {module.__class__.__name__}")
        if module_path:
            logger.error(f"Module path: {module_path}")
        if raise_exception:
            raise ValueError(f"NaN detected in {target_name} of module {module.__class__.__name__}")


def debug_nan_hook(
    module: torch.nn.Module,
    input: torch.Tensor | tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    module_path: str | None = None,
    raise_exception: bool = False,
):
    """Hook to detect NaN in forward pass"""
    _detect_nan(module, module_path, target=input, target_name="input", raise_exception=raise_exception)
    _detect_nan(module, module_path, target=output, target_name="output", raise_exception=raise_exception)


def print_forward_hook(
    module: torch.nn.Module,
    input: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, Any],
    output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | dict[str, Any],
    module_path: str | None = None,
    print_shape_only: bool = False,
):
    """Hook to print input and output shapes during forward pass"""
    if isinstance(input, (list, tuple)):
        input_shapes = [inp.shape for inp in input if isinstance(inp, torch.Tensor)]
    elif isinstance(input, torch.Tensor):
        input_shapes = [input.shape]
    else:
        input_shapes = []

    if isinstance(output, (list, tuple)):
        output_shapes = [out.shape for out in output if isinstance(out, torch.Tensor)]
    elif isinstance(output, torch.Tensor):
        output_shapes = [output.shape]
    else:
        output_shapes = []

    print(
        f"Module: {module.__class__.__name__}, "
        f"Path: {module_path}, "
        f"Input shapes: {input_shapes}, "
        f"Output shapes: {output_shapes}"
    )
    if not print_shape_only:
        print(f">>> Input:\n{input}")
        if hasattr(module, "weight"):
            print(f">>> Weights:\n{module.weight}")
        print(f">>> Output:\n{output}")
