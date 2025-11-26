import logging
import os
from contextlib import contextmanager
from functools import partial

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
):
    if isinstance(target, (list, tuple)):
        if any(torch.isnan(o).any() for o in target if isinstance(o, torch.Tensor)):
            logger.error(f"NaN detected in {target_name} {module.__class__.__name__}")
            if module_path:
                logger.error(f"Module path: {module_path}")
    elif isinstance(target, torch.Tensor) and torch.isnan(target).any():
        logger.error(f"NaN detected in {target_name} {module.__class__.__name__}")
        if module_path:
            logger.error(f"Module path: {module_path}")


def debug_nan_hook(
    module: torch.nn.Module,
    input: torch.Tensor | tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    module_path: str | None = None,
):
    """Hook to detect NaN in forward pass"""
    _detect_nan(module, module_path, input, "input")
    _detect_nan(module, module_path, output, "output")


def register_nan_hooks(model: torch.nn.Module):
    """Register NaN detection hooks on all modules"""
    for name, module in model.named_modules():
        module.register_forward_hook(partial(debug_nan_hook, module_path=name))
