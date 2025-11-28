from functools import partial
from typing import Any

import torch

from modalities.utils.debug import debug_nan_hook, enable_deterministic_cuda, print_forward_hook


class Debugging:
    def __init__(self, *, forward_hooks: list[list[torch.utils.hooks.RemovableHandle]], enable_determinism: bool):
        self.forward_hooks = forward_hooks
        self.enable_determinism = enable_determinism
        if self.enable_determinism:
            self._deterministic_context = enable_deterministic_cuda()
            self._deterministic_context.__enter__()

    def __del__(self):
        for hook_group in self.forward_hooks:
            for handle in hook_group:
                handle.remove()
        if self.enable_determinism:
            self._deterministic_context.__exit__(None, None, None)


class HookRegistration:
    """Utility component to register and manage hooks on a PyTorch model."""

    @staticmethod
    def register_nan_hooks(
        model: torch.nn.Module,
        raise_exception: bool = False,
        module_filter: Any = lambda module: True,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Registers NaN detection hooks on all modules that satisfy the module_filter condition.

        Args:
            model (torch.nn.Module): The PyTorch model to register hooks on.
            raise_exception (bool, optional): Whether to raise an exception when NaN is detected. Defaults to False.
            module_filter (Any, optional): A function that takes a module and
                returns True if the hook should be registered.
                Defaults to a function that always returns True.

        Returns:
            list[torch.utils.hooks.RemovableHandle]: A list of handles for the registered hooks.
        """
        return HookRegistration.register_forward_hooks(
            model, hook_fn=partial(debug_nan_hook, raise_exception=raise_exception), module_filter=module_filter
        )

    @staticmethod
    def register_print_forward_hooks(
        model: torch.nn.Module,
        print_shape_only: bool = False,
        module_filter: Any = lambda module: True,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Registers print hooks on all modules that satisfy the module_filter condition.

        Args:
            model (torch.nn.Module): The PyTorch model to register hooks on.
            module_filter (Any, optional): A function that takes a module and
                returns True if the hook should be registered.
                Defaults to a function that always returns True.

        Returns:
            list[torch.utils.hooks.RemovableHandle]: A list of handles for the registered hooks.
        """
        return HookRegistration.register_forward_hooks(
            model, hook_fn=partial(print_forward_hook, print_shape_only=print_shape_only), module_filter=module_filter
        )

    @staticmethod
    def register_forward_hooks(
        model: torch.nn.Module,
        hook_fn: Any,
        module_filter: Any = lambda module: True,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Registers forward hooks on all modules that satisfy the module_filter condition.

        Args:
            model (torch.nn.Module): The PyTorch model to register hooks on.
            hook_fn (Any): The hook function to be registered.
            module_filter (Any, optional): A function that takes a module and
                returns True if the hook should be registered.
                Defaults to a function that always returns True.

        Returns:
            list[torch.utils.hooks.RemovableHandle]: A list of handles for the registered hooks.
        """
        handles = []
        for name, module in model.named_modules():
            if module_filter(module):
                handle = module.register_forward_hook(partial(hook_fn, module_path=name))
                handles.append(handle)
        return handles
