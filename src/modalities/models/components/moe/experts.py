"""Grouped expert feed-forward networks for mixture-of-experts layers.

All experts of a layer are stored in two stacked weight tensors so that the whole layer can be
evaluated with two grouped matrix multiplications instead of one small matmul per expert. This is
the same layout TorchTitan uses, and it is what makes 128 experts practical.

Nemotron-3 Nano's experts are *not* gated: a single up-projection followed by a squared ReLU and a
down-projection. That halves the number of expert matrices compared to a SwiGLU expert.
"""

import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertsBackend(str, Enum):
    """
    Enum of the available grouped expert matmul implementations.

    Attributes:
        GROUPED_MM (str): ``torch._grouped_mm``, a single kernel over all experts. Requires a
            CUDA device and bfloat16 inputs; falls back to ``LOOPED`` automatically otherwise.
        LOOPED (str): A Python loop over experts. Slower, but runs anywhere and in any dtype,
            which makes it the reference implementation for tests.
    """

    GROUPED_MM = "grouped_mm"
    LOOPED = "looped"


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the squared ReLU activation, ``max(x, 0) ** 2``.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The activated tensor.
    """
    return torch.square(F.relu(x))


class GroupedExperts(nn.Module):
    """
    A stack of ``num_experts`` non-gated feed-forward experts evaluated with grouped matmuls.

    The forward pass expects tokens that are already sorted by expert assignment, together with
    the number of tokens per expert, so that each expert's slice is a contiguous block.
    """

    def __init__(
        self,
        n_embd: int,
        ffn_hidden: int,
        num_experts: int,
        backend: ExpertsBackend = ExpertsBackend.GROUPED_MM,
    ):
        """
        Initializes the GroupedExperts module.

        Args:
            n_embd (int): The model dimension.
            ffn_hidden (int): The per-expert hidden dimension.
            num_experts (int): The number of experts.
            backend (ExpertsBackend): Which grouped matmul implementation to use.
        """
        super().__init__()
        self.n_embd = n_embd
        self.ffn_hidden = ffn_hidden
        self.num_experts = num_experts
        self.backend = ExpertsBackend(backend)

        if self.backend == ExpertsBackend.GROUPED_MM:
            # torch._grouped_mm requires every non-unit stride to be a multiple of 16 bytes. At
            # 2 bytes per element that means both matmul inner dimensions must be multiples of 8.
            # Checking here turns a cryptic mid-forward RuntimeError into a config-time error.
            unaligned = {
                name: value for name, value in (("n_embd", n_embd), ("ffn_hidden", ffn_hidden)) if value % 8 != 0
            }
            if unaligned:
                raise ValueError(
                    f"ExpertsBackend.GROUPED_MM requires 16-byte aligned strides, so "
                    f"{' and '.join(f'{name}={value}' for name, value in unaligned.items())} must be "
                    f"a multiple of 8. Adjust the dimensions or use ExpertsBackend.LOOPED."
                )

        # Expert dimension first, so that FSDP2 shards along the expert axis by default.
        self.w1 = nn.Parameter(torch.empty(num_experts, ffn_hidden, n_embd))
        self.w2 = nn.Parameter(torch.empty(num_experts, n_embd, ffn_hidden))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes the expert weights with the default ``nn.Linear`` scheme.

        These are raw ``nn.Parameter`` tensors rather than ``nn.Linear`` modules, so nothing else
        would initialize them. Modalities calls ``reset_parameters`` on every submodule before the
        configured weight initializer runs, which means a misconfigured regex filter degrades to
        a sane default instead of leaving uninitialized memory in the model.
        """
        if self.w1.device.type == "meta":
            # Nothing to initialize on the meta device; the factory materializes and re-runs this.
            return
        with torch.no_grad():
            for weight in (self.w1, self.w2):
                # Match nn.Linear: kaiming_uniform_ with a=sqrt(5) over the fan-in of each expert.
                fan_in = weight.shape[-1]
                bound = math.sqrt(1.0 / fan_in) * math.sqrt(3.0)
                nn.init.uniform_(weight, -bound, bound)

    def _use_grouped_mm(self, x: torch.Tensor) -> bool:
        """
        Decides whether the fused grouped matmul can be used for this input.

        Args:
            x (torch.Tensor): The (sorted) token tensor.

        Returns:
            bool: True if ``torch._grouped_mm`` is available and applicable.
        """
        return (
            self.backend == ExpertsBackend.GROUPED_MM
            and hasattr(torch, "_grouped_mm")
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        )

    def forward(self, x_sorted: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Applies each expert to its contiguous slice of the sorted token tensor.

        Args:
            x_sorted (torch.Tensor): Tokens sorted by expert, of shape ``(num_routed, n_embd)``.
            tokens_per_expert (torch.Tensor): Token count per expert, of shape ``(num_experts,)``,
                integer dtype. Must sum to ``num_routed``.

        Returns:
            torch.Tensor: Expert outputs of shape ``(num_routed, n_embd)``.
        """
        if self._use_grouped_mm(x_sorted):
            offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
            hidden = squared_relu(torch._grouped_mm(x_sorted, self.w1.transpose(-2, -1), offs=offsets))
            return torch._grouped_mm(hidden, self.w2.transpose(-2, -1), offs=offsets)

        return self._forward_looped(x_sorted, tokens_per_expert)

    def _forward_looped(self, x_sorted: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation: loop over experts and apply each to its slice.

        Args:
            x_sorted (torch.Tensor): Tokens sorted by expert, of shape ``(num_routed, n_embd)``.
            tokens_per_expert (torch.Tensor): Token count per expert, of shape ``(num_experts,)``.

        Returns:
            torch.Tensor: Expert outputs of shape ``(num_routed, n_embd)``.
        """
        outputs = torch.empty_like(x_sorted)
        # A single host sync here instead of one per expert.
        counts = tokens_per_expert.tolist()
        start = 0
        for expert_idx, count in enumerate(counts):
            if count == 0:
                continue
            stop = start + count
            chunk = x_sorted[start:stop]
            hidden = squared_relu(chunk @ self.w1[expert_idx].transpose(0, 1))
            outputs[start:stop] = hidden @ self.w2[expert_idx].transpose(0, 1)
            start = stop
        return outputs
