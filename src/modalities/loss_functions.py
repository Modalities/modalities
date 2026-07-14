from abc import ABC, abstractmethod
from typing import overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100


class Loss(ABC):
    def __init__(self, tag: str):
        self._tag = tag

    @property
    def tag(self) -> str:
        return self._tag

    @abstractmethod
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Calculates the loss
        :return: Loss tensor
        """
        raise NotImplementedError


class CLMCrossEntropyLoss(Loss):
    def __init__(self, target_key: str, prediction_key: str, tag: str = "CLMCrossEntropyLoss"):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        # Mean over the tokens in the local-batch (batch per rank)
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, lm_logits = self._parse_arguments(args, kwargs)

        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous().long()
        # Flatten the tokens. We compute here, the loss per token.
        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def _parse_arguments(
        self,
        args: list[torch.Tensor] | list[InferenceResultBatch],
        kwargs: dict[str, torch.Tensor] | dict[str, InferenceResultBatch],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(args) == 1 and isinstance(args[0], InferenceResultBatch):
            forward_batch = args[0]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], InferenceResultBatch):
            forward_batch = kwargs["forward_batch"]
            labels = forward_batch.get_targets(self.target_key)
            lm_logits = forward_batch.get_predictions(self.prediction_key)
        elif len(args) == 2 and all(isinstance(arg, torch.Tensor) for arg in args):
            lm_logits, labels = args
        elif (
            "outputs" in kwargs
            and "targets" in kwargs
            and isinstance(kwargs["outputs"], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = kwargs["outputs"]
            labels = kwargs["targets"]
        elif (
            len(args) == 1
            and "targets" in kwargs
            and isinstance(args[0], torch.Tensor)
            and isinstance(kwargs["targets"], torch.Tensor)
        ):
            lm_logits = args[0]
            labels = kwargs["targets"]
        else:
            raise TypeError("Invalid arguments for CLMCrossEntropyLoss.__call__")
        return labels, lm_logits


def _ce_loss_mean(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="mean")


def _ce_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="sum")


class ChunkedCLMCrossEntropyLoss(Loss):
    """Cross-entropy loss computed in chunks with optional torch.compile.

    Splits the flattened token sequence into ``num_chunks`` chunks and
    accumulates a sum-then-normalize loss.  This limits the peak memory for
    intermediate float32 tensors (log-softmax, gradients) from O(B·L·V) to
    O(B·L/num_chunks·V).  Setting ``use_compile=True`` additionally fuses the
    per-chunk loss kernels via ``torch.compile``, avoiding float32 intermediate
    materialisations entirely for each chunk.

    Use ``num_chunks=1, use_compile=True`` for compiled-only mode (kernel
    fusion savings without chunking).

    Args:
        target_key: Key to access labels in the ``InferenceResultBatch``.
        prediction_key: Key to access logits in the ``InferenceResultBatch``.
        num_chunks: Number of chunks to split the token sequence into. Must
            be >= 1.  Defaults to 1 (no splitting).
        use_compile: Apply ``torch.compile`` to the per-chunk loss function.
            Defaults to ``True``.
        tag: Loss tag used for logging. Defaults to
            ``"ChunkedCLMCrossEntropyLoss"``.
    """

    def __init__(
        self,
        target_key: str,
        prediction_key: str,
        num_chunks: int = 1,
        use_compile: bool = True,
        tag: str = "ChunkedCLMCrossEntropyLoss",
    ):
        super().__init__(tag)
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.num_chunks = num_chunks

        if num_chunks == 1:
            # No chunking — use mean-reduction directly.
            base_fn = _ce_loss_mean
            self._use_chunks = False
        else:
            base_fn = _ce_loss_sum
            self._use_chunks = True

        self._loss_fn = torch.compile(base_fn) if use_compile else base_fn

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        labels = forward_batch.get_targets(self.target_key)
        lm_logits = forward_batch.get_predictions(self.prediction_key)

        labels = labels.to(lm_logits.device)
        flat_logits = lm_logits.contiguous().view(-1, lm_logits.size(-1))
        flat_labels = labels.contiguous().view(-1).long()

        if not self._use_chunks:
            return self._loss_fn(flat_logits, flat_labels)

        logit_chunks = flat_logits.tensor_split(self.num_chunks, dim=0)
        label_chunks = flat_labels.tensor_split(self.num_chunks, dim=0)

        total_loss = flat_logits.new_zeros(())
        for logit_chunk, label_chunk in zip(logit_chunks, label_chunks):
            total_loss = total_loss + self._loss_fn(logit_chunk, label_chunk)

        return total_loss / flat_labels.numel()


class ChunkedLMHeadCrossEntropyLoss(Loss):
    """Chunked lm_head + cross-entropy loss with per-chunk backward.

    In contrast to ``ChunkedCLMCrossEntropyLoss`` (which chunks the loss math over
    already-materialized logits), this loss receives the *hidden states* from the model
    (see ``GPT2LLMConfig.return_hidden_states``) and applies the lm_head projection
    chunk-wise along the sequence dimension. The full ``[B, L, vocab_size]`` logits
    tensor is never materialized; peak memory for the head is ``O(B * L/num_chunks * V)``.

    During training (``compute_and_backward``), each chunk's loss is backpropagated
    immediately through the lm_head so its autograd graph is freed before the next chunk
    runs. The gradients w.r.t. the hidden states are collected in a pre-allocated buffer
    and propagated through the transformer trunk with a single ``backward`` call at the
    end. This loss therefore OWNS the backward pass; the trainer must not call
    ``loss.backward()`` again.

    Under FSDP2, the lm_head must be its own ``fully_shard`` unit (see
    ``ModelFactory.get_fsdp2_wrapped_model(..., wrap_lm_head_separately=True)``). The
    lm_head weight is kept unsharded across all chunks (avoiding repeated all-gathers)
    and its gradient reduce-scatter is coalesced into the last chunk's backward.
    Weight tying is not supported in that setting, since wte and lm_head would then
    belong to different FSDP units.

    Args:
        target_key: Key to access labels in the ``InferenceResultBatch``.
        prediction_key: Key to access the hidden states in the ``InferenceResultBatch``.
        num_chunks: Number of chunks to split the sequence dimension into. The sequence
            length must be divisible by this. Defaults to 8 (as in torchtitan).
        use_compile: Apply ``torch.compile`` to the per-chunk cross-entropy function.
            Defaults to ``True``.
        tag: Loss tag used for logging.
    """

    def __init__(
        self,
        target_key: str,
        prediction_key: str,
        num_chunks: int = 8,
        use_compile: bool = True,
        tag: str = "ChunkedLMHeadCrossEntropyLoss",
    ):
        super().__init__(tag)
        if num_chunks < 1:
            raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.num_chunks = num_chunks
        self.lm_head: nn.Module | None = None
        self._loss_fn = torch.compile(_ce_loss_sum) if use_compile else _ce_loss_sum

    def bind_lm_head(self, model: nn.Module) -> None:
        """Binds the (possibly FSDP-wrapped) lm_head module of the given model to this loss.

        Must be called after the model is fully wrapped (FSDP etc.) and before the first
        loss computation. The trainer and evaluator call this lazily.
        """
        self.lm_head = model.get_submodule("transformer.lm_head")
        if isinstance(self.lm_head, FSDPModule):
            # FSDP2's default backward-prefetch heuristic picks its target from a global,
            # cross-module post-forward-call order. Since lm_head.forward() is invoked
            # num_chunks times per step, interleaved with its own per-chunk backward calls
            # (forward(chunk0) -> backward(chunk0) -> forward(chunk1) -> ...), that ordering
            # no longer reflects real data dependencies: the heuristic ends up re-gathering
            # unrelated modules (e.g., the last transformer block, which no chunk backward
            # actually needs, since each hidden_chunk is a detached leaf) or lm_head's own
            # earlier forward call. Overriding the prefetch target to lm_head itself makes
            # this a no-op (unshard() returns immediately when already unsharded, which it
            # is throughout the chunk loop -- see compute_and_backward), eliminating the
            # wasted re-gathers. An empty list would NOT disable prefetching here: FSDP
            # treats an empty override list the same as no override (falls back to the
            # default heuristic), so a self-reference is required.
            self.lm_head.set_modules_to_backward_prefetch([self.lm_head])

    def _chunk(self, t: torch.Tensor) -> list[torch.Tensor]:
        seq_len = t.shape[1]
        if seq_len % self.num_chunks != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by num_chunks {self.num_chunks}."
            )
        return list(t.tensor_split(self.num_chunks, dim=1))

    def _chunk_ce_sum(self, hidden_chunk: torch.Tensor, label_chunk: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_chunk)
        return self._loss_fn(logits.reshape(-1, logits.size(-1)), label_chunk.reshape(-1).long())

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """Forward-only chunked loss (used for evaluation, typically under ``torch.no_grad()``)."""
        if self.lm_head is None:
            raise RuntimeError(
                "lm_head is not set. Call bind_lm_head(model) before computing "
                "ChunkedLMHeadCrossEntropyLoss."
            )
        hidden_states = forward_batch.get_predictions(self.prediction_key)
        labels = forward_batch.get_targets(self.target_key).to(hidden_states.device)

        num_valid = (labels != IGNORE_INDEX).sum().clamp(min=1)
        total_loss = hidden_states.new_zeros(())
        for hidden_chunk, label_chunk in zip(self._chunk(hidden_states), self._chunk(labels)):
            total_loss = total_loss + self._chunk_ce_sum(hidden_chunk, label_chunk)
        return total_loss / num_valid

    def compute_and_backward(
        self, hidden_states: torch.Tensor, labels: torch.Tensor, grad_scale: float = 1.0
    ) -> torch.Tensor:
        """Computes the loss chunk-wise and runs the full backward pass.

        Args:
            hidden_states: Model output of shape [B, L, D] (after lm_head_norm), requires grad.
            labels: Target token ids of shape [B, L]; ``IGNORE_INDEX`` entries are masked out.
            grad_scale: Factor multiplied into the backpropagated loss only (e.g.,
                1/gradient_accumulation_steps). The returned loss value is unscaled.

        Returns:
            The detached mean loss over non-ignored tokens.
        """
        if self.lm_head is None:
            raise RuntimeError(
                "lm_head is not set. Call bind_lm_head(model) before computing "
                "ChunkedLMHeadCrossEntropyLoss."
            )
        if not hidden_states.requires_grad:
            raise RuntimeError(
                "compute_and_backward requires hidden_states with requires_grad=True. "
                "Use __call__ for evaluation."
            )

        labels = labels.to(hidden_states.device)
        hidden_chunks = [c.detach().requires_grad_() for c in self._chunk(hidden_states)]
        label_chunks = self._chunk(labels)

        num_valid = (labels != IGNORE_INDEX).sum().clamp(min=1)
        grad_buffer = torch.zeros_like(hidden_states)
        total_loss = hidden_states.new_zeros(())

        # Keep the lm_head weight unsharded across all chunks (one all-gather instead of
        # num_chunks) and coalesce the gradient reduce-scatter into the last chunk's backward.
        is_fsdp2 = isinstance(self.lm_head, FSDPModule)
        if is_fsdp2:
            self.lm_head.set_reshard_after_forward(False)
            self.lm_head.set_reshard_after_backward(False)
            # The chunk backwards are intermediate backwards of the same training step. By
            # default FSDP2 treats every backward() as the last one: its root post-backward
            # callback then finalizes ALL param groups, clearing the global post_forward_order
            # and every transformer block's _post_forward_indices -- the bookkeeping that
            # backward prefetching relies on. The subsequent trunk backward would then run
            # with prefetching disabled, exposing every block's all-gather (~10 ms GPU idle
            # per block, +~280 ms/step measured for the 8B model at micro-batch size 2).
            # Marking the chunk backwards as non-last skips that finalization. The flag lives
            # on the state context shared by the entire FSDP tree, so setting it through the
            # lm_head handle also governs the root callback.
            self.lm_head.set_is_last_backward(False)

        offset = 0
        for chunk_idx, (hidden_chunk, label_chunk) in enumerate(zip(hidden_chunks, label_chunks)):
            is_last = chunk_idx == self.num_chunks - 1
            if is_fsdp2:
                self.lm_head.set_requires_gradient_sync(is_last)
                if is_last:
                    self.lm_head.set_reshard_after_backward(True)

            chunk_loss = self._chunk_ce_sum(hidden_chunk, label_chunk)
            # Backward through the lm_head only (hidden_chunk is a detached leaf); frees
            # this chunk's activations before the next chunk's forward.
            (chunk_loss * (grad_scale / num_valid)).backward()

            chunk_len = hidden_chunk.shape[1]
            grad_buffer[:, offset : offset + chunk_len] = hidden_chunk.grad
            hidden_chunk.grad = None
            offset += chunk_len
            total_loss = total_loss + chunk_loss.detach()

        if is_fsdp2:
            self.lm_head.set_reshard_after_forward(True)
            self.lm_head.reshard()
            # The trunk backward is the true last backward of this step: let the root
            # callback finalize all param groups and clear the prefetch bookkeeping.
            self.lm_head.set_is_last_backward(True)

        # Single backward through the transformer trunk with the accumulated gradient.
        hidden_states.backward(gradient=grad_buffer)

        return total_loss / num_valid


def nce_loss(
    embedding1: torch.Tensor, embedding2: torch.Tensor, device: torch.device, is_asymmetric: bool, temperature: float
) -> torch.Tensor:
    """
    This implementation calculates the noise contrastive estimation loss between embeddings of two different modalities
    Implementation slightly adapted from https://arxiv.org/pdf/1912.06430.pdf, https://github.com/antoine77340/MIL-NCE_HowTo100M
    changes include adding a temperature value and the choice of calculating asymmetric loss w.r.t. one modality
    This implementation is adapted to contrastive loss from CoCa model https://arxiv.org/pdf/2205.01917.pdf

    Args:
        embedding1 (torch.Tensor): embeddings from modality 1 of size batch_size x embed_dim.
        embedding2 (torch.Tensor): embeddings from modality 2 of size batch_size x embed_dim.
        device (torch.device): torch device for calculating loss.
        is_asymmetric (bool): boolean value to specify if the loss is calculated in one direction or both directions.
        temperature (float): temperature value for regulating loss.

    Returns:
            torch.Tensor: loss tensor.
    """
    # calculating the similarity matrix of size (batch_size x batch_size)
    sim_matrix = torch.matmul(embedding1, embedding2.t()) / temperature
    # numerator of loss: using similarity scores for all positive pairs (e.g., image and its caption)
    numerator = sim_matrix * torch.eye(sim_matrix.shape[0], device=device)
    numerator = numerator.sum(dim=0).view(sim_matrix.shape[0], -1)
    numerator = torch.logsumexp(numerator, dim=1)
    if is_asymmetric:
        # denominator of loss: using all similarity scores for all pairs (positive and negative)
        denominator = torch.logsumexp(sim_matrix, dim=1)
    else:
        # calculate bidirectional loss
        numerator *= 2
        denominator = torch.logsumexp(sim_matrix, dim=1) + torch.logsumexp(sim_matrix.t(), dim=1)
    return torch.mean(denominator - numerator)  # calculated in log space


class NCELoss(Loss):
    def __init__(
        self,
        prediction_key1: str,
        prediction_key2: str,
        is_asymmetric: bool = True,
        temperature: float = 1.0,
        tag: str = "NCELoss",
    ):
        """
        Noise Contrastive Estimation Loss

        Args:
            prediction_key1 (str): key to access embedding 1.
            prediction_key2 (str): key to access embedding 2.
            is_asymmetric (bool, optional): specifies symmetric or asymmetric calculation of NCEloss. Defaults to True.
            temperature (float, optional): temperature. Defaults to 1.0.
            tag (str, optional): Defaults to "NCELoss".
        """
        super().__init__(tag)
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.is_asymmetric = is_asymmetric
        self.temperature = temperature

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch.

        Returns:
            torch.Tensor: loss tensor.
        """
        embedding1 = forward_batch.get_predictions(self.prediction_key1)
        embedding2 = forward_batch.get_predictions(self.prediction_key2)

        contiguous_embedding1 = embedding1.contiguous()
        contiguous_embedding2 = embedding2.contiguous()

        loss = nce_loss(
            contiguous_embedding1, contiguous_embedding2, embedding1.device, self.is_asymmetric, self.temperature
        )
        return loss
