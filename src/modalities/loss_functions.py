from abc import ABC, abstractmethod
from typing import Callable, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule as FSDP2
from torch.utils.checkpoint import checkpoint

from modalities.batch import InferenceResultBatch

# PyTorch's default ignore index for cross-entropy loss. Tokens with this label are
# excluded from both the loss value and the (valid-)token normalization.
IGNORE_INDEX = -100


def clm_cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pure-tensor causal-LM cross-entropy with mean reduction over valid tokens.

    This is a free function (not a bound method) so it can be handed to
    ``torch.compile`` as a clean, ``self``-free callable, mirroring TorchTitan's
    module-level ``cross_entropy_loss`` compile target
    (torchtitan/components/loss.py). Tokens labelled ``IGNORE_INDEX`` are ignored.

    Args:
        logits (torch.Tensor): Unnormalized predictions of shape (..., vocab_size).
        labels (torch.Tensor): Target token ids, broadcastable to ``logits[..., 0]``.

    Returns:
        torch.Tensor: Scalar mean cross-entropy loss.
    """
    # move labels to correct device to enable model parallelism
    labels = labels.to(logits.device)
    logits = logits.contiguous()
    labels = labels.contiguous().long()
    # Flatten the tokens. We compute here, the loss per token.
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean", ignore_index=IGNORE_INDEX
    )


def clm_cross_entropy_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pure-tensor causal-LM cross-entropy with *sum* reduction over valid tokens.

    Used by the chunked loss: summing per-chunk contributions and dividing by the
    total valid-token count reproduces the global mean of :func:`clm_cross_entropy_loss`,
    while allowing each chunk to be computed (and freed) independently.
    """
    labels = labels.to(logits.device)
    logits = logits.contiguous()
    labels = labels.contiguous().long()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum", ignore_index=IGNORE_INDEX
    )


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

    def compile(self, backend: str = "inductor") -> None:
        """Compile the pure-tensor computation of this loss in place.

        Mirrors TorchTitan's ``BaseLoss._maybe_compile``: only the numeric
        tensor-in/tensor-out core (``self.fn``) is compiled, never the batch/
        container unpacking. Subclasses with a compile-friendly core override this.

        Args:
            backend (str): torch.compile backend. Defaults to "inductor".
        """
        raise NotImplementedError(f"{type(self).__name__} does not support loss compilation.")


class LossFactory:
    """Factory that applies training-time transformations to loss functions,
    mirroring :class:`~modalities.models.model_factory.ModelFactory`."""

    @staticmethod
    def get_compiled_loss(loss: Loss, backend: str = "inductor") -> Loss:
        """Compile the pure-tensor core of the given loss in place and return it.

        Follows the same in-place-mutate-and-return contract as
        ``ModelFactory.get_compiled_model``. Composes with any ``Loss`` that
        implements ``compile`` (e.g. wrapping a chunked loss compiles its CE core).

        Args:
            loss (Loss): The loss whose numeric core should be compiled.
            backend (str): torch.compile backend. Defaults to "inductor".

        Returns:
            Loss: The same loss instance with its ``fn`` compiled.
        """
        loss.compile(backend=backend)
        return loss


class CLMCrossEntropyLoss(Loss):
    def __init__(self, target_key: str, prediction_key: str, tag: str = "CLMCrossEntropyLoss"):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        # Pure-tensor core. Swapped for a compiled variant by `compile`.
        # Mean over the (valid) tokens in the local-batch (batch per rank).
        self.fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = clm_cross_entropy_loss

    def compile(self, backend: str = "inductor") -> None:
        # Compile only the tensor core, not the InferenceResultBatch unpacking.
        # Note: unlike model/block compilation we do not pass fullgraph=True here,
        # matching TorchTitan's loss compile (torchtitan/components/loss.py).
        self.fn = torch.compile(self.fn, backend=backend)

    @overload
    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        ...

    @overload
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        labels, lm_logits = self._parse_arguments(args, kwargs)
        return self.fn(lm_logits, labels)

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


class ChunkedCLMCrossEntropyLoss(Loss):
    """Memory-efficient causal-LM cross-entropy that never materializes the full
    ``[batch, seq_len, vocab_size]`` logits tensor.

    Same goal as TorchTitan's ``ChunkedLossWrapper`` (torchtitan/components/loss.py):
    the language-model head is moved *out* of the model's forward pass and applied
    chunk-by-chunk inside the loss, so peak activation memory of the head + the
    float32 up-cast inside cross-entropy is reduced by roughly ``num_chunks``. For a
    131k vocabulary this is the single largest activation in the model.

    Mechanism (adapted to modalities' idioms):
        1. The model runs with ``skip_lm_head=True`` and returns the post-norm
           hidden states ``[batch, seq_len, n_embd]`` under ``prediction_key``.
        2. Hidden states and labels are split into ``num_chunks`` along the
           sequence dimension.
        3. Each chunk is pushed through the (referenced) ``lm_head`` and cross-entropy
           inside ``torch.utils.checkpoint``, so the chunk's logits are freed after
           the forward and recomputed on demand during backward. Only one chunk's
           logits are alive at any time.
        4. Per-chunk *sum*-reduced losses are accumulated and divided by the global
           valid-token count, which is numerically equal to the mean reduction of
           :class:`CLMCrossEntropyLoss`.

    Unlike TorchTitan, we rely on ``torch.utils.checkpoint`` (recomputing the head in
    backward) rather than a manual per-chunk backward + custom autograd bridge. This
    keeps the implementation torch-native and consistent with modalities' existing
    activation-checkpointing approach; the trade-off is one extra ``lm_head`` forward
    per chunk during backward.

    Note:
        Under FSDP2 with ``reshard_after_forward=True`` the ``lm_head`` must be its
        own FSDP unit so that the per-chunk (and recomputed) head calls trigger the
        parameter all-gather. Tensor-parallel loss-parallel cross-entropy is not
        handled here (modalities' plain CE is not loss-parallel either).
    """

    def __init__(
        self,
        model: nn.Module,
        target_key: str,
        prediction_key: str,
        num_chunks: int = 8,
        tag: str = "ChunkedCLMCrossEntropyLoss",
    ):
        """
        Args:
            model (nn.Module): The (already wrapped) model that owns the ``lm_head``.
                Passed BY_REFERENCE so this loss can borrow the head and switch the
                model into ``skip_lm_head`` mode. Must expose ``lm_head`` and
                ``set_skip_lm_head`` (see GPT2LLM).
            target_key (str): Key of the label tensor in the batch targets.
            prediction_key (str): Key under which the model stores the hidden states.
            num_chunks (int): Number of sequence-dimension chunks. Defaults to 8.
            tag (str): Loss tag. Defaults to "ChunkedCLMCrossEntropyLoss".
        """
        super().__init__(tag)
        if not isinstance(model, nn.Module) or not hasattr(model, "lm_head") or not hasattr(model, "set_skip_lm_head"):
            raise ValueError(
                "ChunkedCLMCrossEntropyLoss requires a single nn.Module exposing `lm_head` and "
                "`set_skip_lm_head` (e.g. GPT2LLM). Pipeline-parallel model parts are not supported."
            )
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.num_chunks = num_chunks
        self._lm_head = model.lm_head
        # Move the head out of the model's forward; the head is applied here instead.
        model.set_skip_lm_head(True)
        # Pure-tensor core (sum reduction). Swapped for a compiled variant by `compile`.
        self.fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = clm_cross_entropy_loss_sum

    def compile(self, backend: str = "inductor") -> None:
        # Compile only the cross-entropy core; the lm_head is intentionally left
        # uncompiled (matches TorchTitan's chunked loss).
        self.fn = torch.compile(self.fn, backend=backend)

    def _chunk_loss(self, hidden_chunk: torch.Tensor, label_chunk: torch.Tensor) -> torch.Tensor:
        # Runs inside checkpoint: the chunk logits produced here are not stored for
        # backward but recomputed, so peak memory holds only one chunk of logits.
        logits = self._lm_head(hidden_chunk)
        return self.fn(logits, label_chunk)

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        hidden_states = forward_batch.get_predictions(self.prediction_key)
        labels = forward_batch.get_targets(self.target_key).to(hidden_states.device)

        # Normalize by the global valid-token count so the summed per-chunk losses
        # equal the mean over valid tokens (clamped to avoid div-by-zero on a fully
        # masked micro-batch).
        num_valid_tokens = (labels != IGNORE_INDEX).sum().clamp(min=1)

        hidden_chunks = torch.chunk(hidden_states, self.num_chunks, dim=1)
        label_chunks = torch.chunk(labels, self.num_chunks, dim=1)

        # When the lm_head is its own FSDP2 unit, keep its parameters unsharded across
        # all chunk (and recompute) calls to avoid a fresh all-gather per chunk, then
        # restore the default behaviour afterwards. Mirrors TorchTitan's
        # ChunkedLossWrapper FSDP handling. No-op when the head is not an FSDPModule
        # (single-device / DDP / head folded into the root FSDP unit).
        head_is_fsdp_unit = isinstance(self._lm_head, FSDP2)
        if head_is_fsdp_unit:
            self._lm_head.set_reshard_after_forward(False)

        try:
            total_loss = hidden_states.new_zeros(())
            for hidden_chunk, label_chunk in zip(hidden_chunks, label_chunks):
                # use_reentrant=False is required for correct grads with non-tensor
                # closure state and is the recommended checkpoint variant.
                total_loss = total_loss + checkpoint(self._chunk_loss, hidden_chunk, label_chunk, use_reentrant=False)
        finally:
            if head_is_fsdp_unit:
                self._lm_head.set_reshard_after_forward(True)
                self._lm_head.reshard()
        return total_loss / num_valid_tokens


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
