from abc import ABC, abstractmethod
from typing import Optional, overload

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch
from modalities.running_env.fsdp.device_mesh import (
    ParallelismDegrees,
    get_mesh_for_parallelism_method,
    has_parallelism_method,
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


class MoECrossEntropyLoss(Loss):
    """Cross entropy loss plus the global-batch load balancing loss (and router z-loss) of the MoE layers."""

    def __init__(
        self,
        target_key: str,
        prediction_key: str,
        model,
        device_mesh: Optional[DeviceMesh] = None,
        tag: str = "MoECrossEntropyLoss",
    ):
        """
        Args:
            target_key (str): Key of the targets in the batch.
            prediction_key (str): Key of the logits in the model output.
            model: The (unwrapped or FSDP-wrapped) MoE model whose MoE blocks provide the router statistics.
            device_mesh (Optional[DeviceMesh]): Device mesh; the load balancing token counts are aggregated over
                its data parallel dimensions. If None, they are aggregated over the default (world) group.
            tag (str): Tag of the loss.
        """
        from modalities.models.moe.qwen_model import GlobalBatchLoadBalancingLoss

        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.loss_fun = CrossEntropyLoss(reduction="mean")
        process_groups = None
        if device_mesh is not None:
            process_groups = [
                get_mesh_for_parallelism_method(device_mesh, method).get_group()
                for method in (ParallelismDegrees.DP_REPLICATE, ParallelismDegrees.DP_SHARD)
                if has_parallelism_method(device_mesh, method)
            ]
        self.load_balancing_loss = GlobalBatchLoadBalancingLoss(model=model, process_groups=process_groups)

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        labels = forward_batch.get_targets(self.target_key)
        lm_logits = forward_batch.get_predictions(self.prediction_key)

        labels = labels.to(lm_logits.device)
        loss = self.loss_fun(
            lm_logits.contiguous().view(-1, lm_logits.size(-1)),
            labels.contiguous().long().view(-1),
        )

        moe_loss = self.load_balancing_loss()
        if moe_loss is not None:
            loss = loss + moe_loss.to(loss.dtype)
        return loss


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
