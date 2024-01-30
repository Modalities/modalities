from abc import ABC, abstractmethod

import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch


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
        self.loss_fun = CrossEntropyLoss()

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        labels = forward_batch.get_targets(self.target_key)
        lm_logits = forward_batch.get_predictions(self.prediction_key)

        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        shift_logits = lm_logits.contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        loss = self.loss_fun(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss


class NCELoss(torch.nn.Module):
    def __init__(self, is_asymmetric: bool, temperature: float):
        """
        This calculates the noise contrastive estimation loss between embeddings of two different modalities
        Implementation slightly adapted from https://arxiv.org/pdf/1912.06430.pdf, https://github.com/antoine77340/MIL-NCE_HowTo100M
        changes include adding a temperature value and the choice of calculating asymmetric loss w.r.t. one modality

        Args:
            is_asymmetric (bool): specifies if the loss is calculated in one direction or both directions
            temperature (float): temperature value for regulating loss
        """
        super(NCELoss, self).__init__()
        self.is_asymmetric = is_asymmetric
        self.temperature = temperature

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            embedding1 (torch.Tensor): embeddings from modality 1 of size batch x dim
            embedding2 (torch.Tensor): embeddings from modality 2 of size batch x dim
            device (torch.device): torch device for calculating loss

        Returns:
            torch.Tensor: loss tensor
        """
        # calculating the similarity matrix of size (batch_size x batch_size)
        sim_matrix = torch.matmul(embedding1, embedding2.t()) / self.temperature
        # numerator of loss: using similarity scores for all positive pairs (e.g., image and its caption)
        numerator = sim_matrix * torch.eye(sim_matrix.shape[0], device=device)
        numerator = numerator.sum(dim=0).view(sim_matrix.shape[0], -1)
        numerator = torch.logsumexp(numerator, dim=1)
        # denominator of loss: using all similarity scores for all pairs (positive and negative)
        if self.is_asymmetric:
            denominator = torch.logsumexp(sim_matrix, dim=1)
        else:
            denominator = torch.logsumexp(torch.cat((sim_matrix, sim_matrix.t()), dim=1), dim=1)
        return torch.mean(denominator - numerator)  # calculated in log space


class AsymmNCELoss(Loss):
    def __init__(self, prediction_key1: str, prediction_key2: str, tag: str = "AsymmNCELoss", temperature: float = 1.0):
        """
        Asymmetric noise contrastive estimation loss

        Args:
            prediction_key1 (str): key to access embedding 1
            prediction_key2 (str): key to access embedding 2
            tag (str, optional): defaults to "AsymmNCELoss".
            temperature (float, optional): temperature. Defaults to 1.0.
        """
        super().__init__(tag)
        self.prediction_key1 = prediction_key1
        self.prediction_key2 = prediction_key2
        self.loss_fun = NCELoss(is_asymmetric=True, temperature=temperature)

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Args:
            forward_batch (InferenceResultBatch): data batch

        Returns:
            torch.Tensor: loss tensor
        """
        embedding1 = forward_batch.get_predictions(self.prediction_key1)
        embedding2 = forward_batch.get_predictions(self.prediction_key2)

        shift_embedding1 = embedding1.contiguous()
        shift_embedding2 = embedding2.contiguous()

        loss = self.loss_fun(shift_embedding1, shift_embedding2, embedding1.device)
        return loss
