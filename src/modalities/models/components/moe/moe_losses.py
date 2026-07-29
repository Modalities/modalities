"""Loss components for mixture-of-experts models.

Nemotron-3 Nano combines the language modelling objective with a small load-balancing penalty. The
penalty itself is computed inside each MoE layer (it needs the router scores) and surfaced through
the model output dict; the components here read it back out and combine it with the main loss.
"""

from typing import Annotated

import torch
from pydantic import BaseModel, Field, model_validator

from modalities.batch import InferenceResultBatch
from modalities.config.pydantic_if_types import PydanticLossIFType
from modalities.loss_functions import Loss


class MoEAuxLoss(Loss):
    """
    Reads the pre-computed mixture-of-experts auxiliary loss out of the model output.

    The value is produced by the MoE layers during the forward pass (already multiplied by the
    per-layer coefficient) and summed by the model. This class only surfaces it as a
    :class:`~modalities.loss_functions.Loss` so that it can be logged and combined like any other
    loss term.
    """

    def __init__(self, prediction_key: str, tag: str = "MoEAuxLoss"):
        """
        Initializes the MoEAuxLoss.

        Args:
            prediction_key (str): Key under which the model stores the summed auxiliary loss.
            tag (str): Human-readable tag used for logging.
        """
        super().__init__(tag)
        self.prediction_key = prediction_key

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Returns the auxiliary loss recorded in the forward batch.

        Args:
            forward_batch (InferenceResultBatch): The model output.

        Returns:
            torch.Tensor: The scalar auxiliary loss.
        """
        return forward_batch.get_predictions(self.prediction_key)


class WeightedSumLoss(Loss):
    """
    A weighted sum of several losses.

    Used to add the MoE load-balancing penalty to the language modelling loss without changing the
    training loop, which evaluates exactly one loss component.
    """

    def __init__(self, losses: list[Loss], weights: list[float], tag: str = "WeightedSumLoss"):
        """
        Initializes the WeightedSumLoss.

        Args:
            losses (list[Loss]): The losses to combine.
            weights (list[float]): One weight per loss.
            tag (str): Human-readable tag used for logging.

        Raises:
            ValueError: If the number of losses and weights differ, or no loss is given.
        """
        super().__init__(tag)
        if len(losses) == 0:
            raise ValueError("WeightedSumLoss requires at least one loss.")
        if len(losses) != len(weights):
            raise ValueError(f"Got {len(losses)} losses but {len(weights)} weights; they must match.")
        self.losses = losses
        self.weights = weights

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        """
        Computes the weighted sum of the configured losses.

        Args:
            forward_batch (InferenceResultBatch): The model output.

        Returns:
            torch.Tensor: The combined scalar loss.
        """
        total = None
        for loss, weight in zip(self.losses, self.weights, strict=True):
            term = weight * loss(forward_batch)
            total = term if total is None else total + term
        return total


class MoEAuxLossConfig(BaseModel):
    """
    Configuration of :class:`MoEAuxLoss`.

    Attributes:
        prediction_key (str): Key under which the model stores the summed auxiliary loss. Must match
            the model's ``aux_loss_key``.
        tag (str): Human-readable tag used for logging.
    """

    prediction_key: str
    tag: str = "MoEAuxLoss"


class WeightedSumLossConfig(BaseModel):
    """
    Configuration of :class:`WeightedSumLoss`.

    Attributes:
        losses (list[Loss]): The losses to combine.
        weights (list[float]): One weight per loss.
        tag (str): Human-readable tag used for logging.
    """

    losses: list[PydanticLossIFType]
    weights: list[Annotated[float, Field(strict=True)]]
    tag: str = "WeightedSumLoss"

    @model_validator(mode="after")
    def _validate_lengths(self) -> "WeightedSumLossConfig":
        if len(self.losses) != len(self.weights):
            raise ValueError(f"Got {len(self.losses)} losses but {len(self.weights)} weights; they must match.")
        return self
