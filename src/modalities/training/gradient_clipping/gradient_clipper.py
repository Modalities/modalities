from abc import ABC, abstractmethod

import torch

from modalities.config.lookup_enum import LookupEnum


class GradientClippingMode(LookupEnum):
    """
    Enum class representing different modes of gradient clipping.

    Attributes:
        P1_NORM (int): Mode for Manhattan norm based clipping.
        P2_NORM (int): Mode for Euclidean norm based clipping.
        MAX_NORM (str): Mode for maximum norm based clipping.
    """

    P1_NORM = 1
    P2_NORM = 2
    MAX_NORM = "inf"


class GradientClipperIF(ABC):
    """The GradientClipper interface that defines the methods for clipping gradients."""

    @abstractmethod
    def clip_gradients(self) -> torch.Tensor:
        """
        Clip the gradients of the model.

        Returns:
            torch.Tensor: The clipped gradients.
        """
        raise NotImplementedError
