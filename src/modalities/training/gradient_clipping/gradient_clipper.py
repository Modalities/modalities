from abc import ABC, abstractmethod

import torch


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
