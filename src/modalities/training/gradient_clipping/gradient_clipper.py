from abc import ABC, abstractmethod

import torch


class GradientClipperIF(ABC):
    @abstractmethod
    def clip_gradients(self) -> torch.Tensor:
        raise NotImplementedError
