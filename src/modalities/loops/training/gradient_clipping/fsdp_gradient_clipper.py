import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from modalities.config.lookup_enum import LookupEnum
from modalities.training.gradient_clipping.gradient_clipper import GradientClipperIF


class GradientClippingMode(LookupEnum):
    P1_NORM = 1  # manhattan norm based clipping.
    P2_NORM = 2  # Euclidean norm based clipping.
    MAX_NORM = "inf"  # Maximum norm based clipping.


class FSDPGradientClipper(GradientClipperIF):
    def __init__(self, wrapped_model: FSDP, max_norm: float, norm_type=GradientClippingMode) -> None:
        self.wrapped_model = wrapped_model
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=self.max_norm, norm_type=self.norm_type.value)
        return gradient_norm_score


class FSDPLoggingOnlyGradientClipper(GradientClipperIF):
    def __init__(self, wrapped_model: FSDP, norm_type=GradientClippingMode) -> None:
        self.wrapped_model = wrapped_model
        self.norm_type = norm_type

    def clip_gradients(self) -> torch.Tensor:
        # we only return the gradient norm score without actually clipping the gradients
        gradient_norm_score = self.wrapped_model.clip_grad_norm_(max_norm=torch.inf, norm_type=self.norm_type.value)
        return gradient_norm_score


class DummyGradientClipper(GradientClipperIF):
    def __init__(self) -> None:
        pass

    def clip_gradients(self) -> torch.Tensor:
        gradient_norm_score = torch.Tensor([-1.0])
        return gradient_norm_score
