from typing import Callable

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from modalities.config.config import GradientClippingMode
from modalities.models.model import NNModel


def build_gradient_clipper(
    gradient_clipping_mode: GradientClippingMode, gradient_clipping_threshold: float
) -> Callable[[NNModel], None]:
    if gradient_clipping_mode == GradientClippingMode.P2_NORM:
        return lambda model: (clip_grad_norm_(model.parameters(), gradient_clipping_threshold, 2), None)[-1]
    if gradient_clipping_mode == GradientClippingMode.MAX_NORM:
        return lambda model: (clip_grad_norm_(model.parameters(), gradient_clipping_threshold, "inf"), None)[-1]
    if gradient_clipping_mode == GradientClippingMode.VALUE:
        return lambda model: clip_grad_value_(model.parameters(), gradient_clipping_threshold)
    return lambda model: None
