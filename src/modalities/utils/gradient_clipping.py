from typing import Callable, Optional

from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from modalities.config.config import GradientClippingMode
from modalities.models.model import NNModel


def build_gradient_clipper(
    gradient_clipping_mode: GradientClippingMode, gradient_clipping_threshold: Optional[float]
) -> Callable[[NNModel], None]:
    """Returns a function that applies gradient clipping to a given model (in place).

    :param gradient_clipping_mode: Selection between different norm based modes,
        value based clipping and no clipping
    :type gradient_clipping_mode: GradientClippingMode
    :param gradient_clipping_threshold: Value at which will be clipped.
    :type gradient_clipping_threshold: float
    :return: A function taking a model as input and producing no output.
    :rtype: Callable[[NNModel], None]
    """
    if (gradient_clipping_threshold is None) != (gradient_clipping_mode == GradientClippingMode.NONE):
        raise ValueError(
            "Either gradient clipping is deactivated and no threshold given or activated and a threshold set."
        )
    if gradient_clipping_mode == GradientClippingMode.P2_NORM:
        # Always return None to satisfy the Callable[[NNModel], None] interface.
        return lambda model: (clip_grad_norm_(model.parameters(), gradient_clipping_threshold, 2), None)[-1]
    if gradient_clipping_mode == GradientClippingMode.MAX_NORM:
        # Always return None to satisfy the Callable[[NNModel], None] interface.
        return lambda model: (clip_grad_norm_(model.parameters(), gradient_clipping_threshold, "inf"), None)[-1]
    if gradient_clipping_mode == GradientClippingMode.VALUE:
        return lambda model: clip_grad_value_(model.parameters(), gradient_clipping_threshold)
    return lambda model: None
