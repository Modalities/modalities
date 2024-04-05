from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from modalities.config.config import GradientClippingMode


def build_gradient_clipper(
    gradient_clipping_mode: GradientClippingMode, gradient_clipping_threshold: Optional[float]
) -> Callable[[nn.Module], torch.Tensor]:
    """Returns a function that applies gradient clipping to a given model (in place).

    :param gradient_clipping_mode: Selection between different norm based modes,
        value based clipping and no clipping
    :type gradient_clipping_mode: GradientClippingMode
    :param gradient_clipping_threshold: Value at which will be clipped.
    :type gradient_clipping_threshold: float
    :return: A function taking a model as input and producing no output.
    :rtype: Callable[[nn.Module], None]
    """
    if (gradient_clipping_threshold is None) != (gradient_clipping_mode == GradientClippingMode.NONE):
        raise ValueError(
            "Either gradient clipping is deactivated and no threshold given or activated and a threshold set."
        )
    if gradient_clipping_mode == GradientClippingMode.P1_NORM:
        return lambda model: clip_grad_norm_(
            model.parameters(), max_norm=gradient_clipping_threshold, norm_type=1
        ).sum()
    if gradient_clipping_mode == GradientClippingMode.P2_NORM:
        return lambda model: clip_grad_norm_(
            model.parameters(), max_norm=gradient_clipping_threshold, norm_type=2
        ).sum()
    if gradient_clipping_mode == GradientClippingMode.MAX_NORM:
        return lambda model: clip_grad_norm_(
            model.parameters(), max_norm=gradient_clipping_threshold, norm_type="inf"
        ).sum()
    if gradient_clipping_mode == GradientClippingMode.VALUE:

        def norm_calc(model: nn.Module) -> torch.Tensor:
            # we just calculate the sum of the gradients' absolute value
            # (max_norm=torch.inf makes sure that we don't clip anything)
            gradient_norm_score = clip_grad_norm_(model.parameters(), max_norm=torch.inf, norm_type=1).sum()
            clip_grad_value_(model.parameters(), gradient_clipping_threshold)
            return gradient_norm_score

        return norm_calc
    # we just calculate the sum of the gradients' absolute value
    # (max_norm=torch.inf makes sure that we don't clip anything)
    return lambda model: clip_grad_norm_(model.parameters(), max_norm=torch.inf, norm_type=1).sum()
