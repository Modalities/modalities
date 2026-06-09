import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import Loss


class MoECrossEntropyLoss(Loss):
    """Cross Entropy Loss with auxiliary loss support for router balancing"""

    def __init__(
        self,
        target_key: str,
        prediction_key: str,
        model,
        tag: str = "MoECrossEntropyLoss",
    ):
        super().__init__(tag)
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.model = model
        self.loss_fun = CrossEntropyLoss(reduction="mean")

    def __call__(self, forward_batch: InferenceResultBatch) -> torch.Tensor:
        labels = forward_batch.get_targets(self.target_key)
        lm_logits = forward_batch.get_predictions(self.prediction_key)

        labels = labels.to(lm_logits.device)
        loss = self.loss_fun(
            lm_logits.contiguous().view(-1, lm_logits.size(-1)),
            labels.contiguous().long().view(-1),
        )

        # Aux loss
        for layer in self.model.layers.values():
            if hasattr(layer, "aux_loss") and layer.aux_loss is not None:
                loss = loss + layer.aux_loss.to(loss.dtype)

        return loss
