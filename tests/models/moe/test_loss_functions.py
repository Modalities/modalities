import torch
from torch.nn import CrossEntropyLoss

from modalities.batch import InferenceResultBatch
from modalities.models.moe.loss_functions import MoECrossEntropyLoss


class DummyLayer:
    def __init__(self, aux_loss):
        self.aux_loss = aux_loss


class DummyModel:
    def __init__(self, aux_losses: list[torch.Tensor | None]):
        self.layers = {str(i): DummyLayer(aux) for i, aux in enumerate(aux_losses)}


def test_moe_cross_entropy_loss_adds_aux_losses():
    logits = torch.tensor(
        [
            [[1.2, 0.3, -0.5], [0.1, 1.8, -0.3]],
            [[0.5, -0.4, 1.1], [0.7, 0.2, -0.1]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)

    batch = InferenceResultBatch(
        targets={"targets": targets},
        predictions={"logits": logits},
    )

    aux_1 = torch.tensor(0.2)
    aux_2 = torch.tensor(0.3)
    model = DummyModel(aux_losses=[aux_1, None, aux_2])
    loss_fn = MoECrossEntropyLoss(target_key="targets", prediction_key="logits", model=model)

    loss = loss_fn(batch)
    base_ce = CrossEntropyLoss(reduction="mean")(logits.view(-1, logits.size(-1)), targets.view(-1))

    assert torch.allclose(loss, base_ce + aux_1 + aux_2)


def test_moe_cross_entropy_loss_without_aux_matches_plain_ce():
    logits = torch.randn(2, 3, 5)
    targets = torch.randint(0, 5, (2, 3), dtype=torch.long)

    batch = InferenceResultBatch(
        targets={"labels": targets},
        predictions={"pred": logits},
    )

    model = DummyModel(aux_losses=[None, None])
    loss_fn = MoECrossEntropyLoss(target_key="labels", prediction_key="pred", model=model)

    loss = loss_fn(batch)
    expected = CrossEntropyLoss(reduction="mean")(logits.view(-1, logits.size(-1)), targets.view(-1))

    assert torch.allclose(loss, expected)
