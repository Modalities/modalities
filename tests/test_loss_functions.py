import pytest
import torch

from modalities.batch import InferenceResultBatch
from modalities.loss_functions import ClipLoss, CLMCrossEntropyLoss, MultipleFunctionsLoss, NCELoss, nce_loss


@pytest.fixture
def dummy_result_batch() -> InferenceResultBatch:
    predictions = {"embedding": torch.rand(1024, 512)}
    targets = {"target": torch.zeros(1024, 512)}
    batch_dim = 1024
    result_batch = InferenceResultBatch(targets, predictions, batch_dim)
    return result_batch


# calculating asymmetric NCELoss between a batch of embeddings and itself --> zero
@pytest.mark.parametrize("key", ["embedding"])
def test_asymm_NCELoss_is_zero(dummy_result_batch, key):
    loss_func = NCELoss(prediction_key1=key, prediction_key2=key)
    assert loss_func(dummy_result_batch) <= 10e-6


# calculating nce_loss for two randomly generated batch of embeddings (manually calculated)
@pytest.mark.parametrize(
    "embedding1,embedding2",
    [
        (
            torch.Tensor([[0.38, 0.18], [0.36, 0.66], [0.72, 0.09]]),
            torch.Tensor([[0.48, 0.01], [0.54, 0.28], [0.08, 0.34]]),
        )
    ],
)
def test_nce_loss_correctness(embedding1, embedding2):
    unidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=True, temperature=1.0)
    bidirectional_loss = nce_loss(embedding1, embedding2, device="cpu", is_asymmetric=False, temperature=1.0)
    assert unidirectional_loss == pytest.approx(1.1300, 0.0001)
    assert bidirectional_loss == pytest.approx(2.2577, 0.0001)


@pytest.fixture
def clm_cross_entropy_loss_object() -> CLMCrossEntropyLoss:
    return CLMCrossEntropyLoss(target_key="target_ids", prediction_key="logits")


@pytest.fixture
def clip_loss_object() -> ClipLoss:
    return ClipLoss(
        logit_scale_key="logit_scale",
        prediction_keys=["image_cls", "image_text_cls"],
        local_loss=False,
    )


@pytest.fixture
def clip_loss_forward_batch() -> InferenceResultBatch:
    # BATCH SIZE, LENGTH OF SEQUENCE, EMBEDDING SIZE
    predictions = {
        "image_cls": torch.Tensor([[1, 2, 3], [4, 5, 6]]).to("cuda"),
        "image_text_cls": torch.Tensor([[7, 8, 9], [10, 11, 12]]).to("cuda"),
        "logit_scale": 0.07,
    }
    return InferenceResultBatch(targets={}, predictions=predictions)


@pytest.fixture
def setup_distributed(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")

    dist.init_process_group(backend="nccl")
    yield
    dist.destroy_process_group()


def test_clip_loss(clip_loss_object, clip_loss_forward_batch, setup_distributed):

    loss_fn = clip_loss_object
    forward_batch = clip_loss_forward_batch
    loss_fn(clip_loss_forward_batch)


@pytest.fixture
def multiple_functions_loss_object_with_two_losses(
    clm_cross_entropy_loss_object, clip_loss_object
) -> MultipleFunctionsLoss:
    return MultipleFunctionsLoss(
        [clm_cross_entropy_loss_object, clip_loss_object],
        corrsp_weights=[1.0, 1.0],
    )


def test_multiple_functions_loss_initialized_with_single_loss(
    clm_cross_entropy_loss_object,
):
    with pytest.raises(ValueError, match="Number of losses used should be more than 1."):
        MultipleFunctionsLoss([clm_cross_entropy_loss_object], corrsp_weights=[1.0])


def test_multiple_functions_loss_reset_cumulated_individual_losses(
    multiple_functions_loss_object_with_two_losses,
):

    loss = multiple_functions_loss_object_with_two_losses
    num_losses = len(loss.groups)
    loss.cumulated_individual_losses = torch.randn(num_losses)
    loss.reset_cumulated_individual_losses()

    assert (loss.cumulated_individual_losses, torch.zeros(num_losses))


@pytest.fixture
def multiple_functions_loss_forward_batch() -> InferenceResultBatch:

    targets = {"target_ids": torch.Tensor([[1, 2, 1], [1, 1, 2]])}
    predictions = {
        "image_cls": torch.Tensor([[1, 2, 3], [4, 5, 6]]).to("cuda"),
        "image_text_cls": torch.Tensor([[7, 8, 9], [10, 11, 12]]).to("cuda"),
        "logit_scale": 0.07,
        "logits": torch.Tensor(
            [[[0.1, 0.2, 0.7], [0.3, 0.2, 0.5], [0.0, 0.3, 0.7]], [[0.1, 0.2, 0.7], [0.3, 0.2, 0.5], [0.0, 0.3, 0.7]]]
        ),
    }

    return InferenceResultBatch(targets=targets, predictions=predictions)


def test_multiple_functions_loss(
    multiple_functions_loss_object_with_two_losses,
    multiple_functions_loss_forward_batch,
    setup_distributed,
):
    multiple_functions_loss_object_with_two_losses(multiple_functions_loss_forward_batch)
