import torch
import pytest
from modalities.training.logging import MetricsAccumulator, format_metrics
from modalities.batch import ResultItem


def test_metrics_accumulator_accumulation_and_sync():
    device = torch.device("cpu")
    accum = MetricsAccumulator()

    # Step 1: Accumulate first batch
    metrics_1 = {
        "scalars": {
            "p_weight": torch.tensor(0.5)
        },
        "per_layer_scalars": {
            "cost": torch.tensor([1.0, 2.0])
        },
        "per_layer_vectors": {
            "vec": torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        },
        "per_layer_histograms": {
            "hist": torch.tensor([[0.5, 0.5], [0.8, 0.2]])
        }
    }
    accum.accumulate({"ce_loss": torch.tensor(2.0), "metrics": metrics_1})

    # Step 2: Accumulate second batch
    metrics_2 = {
        "scalars": {
            "p_weight": torch.tensor(1.5)
        },
        "per_layer_scalars": {
            "cost": torch.tensor([3.0, 4.0])
        },
        "per_layer_vectors": {
            # Vectors are last-batch-only in trainer design
            "vec": torch.tensor([[1.1, 1.2], [1.3, 1.4]])
        },
        "per_layer_histograms": {
            "hist": torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        }
    }
    accum.accumulate({"ce_loss": torch.tensor(4.0), "metrics": metrics_2})

    assert accum.count == 2

    # Step 3: Build sync tensor
    sync_tensor, scalar_names, pl_names, pl_sizes, hist_names, hist_shapes = accum.build_sync_tensor(device)

    # Expected averages:
    # ce_loss = (2 + 4) / 2 = 3.0
    # scalars: p_weight = (0.5 + 1.5) / 2 = 1.0
    # per_layer_scalars: cost = [(1+3)/2, (2+4)/2] = [2.0, 3.0]
    # per_layer_histograms: hist = [[(0.5+0.3)/2, (0.5+0.7)/2], [(0.8+0.6)/2, (0.2+0.4)/2]] = [[0.4, 0.6], [0.7, 0.3]]

    # Step 4: Unpack
    loss, scalars, pl_scalars, pl_hist = MetricsAccumulator.unpack_synced_tensor(
        sync_tensor, scalar_names, pl_names, pl_sizes, hist_names, hist_shapes
    )

    assert torch.allclose(loss, torch.tensor(3.0))
    assert torch.allclose(scalars["p_weight"], torch.tensor(1.0))
    assert torch.allclose(pl_scalars["cost"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(pl_hist["hist"], torch.tensor([[0.4, 0.6], [0.7, 0.3]]))
    assert torch.allclose(accum.last_per_layer_vectors["vec"], torch.tensor([[1.1, 1.2], [1.3, 1.4]]))


def test_format_metrics():
    loss = torch.tensor(3.0)
    scalars = {"p_weight": torch.tensor(1.0)}
    pl_scalars = {"cost": torch.tensor([2.0, 3.0])}
    pl_vectors = {"vec": torch.tensor([[1.1, 1.2], [1.3, 1.4]])}
    pl_hists = {"hist": torch.tensor([[0.4, 0.6], [0.7, 0.3]])}

    # Test summary_only = False
    losses, metrics = format_metrics(
        loss=loss,
        scalars=scalars,
        per_layer_scalars=pl_scalars,
        per_layer_vectors=pl_vectors,
        summary_only=False,
        per_layer_histograms=pl_hists
    )

    assert losses["loss/ce_avg"].value.item() == pytest.approx(3.0)
    assert metrics["adaptive/p_weight"].value.item() == pytest.approx(1.0)
    assert metrics["summary/cost"].value.item() == pytest.approx(2.5)
    assert metrics["layer_0/cost"].value.item() == pytest.approx(2.0)
    assert metrics["layer_1/cost"].value.item() == pytest.approx(3.0)

    # Vectors
    assert metrics["summary/vec"].value.item() == pytest.approx(1.25)
    assert metrics["layer_0/vec_0"].value.item() == pytest.approx(1.1)
    assert metrics["layer_0/vec_1"].value.item() == pytest.approx(1.2)
    assert metrics["layer_1/vec_0"].value.item() == pytest.approx(1.3)
    assert metrics["layer_1/vec_1"].value.item() == pytest.approx(1.4)

    # Histograms
    assert metrics["hist/hist/bin_0"].value.item() == pytest.approx(0.55)
    assert metrics["hist/hist/bin_1"].value.item() == pytest.approx(0.45)
    assert metrics["hist/hist/layer_0/bin_0"].value.item() == pytest.approx(0.4)
    assert metrics["hist/hist/layer_1/bin_1"].value.item() == pytest.approx(0.3)

    # Test summary_only = True
    losses, metrics = format_metrics(
        loss=loss,
        scalars=scalars,
        per_layer_scalars=pl_scalars,
        per_layer_vectors=pl_vectors,
        summary_only=True,
        per_layer_histograms=pl_hists
    )

    assert "layer_0/cost" not in metrics
    assert "layer_0/vec_0" not in metrics
    assert "hist/hist/layer_0/bin_0" not in metrics
    assert metrics["summary/cost"].value.item() == pytest.approx(2.5)
    assert metrics["summary/vec"].value.item() == pytest.approx(1.25)
    assert metrics["hist/hist/bin_0"].value.item() == pytest.approx(0.55)
