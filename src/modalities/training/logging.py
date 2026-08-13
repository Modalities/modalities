from typing import Optional
import torch
from modalities.batch import ResultItem


# =============================================================================
# Generic Metrics Accumulator
# =============================================================================

class MetricsAccumulator:
    """Accumulates metrics across batches and produces a single flat tensor
    for cross-rank reduction.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_sum: float = 0.0
        self.scalar_sums: dict[str, float] = {}
        self.per_layer_scalar_sums: dict[str, torch.Tensor] = {}
        self.last_per_layer_vectors: dict[str, torch.Tensor] = {}
        self.per_layer_hist_sums: dict[str, torch.Tensor] = {}
        self.count: int = 0

    def accumulate(self, loss_metrics: dict):
        if "ce_loss" in loss_metrics:
            self.loss_sum += loss_metrics["ce_loss"].item()
        elif "loss" in loss_metrics:
            self.loss_sum += loss_metrics["loss"].item()
        self.count += 1

        bag = loss_metrics.get("metrics")
        if bag is None:
            return

        for name, tensor in bag.get("scalars", {}).items():
            self.scalar_sums[name] = self.scalar_sums.get(name, 0.0) + tensor.item()

        for name, tensor in bag.get("per_layer_scalars", {}).items():
            if name not in self.per_layer_scalar_sums:
                self.per_layer_scalar_sums[name] = torch.zeros_like(tensor, dtype=torch.float32)
            self.per_layer_scalar_sums[name] += tensor.float()

        for name, tensor in bag.get("per_layer_vectors", {}).items():
            self.last_per_layer_vectors[name] = tensor

        for name, tensor in bag.get("per_layer_histograms", {}).items():
            if name not in self.per_layer_hist_sums:
                self.per_layer_hist_sums[name] = torch.zeros_like(tensor, dtype=torch.float32)
            self.per_layer_hist_sums[name] += tensor.float()

    def build_sync_tensor(
        self, device: torch.device
    ) -> tuple[
        torch.Tensor,
        list[str],
        list[str],
        dict[str, int],
        list[str],
        dict[str, tuple],
    ]:
        if self.count == 0:
            return torch.zeros(1, device=device), [], [], {}, [], {}

        n = self.count
        values = [self.loss_sum / n]

        scalar_names = sorted(self.scalar_sums.keys())
        for name in scalar_names:
            values.append(self.scalar_sums[name] / n)

        per_layer_names = sorted(self.per_layer_scalar_sums.keys())
        per_layer_sizes = {}
        layer_tensors = []
        for name in per_layer_names:
            t = self.per_layer_scalar_sums[name] / n
            layer_tensors.append(t.to(device))
            per_layer_sizes[name] = t.numel()

        hist_names = sorted(self.per_layer_hist_sums.keys())
        hist_shapes: dict[str, tuple] = {}
        hist_tensors = []
        for name in hist_names:
            t = self.per_layer_hist_sums[name] / n
            hist_shapes[name] = tuple(t.shape)
            hist_tensors.append(t.to(device).flatten())

        combined = torch.tensor(values, device=device, dtype=torch.float32)
        if layer_tensors:
            combined = torch.cat([combined, torch.cat(layer_tensors)])
        if hist_tensors:
            combined = torch.cat([combined, torch.cat(hist_tensors)])

        return combined, scalar_names, per_layer_names, per_layer_sizes, hist_names, hist_shapes

    @staticmethod
    def unpack_synced_tensor(
        synced: torch.Tensor,
        scalar_names: list[str],
        per_layer_names: list[str],
        per_layer_sizes: dict[str, int],
        hist_names: list[str] = None,
        hist_shapes: dict[str, tuple] = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        hist_names = hist_names or []
        hist_shapes = hist_shapes or {}

        idx = 0
        loss = synced[idx]; idx += 1

        scalars = {}
        for name in scalar_names:
            scalars[name] = synced[idx]; idx += 1

        per_layer_scalars = {}
        for name in per_layer_names:
            size = per_layer_sizes[name]
            per_layer_scalars[name] = synced[idx : idx + size]; idx += size

        per_layer_histograms = {}
        for name in hist_names:
            shape = hist_shapes[name]
            size = 1
            for dim in shape:
                size *= dim
            per_layer_histograms[name] = synced[idx : idx + size].reshape(shape)
            idx += size

        return loss, scalars, per_layer_scalars, per_layer_histograms


# =============================================================================
# Metrics Formatter
# =============================================================================

def format_metrics(
    loss: torch.Tensor,
    scalars: dict[str, torch.Tensor],
    per_layer_scalars: dict[str, torch.Tensor],
    per_layer_vectors: dict[str, torch.Tensor],
    summary_only: bool = False,
    per_layer_histograms: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[dict[str, ResultItem], dict[str, ResultItem]]:
    per_layer_histograms = per_layer_histograms or {}

    losses = {
        "loss/ce_avg": ResultItem(loss, decimal_places=2),
    }

    metrics: dict[str, ResultItem] = {}

    for name, val in scalars.items():
        metrics[f"adaptive/{name}"] = ResultItem(val, 4)

    for name, vals in per_layer_scalars.items():
        metrics[f"summary/{name}"] = ResultItem(vals.mean(), 4)
        if not summary_only:
            for i, v in enumerate(vals):
                metrics[f"layer_{i}/{name}"] = ResultItem(v, 4)

    for name, tensor in per_layer_vectors.items():
        if tensor.numel() == 0:
            continue
        t = tensor.float().cpu()
        n_layers, n_loops = t.shape

        metrics[f"summary/{name}"] = ResultItem(t.mean(), 4)

        if not summary_only:
            for i in range(n_layers):
                metrics[f"layer_{i}/avg_{name}"] = ResultItem(t[i].mean(), 4)
                for j in range(n_loops):
                    metrics[f"layer_{i}/{name}_{j}"] = ResultItem(t[i, j], 4)

            for j in range(n_loops):
                metrics[f"loop_{j}/{name}"] = ResultItem(t[:, j].mean(), 4)

    for name, tensor in per_layer_histograms.items():
        if tensor.numel() == 0:
            continue
        t = tensor.float().cpu()
        n_layers, n_bins = t.shape

        for b in range(n_bins):
            metrics[f"hist/{name}/bin_{b}"] = ResultItem(t[:, b].mean(), 4)

        if not summary_only:
            for i in range(n_layers):
                for b in range(n_bins):
                    metrics[f"hist/{name}/layer_{i}/bin_{b}"] = ResultItem(t[i, b], 4)

    return losses, metrics
