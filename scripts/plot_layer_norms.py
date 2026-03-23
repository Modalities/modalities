#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot parameter norms across checkpoints from a JSON log file.")
    parser.add_argument(
        "--layer-norms-json-path",
        type=Path,
        required=True,
        help="Path to JSON produced by scripts/compute_layer_norms.py.",
    )
    parser.add_argument(
        "--plot-output-path",
        type=Path,
        default=Path("layer_norms_across_checkpoints.png"),
        help="Output image path for cross-checkpoint layer-norm visualization.",
    )
    parser.add_argument(
        "--layer-filter-regex",
        type=str,
        default=r".*",
        help="Regex to select layer keys in the visualization.",
    )
    return parser.parse_args()


def _load_results(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list) or not results:
        raise ValueError("Expected a non-empty JSON list of checkpoint results.")
    return results


def _plot_checkpoint_comparison(
    results: list[dict],
    plot_output_path: Path,
    layer_filter_regex: str,
) -> None:
    metric_key = "parameter_norms" if "parameter_norms" in results[0] else "layer_norms"
    layer_pattern = re.compile(layer_filter_regex)
    filtered_layers = sorted(
        {
            layer_name
            for checkpoint_result in results
            for layer_name in checkpoint_result[metric_key].keys()
            if layer_pattern.search(layer_name)
        }
    )
    if not filtered_layers:
        raise ValueError(f"No layer names matched --layer-filter-regex={layer_filter_regex!r}.")

    checkpoint_labels = [checkpoint_result["checkpoint_label"] for checkpoint_result in results]
    matrix = torch.tensor(
        [
            [checkpoint_result[metric_key].get(layer_name, float("nan")) for layer_name in filtered_layers]
            for checkpoint_result in results
        ],
        dtype=torch.float32,
    )

    fig_width = max(12, 0.55 * len(checkpoint_labels))
    fig_height = max(8, 0.25 * len(filtered_layers))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.T.numpy(), aspect="auto", interpolation="nearest")

    ax.set_title("Parameter Norms Across Checkpoints")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Parameter")
    ax.set_xticks(range(len(checkpoint_labels)))
    ax.set_xticklabels(checkpoint_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(filtered_layers)))
    ax.set_yticklabels(filtered_layers)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("L2 norm")

    fig.tight_layout()
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results = _load_results(args.layer_norms_json_path)
    _plot_checkpoint_comparison(
        results=results,
        plot_output_path=args.plot_output_path,
        layer_filter_regex=args.layer_filter_regex,
    )
    print(f"Saved cross-checkpoint parameter-norm plot to {args.plot_output_path}")


if __name__ == "__main__":
    main()
