#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
        default=Path("parameter_norms_grouped_by_layer.pdf"),
        help="Output PDF path containing one plot page per layer.",
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


def _extract_layer_key(parameter_name: str) -> str:
    tokens = parameter_name.split(".")
    for i in range(len(tokens) - 1):
        if tokens[i] in {"h", "layers", "blocks"} and tokens[i + 1].isdigit():
            if i > 0:
                return ".".join(tokens[i - 1 : i + 2])
            return ".".join(tokens[i : i + 2])
    return ".".join(tokens[:-1]) if len(tokens) > 1 else parameter_name


def _layer_sort_key(layer_key: str) -> tuple:
    # Prefer numeric ordering for transformer block keys like h.0, layers.12, blocks.3.
    match = re.search(r"(?:^|\.)(h|layers|blocks)\.(\d+)(?:\.|$)", layer_key)
    if match:
        return (0, match.group(1), int(match.group(2)), layer_key)
    return (1, layer_key)


def _plot_checkpoint_comparison(
    results: list[dict],
    plot_output_path: Path,
    layer_filter_regex: str,
) -> None:
    metric_key = "parameter_norms" if "parameter_norms" in results[0] else "layer_norms"
    layer_pattern = re.compile(layer_filter_regex)
    filtered_parameters = sorted(
        {
            parameter_name
            for checkpoint_result in results
            for parameter_name in checkpoint_result[metric_key].keys()
            if layer_pattern.search(parameter_name)
        }
    )
    if not filtered_parameters:
        raise ValueError(f"No layer names matched --layer-filter-regex={layer_filter_regex!r}.")

    checkpoint_labels = [checkpoint_result["checkpoint_label"] for checkpoint_result in results]

    grouped_parameters: dict[str, list[str]] = {}
    for parameter_name in filtered_parameters:
        layer_key = _extract_layer_key(parameter_name)
        grouped_parameters.setdefault(layer_key, []).append(parameter_name)
    ordered_layer_keys = sorted(grouped_parameters, key=_layer_sort_key)

    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(plot_output_path) as pdf:
        # First page: quick summary of layers and parameter counts.
        summary_lines = [
            f"checkpoints: {len(checkpoint_labels)}",
            f"layers: {len(grouped_parameters)}",
            f"parameters plotted: {len(filtered_parameters)}",
            "",
            "Layer -> #parameters",
        ]
        for layer_key in ordered_layer_keys:
            summary_lines.append(f"{layer_key}: {len(grouped_parameters[layer_key])}")

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.axis("off")
        ax.text(0.01, 0.99, "\n".join(summary_lines), va="top", ha="left", fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # One page per layer with all parameter curves for that layer.
        x = list(range(len(checkpoint_labels)))
        for layer_key in ordered_layer_keys:
            parameter_names = sorted(grouped_parameters[layer_key])
            fig, ax = plt.subplots(figsize=(12, 6))
            for parameter_name in parameter_names:
                y = [checkpoint_result[metric_key].get(parameter_name, float("nan")) for checkpoint_result in results]
                short_name = (
                    parameter_name[len(layer_key) + 1 :]
                    if parameter_name.startswith(layer_key + ".")
                    else parameter_name
                )
                ax.plot(x, y, marker="o", linewidth=1.5, label=short_name)

            ax.set_title(f"{layer_key} parameter norms across checkpoints")
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("L2 norm")
            ax.set_xticks(x)
            ax.set_xticklabels(checkpoint_labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = _parse_args()
    results = _load_results(args.layer_norms_json_path)
    _plot_checkpoint_comparison(
        results=results,
        plot_output_path=args.plot_output_path,
        layer_filter_regex=args.layer_filter_regex,
    )
    print(f"Saved grouped parameter-norm plots to {args.plot_output_path}")


if __name__ == "__main__":
    main()
