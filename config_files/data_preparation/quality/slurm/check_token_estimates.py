#!/usr/bin/env python3
"""Compares the pipeline's token estimates against a real tokenization of the output.

Every figure `preview` reports -- the blend's yield, each dataset's share, whether the run
wraps -- is estimated from text bytes and a per-dataset calibration. Nothing in the pipeline
tokenizes any more, so nothing checks that model unless something like this does.

This tokenizes a sample of the exported JSONL and reports the error per dataset. It is a
sample rather than a full pass because the point is to catch a calibration that is wrong by
tens of percent, not to measure the last percent: the calibration itself was built from
64 KB slices, and a few thousand documents is already far more than that.

An error of a few percent is expected and fine. An error of tens of percent means the
calibration is measuring something other than what the export writes -- a changed tokenizer,
a text field that is not the one being counted, or a dataset whose records shifted shape.

Reads only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from modalities.config.config import load_app_config_dict
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer


def sample_lines(shard_paths: list[Path], n: int, rng: random.Random) -> list[str]:
    """Takes roughly `n` lines spread across a dataset's shards.

    Args:
        shard_paths (list[Path]): The dataset's shards.
        n (int): How many lines to aim for.
        rng (random.Random): Chooses which shards to read.

    Returns:
        list[str]: The sampled lines.
    """
    chosen = rng.sample(shard_paths, min(4, len(shard_paths)))
    per_shard = max(1, n // len(chosen))
    lines: list[str] = []
    for shard in chosen:
        with shard.open("r") as f:
            for i, line in enumerate(f):
                if i >= per_shard:
                    break
                lines.append(line)
    return lines


def main() -> int:
    """Reports estimated against measured tokens.

    Returns:
        int: 0 always; this is a report, and what counts as too much error is a judgement
            the reader makes with the table in front of them.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="export_manifest.yaml")
    parser.add_argument("--mix_manifest", type=Path, required=True, help="mix_manifest.yaml, for the estimates.")
    parser.add_argument("--tokenizer_config", type=Path, required=True, help="Config holding the tokenizer section.")
    parser.add_argument("--sample", type=int, default=2000, help="Lines to tokenize per dataset.")
    parser.add_argument("--text_field", default="text", help="The JSON field holding the document text.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing shards.")
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = yaml.safe_load(f)
    with args.mix_manifest.open() as f:
        mix = yaml.safe_load(f)
    root = args.manifest.parent

    estimated_tokens: dict[str, float] = {}
    estimated_documents: dict[str, int] = {}
    for row in mix["datasets"]:
        name = row.get("source_dataset") or row["name"]
        estimated_tokens[name] = estimated_tokens.get(name, 0.0) + row.get("est_effective_tokens", 0)
        estimated_documents[name] = estimated_documents.get(name, 0) + row["n_documents_kept"]

    section = load_app_config_dict(args.tokenizer_config)["tokenizer"]["config"]
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=section["pretrained_model_name_or_path"],
        padding=section.get("padding", False),
        truncation=section.get("truncation", False),
    )

    rng = random.Random(args.seed)
    print(f"{'dataset':<18} {'sampled':>9} {'est tok/doc':>12} {'real tok/doc':>13} {'error':>9}")
    print("-" * 66)
    total_estimated = total_measured = 0.0
    for dataset in manifest["datasets"]:
        name = dataset["name"]
        shards = sorted((root / name).rglob("*.jsonl"))
        if not shards:
            continue
        lines = sample_lines(shards, args.sample, rng)
        if not lines:
            continue
        measured = sum(len(tokenizer.tokenize(json.loads(line).get(args.text_field, ""))) for line in lines)
        measured_per_line = measured / len(lines)

        # The estimate is per drawn line too: est_effective_tokens already has the repeat
        # factor in it, matching n_lines rather than n_documents.
        estimated_per_line = estimated_tokens[name] / dataset["n_lines"] if dataset["n_lines"] else 0.0
        error = (estimated_per_line - measured_per_line) / measured_per_line if measured_per_line else 0.0
        total_estimated += estimated_per_line * dataset["n_lines"]
        total_measured += measured_per_line * dataset["n_lines"]
        print(
            f"{name:<18} {len(lines):>9,} {estimated_per_line:>12,.1f} {measured_per_line:>13,.1f} "
            f"{error:>8.1%}"
        )
    print("-" * 66)
    overall = (total_estimated - total_measured) / total_measured if total_measured else 0.0
    print(f"{'BLEND':<18} {'':>9} {total_estimated / 1e9:>11,.1f}B {total_measured / 1e9:>12,.1f}B {overall:>8.1%}")
    print()
    print("  A few percent is expected. Tens of percent means the calibration is modelling")
    print("  something other than what the export writes -- check the tokenizer and text field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
