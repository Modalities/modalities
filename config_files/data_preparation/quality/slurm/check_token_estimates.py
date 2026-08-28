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

**Trust the blend row; treat a single dataset's row as indicative.** Sampling in proportion
to length over-weights very large documents, and on a corpus whose bytes-per-token ratio
spans an order of magnitude with the tail holding much of the mass -- FineWiki runs 3.571 for
documents under a kilobyte to 34.648 for the nine above 256 KB -- the measured tokens-per-byte
comes out low and the estimate looks inflated. Measured on the real blend: finewiki-es
reported +51.1% on one seed and +8.9% on another, while the blend total moved only from -0.8%
to -1.1%. Re-run with a different --seed before believing any single dataset's number.

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


def sample_lines(shard_paths: list[Path], n: int, rng: random.Random) -> list[bytes]:
    """Samples lines at byte offsets spread across a dataset's shards.

    Seeks to a random offset, discards the partial line it lands in, and takes the next whole
    one. That selects documents in proportion to their length, which is what makes the
    measured tokens-per-byte ratio comparable to the estimate -- the calibration is itself a
    bytes-to-tokens ratio.

    Reading a prefix instead, which this did at first, is the bias the calibration work
    already had to fix once: the opening lines of a shard are not a random draw from it, and
    on the real blend that produced errors from -65% to +106% with no consistent sign.

    Args:
        shard_paths (list[Path]): The dataset's shards.
        n (int): How many lines to aim for.
        rng (random.Random): Chooses shards and offsets.

    Returns:
        list[bytes]: The sampled lines, without terminators.
    """
    chosen = rng.sample(shard_paths, min(8, len(shard_paths)))
    per_shard = max(1, n // len(chosen))
    lines: list[bytes] = []
    for shard in chosen:
        size = shard.stat().st_size
        if size < 2:
            continue
        with shard.open("rb") as f:
            for _ in range(per_shard):
                f.seek(rng.randrange(size))
                f.readline()          # discard the partial line this offset fell inside
                line = f.readline()   # the document containing the next boundary
                if not line:
                    f.seek(0)
                    line = f.readline()
                if line.strip():
                    lines.append(line.rstrip(b"\n"))
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
    print(f"{'dataset':<18} {'sampled':>9} {'est tok/kB':>11} {'real tok/kB':>12} {'est tokens':>13} "
          f"{'implied':>13} {'error':>8}")
    print("-" * 84)
    total_estimated = total_measured = 0.0
    for dataset in manifest["datasets"]:
        name = dataset["name"]
        shards = sorted((root / name).rglob("*.jsonl"))
        if not shards:
            continue
        lines = sample_lines(shards, args.sample, rng)
        if not lines:
            continue
        sampled_tokens = sampled_bytes = 0
        for line in lines:
            try:
                text = json.loads(line).get(args.text_field, "")
            except ValueError:
                continue
            sampled_tokens += len(tokenizer.tokenize(text)) if isinstance(text, str) else 0
            sampled_bytes += len(line) + 1  # the newline the export writes

        if not sampled_bytes:
            continue
        # Both sides are byte-weighted, which is what makes them comparable: the estimate is
        # a bytes-to-tokens ratio, and the sample is drawn in proportion to length.
        measured_ratio = sampled_tokens / sampled_bytes
        estimated_ratio = estimated_tokens[name] / dataset["n_bytes"] if dataset["n_bytes"] else 0.0
        implied = measured_ratio * dataset["n_bytes"]
        error = (estimated_tokens[name] - implied) / implied if implied else 0.0
        total_estimated += estimated_tokens[name]
        total_measured += implied
        print(
            f"{name:<18} {len(lines):>9,} {estimated_ratio * 1000:>11,.1f} {measured_ratio * 1000:>12,.1f} "
            f"{estimated_tokens[name] / 1e9:>12,.1f}B {implied / 1e9:>12,.1f}B {error:>7.1%}"
        )
    print("-" * 84)
    overall = (total_estimated - total_measured) / total_measured if total_measured else 0.0
    print(f"{'BLEND':<18} {'':>9} {'':>11} {'':>12} {total_estimated / 1e9:>12,.1f}B "
          f"{total_measured / 1e9:>12,.1f}B {overall:>7.1%}")
    print()
    print("  A few percent is expected. Tens of percent means the calibration is modelling")
    print("  something other than what the export writes -- check the tokenizer and text field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
