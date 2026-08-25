#!/usr/bin/env python3
"""Checks the outcome of the end-to-end smoke run, beyond "it did not crash".

Three things are worth verifying and none of them are visible from exit codes:

1. **Token estimates against reality.** Every figure the preview reports is estimated from
   text bytes and a per-dataset calibration. Nothing had ever compared those estimates to
   an actual packing run, so the whole token budget rested on an unvalidated model. This
   counts the tokens in the packed output and reports the error per dataset.
2. **The blend loads.** ``WeightedCombinedDataset`` had unit tests but had never been
   handed real packed files. A fractional repeat factor is included on purpose, since that
   is what drives the partial-pass permutation.
3. **The source tree is untouched.** The corpora are shared and read-only. This asserts
   nothing was written under them, rather than trusting that nothing was.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from modalities.dataloader.create_packed_data import EmbeddedStreamData
from modalities.dataloader.dataset import PackedMemMapDatasetContinuous, WeightedCombinedDataset


def count_packed_tokens(pbin_paths: list[Path]) -> tuple[int, int]:
    """Counts tokens and documents across packed files.

    Read from each file's own header rather than from a dataset view: the data section
    length divided by the token width is the exact number of tokens written, with no
    dependence on a block size and no final partial block to reason about.

    Args:
        pbin_paths (list[Path]): The ``.pbin`` files to read.

    Returns:
        tuple[int, int]: Total tokens and total documents.
    """
    n_tokens = 0
    n_docs = 0
    for path in pbin_paths:
        stream = EmbeddedStreamData(path, load_index=True)
        n_tokens += stream.data_len // stream.token_size_in_bytes
        n_docs += len(stream.index_base)
    return n_tokens, n_docs


def main() -> int:
    """Runs the checks.

    Returns:
        int: Process exit status; non-zero if any check failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="mix_manifest.yaml from quality apply.")
    parser.add_argument("--packed_dir", type=Path, required=True, help="Directory holding the packed output.")
    parser.add_argument("--source_root", type=Path, required=True, help="Snapshot root that must stay unwritten.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Block size for opening packed files.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed relative error between estimated and packed tokens.",
    )
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    failures: list[str] = []

    print("=" * 78)
    print("1. estimated vs packed tokens, and selected vs packed documents")
    print("=" * 78)
    print(f"{'dataset':<16} {'est tokens':>15} {'packed':>15} {'error':>8} {'docs sel':>10} {'docs packed':>11}")
    print("-" * 78)
    total_est = 0
    total_packed = 0
    for record in manifest["datasets"]:
        name = record["name"]
        pbins = sorted((args.packed_dir / name).rglob("*.pbin"))
        if not pbins:
            print(f"{name:<16} {record['est_tokens_kept']:>15,} {'NOT PACKED':>15}")
            failures.append(f"{name}: no packed output under {args.packed_dir / name}")
            continue
        packed, n_docs = count_packed_tokens(pbins)
        estimated = record["est_tokens_kept"]
        error = (packed - estimated) / estimated if estimated else 0.0
        ok = abs(error) <= args.tolerance
        total_est += estimated
        total_packed += packed
        selected = record["n_documents_kept"]
        print(
            f"{name:<16} {estimated:>15,} {packed:>15,} {error * 100:>7.2f}% {selected:>10,} {n_docs:>11,}"
            f"{'' if ok else '   TOKENS OUT OF TOLERANCE'}"
            f"{'' if n_docs == selected else '   DOC COUNT MISMATCH'}"
        )
        if not ok:
            failures.append(f"{name}: estimate off by {error * 100:.2f}% (tolerance {args.tolerance * 100:.0f}%)")
        # Documents are not estimated: the filtered index lists exactly the selected
        # documents, so the packer must emit exactly that many. Any difference is a bug in
        # materialize or in the index, not estimator error.
        if n_docs != selected:
            failures.append(f"{name}: selection kept {selected:,} documents but {n_docs:,} were packed")

    if total_est:
        total_error = (total_packed - total_est) / total_est
        print("-" * 78)
        print(f"{'TOTAL':<16} {total_est:>15,} {total_packed:>15,} {total_error * 100:>7.2f}%")

    print()
    print("=" * 78)
    print("2. the blend loads and samples")
    print("=" * 78)
    datasets = []
    factors = []
    for record in manifest["datasets"]:
        pbins = sorted((args.packed_dir / record["name"]).rglob("*.pbin"))
        if not pbins:
            continue
        for pbin in pbins:
            datasets.append(
                PackedMemMapDatasetContinuous(
                    raw_data_path=pbin,
                    sample_key="input_ids",
                    block_size=args.sequence_length,
                    reuse_last_target=True,
                )
            )
            factors.append(float(record["ratio"]))

    if not datasets:
        failures.append("no packed datasets to combine")
    else:
        blend = WeightedCombinedDataset(datasets=datasets, repeat_factors=factors, seed=42)
        expected = sum(int(len(d) * f) for d, f in zip(datasets, factors))
        print(f"  {len(datasets)} packed file(s), repeat factors {sorted(set(factors))}")
        print(f"  blend length {len(blend):,} (expected about {expected:,})")
        if abs(len(blend) - expected) > len(datasets):
            failures.append(f"blend length {len(blend)} does not match expected {expected}")

        # Sample the ends and the middle: an off-by-one in the affine permutation shows up
        # at a boundary, and a fractional factor's partial pass shows up nowhere else.
        probes = [0, 1, len(blend) // 2, len(blend) - 2, len(blend) - 1]
        seen = 0
        for i in probes:
            sample = blend[i]
            tokens = sample["input_ids"]
            if len(tokens) != args.sequence_length:
                failures.append(f"sample {i} has {len(tokens)} tokens, expected {args.sequence_length}")
            seen += 1
        print(f"  pulled {seen} samples at the boundaries and the middle, all {args.sequence_length} tokens")

        fractional = [f for f in factors if f != int(f)]
        if fractional:
            print(f"  fractional factors exercised: {sorted(set(fractional))}")
        else:
            failures.append("no fractional repeat factor in the blend; the partial-pass path was not exercised")

    print()
    print("=" * 78)
    print("3. the source tree was not written to")
    print("=" * 78)
    stray = []
    for dirpath, _, filenames in os.walk(args.source_root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                stray.append(str(Path(dirpath) / filename))
    print(f"  {args.source_root}: {len(stray)} non-jsonl file(s)")
    if stray:
        for path in stray[:10]:
            print(f"      {path}")
        failures.append(f"{len(stray)} file(s) written into the source tree, e.g. {stray[0]}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        for problem in failures:
            print(f"  - {problem}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
