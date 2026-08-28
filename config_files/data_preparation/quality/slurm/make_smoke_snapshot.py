#!/usr/bin/env python3
"""Freezes a tiny slice of the annealing corpora so the pipeline can be tested end to end.

Why a snapshot rather than pointing at the corpora directly: the source tree is still
being transferred, and a corpus that is re-sharded halfway through a run invalidates every
byte offset recorded against it. A frozen copy makes the test repeatable and immune to
that. It also keeps the test honest about scale -- 1.5 GB runs in minutes, so a broken
stage is found in minutes.

The five datasets are chosen to cover all four distinct join-key kinds plus the
native-metrics-only path, which is full code coverage of the join. HPLT is deliberately
absent: it uses the same ``field`` key kind as FineWiki and would add 327 GB of annotation
bucket reads to learn nothing new.

Reads only. Nothing is written under the source root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SOURCE_ROOT = Path("/data/annealing")

# (name, relative source dir, glob, byte budget, pointer filter)
# `pointer_part` restricts KletterMix to documents whose source pointer lands in a single
# part of the 1.6 TB original corpus, so resolving them reads one 16 GB file instead of
# scattering across many.
DATASETS = [
    ("finewiki-de", "german/Finewiki", "*.jsonl", 256 << 20, None),
    ("finepdfs-es", "spanish/Finepdfs", "*.jsonl", 256 << 20, None),
    ("climbmix-en", "english/Climbmix", "*.jsonl", 256 << 20, None),
    ("klettermix-de", "german/AIML-TUDA-KletterMix-filtered", "*.jsonl", 3 << 30, "part_65.detokenized.jsonl"),
    ("dolmino", "english/Dolmino", "**/*.jsonl", 64 << 20, None),
]


def copy_prefix(src: Path, dst: Path, budget: int) -> tuple[int, int]:
    """Copies a line-aligned prefix of a JSONL file.

    Args:
        src (Path): Source JSONL file, opened read-only.
        dst (Path): Destination path. Parents are created.
        budget (int): Approximate byte budget. The last, partial line is dropped, since a
            truncated JSON document would fail to parse and look like a pipeline bug.

    Returns:
        tuple[int, int]: Documents and bytes written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_bytes = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for line in fin:
            if not line.endswith(b"\n"):
                break
            fout.write(line)
            n_bytes += len(line)
            n_docs += 1
            if n_bytes >= budget:
                break
    return n_docs, n_bytes


def copy_filtered(src: Path, dst: Path, budget: int, pointer_part: str) -> tuple[int, int]:
    """Copies documents whose ``id`` pointer names one particular source part.

    Args:
        src (Path): Source JSONL file, opened read-only.
        dst (Path): Destination path. Parents are created.
        budget (int): How many source bytes to scan before stopping.
        pointer_part (str): Keep only documents whose ``id`` starts with this.

    Returns:
        tuple[int, int]: Documents and bytes written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_bytes = 0
    scanned = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for line in fin:
            scanned += len(line)
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if str(record.get("id", "")).startswith(pointer_part):
                fout.write(line)
                n_bytes += len(line)
                n_docs += 1
            if scanned >= budget:
                break
    return n_docs, n_bytes


def main() -> int:
    """Builds the snapshot.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Snapshot root. Must not be under the source root.")
    parser.add_argument("--force", action="store_true", help="Replace an existing snapshot.")
    args = parser.parse_args()

    out = args.out.resolve()
    if SOURCE_ROOT in out.parents or out == SOURCE_ROOT:
        print(f"refusing to write inside the source tree {SOURCE_ROOT}", file=sys.stderr)
        return 2
    if out.exists():
        if not args.force:
            print(f"{out} exists; pass --force to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(out)

    total_docs = 0
    total_bytes = 0
    for name, rel, pattern, budget, pointer_part in DATASETS:
        src_dir = SOURCE_ROOT / rel
        if not src_dir.exists():
            print(f"  {name:<14} SKIPPED, {src_dir} does not exist")
            continue
        sources = sorted(src_dir.glob(pattern))
        if not sources:
            print(f"  {name:<14} SKIPPED, no files match {pattern}")
            continue
        # Dolmino's files are small and its metrics live under a subdirectory, so take a
        # handful of whole files rather than a prefix of one.
        picks = sources[:4] if name == "dolmino" else sources[:1]
        docs = written = 0
        for src in picks:
            dst = out / rel / src.relative_to(src_dir)
            if pointer_part:
                d, b = copy_filtered(src, dst, budget, pointer_part)
            else:
                d, b = copy_prefix(src, dst, budget)
            docs += d
            written += b
        print(f"  {name:<14} {docs:>9,} docs  {written / (1 << 20):>8.1f} MiB  from {len(picks)} file(s)")
        total_docs += docs
        total_bytes += written

    print(f"\n  snapshot at {out}: {total_docs:,} docs, {total_bytes / (1 << 30):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
