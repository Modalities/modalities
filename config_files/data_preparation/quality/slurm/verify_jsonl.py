#!/usr/bin/env python3
"""Checks the exported JSONL against the manifest, and against the corpus it came from.

Three questions, in increasing order of how much they would cost to get wrong:

1. Does every shard hold the number of lines its record claims? A short shard means a job
   died mid-write and the resume logic accepted it, which would silently shrink the training
   set by however much it lost.
2. Do the line counts realise the ratios that were asked for? Up- and downsampling is now in
   the bytes, so a dataset at 3.0 that emitted 1x is not a reporting error, it is the wrong
   training set.
3. Are the exported lines the documents they claim to be? Sampled shards are read back and
   compared against the source file at the recorded offset. Nothing is re-serialised by the
   export, so this is a byte comparison, and any mismatch means the corpus moved under us.

Reads only; writes nothing anywhere.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import yaml

from modalities.dataloader.preprocessing.quality.export import copies_for


def load(path: Path) -> dict:
    """Reads a YAML file.

    Args:
        path (Path): The file.

    Returns:
        dict: Its contents.
    """
    with Path(path).open() as f:
        return yaml.safe_load(f)


def count_lines(path: Path) -> int:
    """Counts newlines in a file without holding it in memory.

    Args:
        path (Path): The file.

    Returns:
        int: Number of lines.
    """
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            total += chunk.count(b"\n")
    return total


def main() -> int:
    """Verifies an export.

    Returns:
        int: 0 if everything checks out, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="export_manifest.yaml")
    parser.add_argument("--mix_manifest", type=Path, default=None, help="mix_manifest.yaml, to check the ratios.")
    parser.add_argument("--shards", type=int, default=5, help="Shards to replay against the corpus in full.")
    parser.add_argument("--count_lines", action="store_true", help="Re-count every shard. Reads the whole export.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing which shards to sample.")
    parser.add_argument(
        "--source_root",
        type=Path,
        default=None,
        help="If given, assert the corpus holds nothing but .jsonl files -- i.e. that no stage "
        "wrote indexes or outputs into the source tree.",
    )
    args = parser.parse_args()

    manifest = load(args.manifest)
    mix = load(args.mix_manifest) if args.mix_manifest else None
    root = args.manifest.parent
    problems: list[str] = []

    print(f"export at {root}")
    print(f"  {manifest['n_lines']:,} lines, {manifest['n_documents']:,} documents, "
          f"{manifest['n_bytes'] / 1e12:,.2f} TB")
    if not manifest.get("repeat_factor_applied"):
        problems.append("the manifest does not record that the repeat factors were applied")
    print()

    print(f"{'dataset':<18} {'shards':>8} {'documents':>14} {'lines':>15} {'lines/doc':>10} {'factors':>22}")
    print("-" * 92)
    for dataset in manifest["datasets"]:
        factors = dataset["factors_applied"]
        ratio = dataset["n_lines"] / dataset["n_documents"] if dataset["n_documents"] else 0.0
        described = ",".join(f"{v:g}" for v in sorted(factors.values()))
        print(
            f"{dataset['name']:<18} {dataset['n_shards']:>8,} {dataset['n_documents']:>14,} "
            f"{dataset['n_lines']:>15,} {ratio:>10.2f} {described:>22}"
        )
        if dataset["ratio"] != 1.0:
            problems.append(f"{dataset['name']} reports a training ratio of {dataset['ratio']}, not 1.0")
        # With a single flat factor the realised lines-per-document must land on it. A curve
        # has several factors, so only the range is checkable.
        if len(factors) == 1:
            expected = next(iter(factors.values()))
            if dataset["n_documents"] and abs(ratio - expected) > max(0.05, 0.05 * expected):
                problems.append(
                    f"{dataset['name']} realised {ratio:.3f} lines per document against {expected:g} requested"
                )
        elif factors and dataset["n_documents"] and not (min(factors.values()) <= ratio <= max(factors.values())):
            problems.append(
                f"{dataset['name']} realised {ratio:.3f} lines per document, outside its curve's "
                f"{min(factors.values()):g}-{max(factors.values()):g}"
            )
    print()

    if args.count_lines:
        print("re-counting every shard")
        print("-" * 92)
        for dataset in manifest["datasets"]:
            counted = sum(count_lines(p) for p in sorted((root / dataset["name"]).rglob("*.jsonl")))
            status = "ok" if counted == dataset["n_lines"] else "MISMATCH"
            print(f"  {dataset['name']:<18} {counted:>15,} {status}")
            if counted != dataset["n_lines"]:
                problems.append(
                    f"{dataset['name']} holds {counted:,} lines, manifest says {dataset['n_lines']:,}"
                )
        print()

    print(f"replaying {args.shards} shard(s) against the corpus, line by line")
    print("-" * 92)
    if mix is None:
        print("  skipped: pass --mix_manifest to locate the source files and their indexes")
    else:
        rng = random.Random(args.seed)
        contributions: dict[str, dict[str, list]] = {}
        for row in mix["datasets"]:
            name = row.get("source_dataset") or row["name"]
            for source_path, index_path in row["index_files"].items():
                contributions.setdefault(name, {}).setdefault(source_path, []).append(
                    (index_path, float(row["ratio"]))
                )
        seed = mix.get("seed", 42)

        candidates = [
            (name, source) for name, sources in contributions.items() for source in sources
        ]
        checked_lines = 0
        for name, source_path in rng.sample(candidates, min(args.shards, len(candidates))):
            entries = []
            for index_path, factor in contributions[name][source_path]:
                for offset, length in pickle.loads(Path(index_path).read_bytes()):
                    entries.append((offset, length, factor))
            entries.sort()

            shard = root / name / Path(source_path).name.replace(".jsonl", ".jsonl")
            matches = sorted((root / name).rglob(Path(source_path).name))
            shard = matches[0] if matches else shard
            if not shard.is_file():
                problems.append(f"{name}: no shard for {source_path}")
                continue

            # Replays exactly what the export should have written -- same offsets, same copy
            # counts -- and compares it to what is on disk. This validates content, ordering
            # and repetition together, which counting lines cannot.
            mismatch = None
            with shard.open("rb") as out, Path(source_path).open("rb") as src:
                for offset, length, factor in entries:
                    copies = copies_for(factor, seed, str(source_path), offset)
                    if copies == 0:
                        continue
                    src.seek(offset)
                    expected = src.read(length) + b"\n"
                    for _ in range(copies):
                        actual = out.readline()
                        checked_lines += 1
                        if actual != expected:
                            mismatch = (offset, expected[:80], actual[:80])
                            break
                    if mismatch:
                        break
                trailing = out.readline() if not mismatch else b""
            if mismatch:
                offset, expected, actual = mismatch
                problems.append(f"{name} {shard.name}: line at offset {offset} differs")
                print(f"  MISMATCH {name}/{shard.name} at offset {offset}")
                print(f"    expected {expected!r}")
                print(f"    actual   {actual!r}")
            elif trailing:
                problems.append(f"{name} {shard.name}: has more lines than the index accounts for")
            else:
                print(f"  ok {name}/{shard.name}")
        print(f"  {checked_lines:,} lines replayed byte-for-byte")
    print()

    if args.source_root is not None:
        print(f"checking that nothing was written into {args.source_root}")
        print("-" * 92)
        stray = [
            str(p) for p in args.source_root.rglob("*") if p.is_file() and p.suffix != ".jsonl"
        ]
        if stray:
            problems.append(f"{len(stray):,} non-.jsonl file(s) under the source root")
            for path in stray[:10]:
                print(f"  STRAY {path}")
        else:
            print("  clean: the corpus holds only .jsonl files")
        print()

    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("OK: the export matches its manifest and the ratios it was asked for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
