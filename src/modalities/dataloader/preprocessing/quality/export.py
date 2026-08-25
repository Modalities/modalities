"""Writes a selection out as JSONL, with the up/downsampling baked into the bytes.

The packing stage this replaces left the ratios as metadata: it tokenized the selected
documents once, and ``WeightedCombinedDataset`` applied the repeat factors at training time,
fractional ones included. The output here is meant to be *concatenated* into a training set,
and concatenation carries no weights, so the factors have to become bytes. A dataset at 3.0
has each of its documents written three times; one at 0.6 loses two of every five.

Fractional factors are resolved per document rather than by truncating a list, so a factor of
1.2 means every document once and a hash-chosen fifth of them twice. The choice is a function
of the document's position and the blend's seed, so it is identical on every run and on every
machine, and it does not depend on the order files happen to be processed in.

Output lines are copied verbatim. Nothing is parsed or re-serialised, so each line is
byte-identical to the source line it came from -- which also makes the export a byte-for-byte
auditable operation rather than a transformation.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry
from modalities.utils.logger_utils import get_logger

# The name of the manifest this stage writes, alongside the shards it describes.
EXPORT_MANIFEST = "export_manifest.yaml"

# Resolution of the fractional part of a repeat factor. A factor of 1.2 is realised as "one
# copy always, plus a second for the 200,000 millionths of documents whose hash falls below
# the threshold", so this bounds how precisely a factor can be expressed.
FRACTION_MODULUS = 1_000_000


class ExportError(RuntimeError):
    """Raised when a selection cannot be exported as JSONL."""


def fraction_hash(seed: int, source_path: str, byte_offset: int) -> int:
    """Places one document on the axis that decides its fractional copy.

    Args:
        seed (int): The blend's seed.
        source_path (str): The source file the document lives in.
        byte_offset (int): Its offset within that file.

    Returns:
        int: A value in ``[0, FRACTION_MODULUS)``.

    Note:
        blake2b rather than the built-in ``hash``, whose string seed varies per process --
        the same reasoning as :func:`~...annotation_join.bucket_of`. Position is used as the
        identity because this stage reads indexes rather than sidecars and has no join key
        in hand; it is stable because :mod:`file_manifest` pins every source file's size and
        mtime and ``verify-sidecar`` refuses one that has drifted.
    """
    digest = hashlib.blake2b(f"{seed}\x00{source_path}\x00{byte_offset}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % FRACTION_MODULUS


def copies_for(factor: float, seed: int, source_path: str, byte_offset: int) -> int:
    """How many times one document is written.

    Args:
        factor (float): The dataset's repeat factor.
        seed (int): The blend's seed.
        source_path (str): The source file the document lives in.
        byte_offset (int): Its offset within that file.

    Returns:
        int: Whole copies, at least zero.
    """
    whole = math.floor(factor)
    remainder = factor - whole
    if remainder > 0 and fraction_hash(seed, source_path, byte_offset) < round(remainder * FRACTION_MODULUS):
        return whole + 1
    return whole


@dataclass
class ShardReport:
    """What one output shard holds.

    Attributes:
        output_path (Path): The written shard.
        source_path (Path): The JSONL file it was drawn from.
        n_documents (int): Distinct documents drawn.
        n_lines (int): Lines written, counting repeats.
        n_bytes (int): Bytes written.
        skipped (bool): Whether a complete shard was left alone.
    """

    output_path: Path
    source_path: Path
    n_documents: int = 0
    n_lines: int = 0
    n_bytes: int = 0
    skipped: bool = False


@dataclass
class DatasetExport:
    """What one dataset contributed.

    Attributes:
        name (str): The dataset.
        factors (dict[str, float]): Repeat factor per manifest row that fed it. A curve
            contributes several rows, one per quality bucket, each with its own factor.
        shards (list[ShardReport]): The shards written.
    """

    name: str
    factors: dict[str, float] = field(default_factory=dict)
    shards: list[ShardReport] = field(default_factory=list)

    @property
    def n_lines(self) -> int:
        """Lines across every shard.

        Returns:
            int: Total lines written for this dataset.
        """
        return sum(s.n_lines for s in self.shards)

    @property
    def n_documents(self) -> int:
        """Distinct documents across every shard.

        Returns:
            int: Total documents drawn for this dataset.
        """
        return sum(s.n_documents for s in self.shards)

    @property
    def n_bytes(self) -> int:
        """Bytes across every shard.

        Returns:
            int: Total bytes written for this dataset.
        """
        return sum(s.n_bytes for s in self.shards)


def _meta_path(output_path: Path) -> Path:
    """Where a shard's completion record lives.

    Args:
        output_path (Path): The shard.

    Returns:
        Path: ``<shard>.meta.json``.
    """
    return output_path.with_name(output_path.name + ".meta.json")


def _completed(output_path: Path) -> Optional[dict]:
    """The record of a finished shard, if it finished.

    A shard is complete only when its record exists *and* the file on disk is the size the
    record claims. Existence alone is not health: a killed job leaves a plausible-looking
    partial file, and taking that as done is exactly how a truncated ``.pbin`` once survived
    into a blend.

    Args:
        output_path (Path): The shard.

    Returns:
        Optional[dict]: The record, or None if the shard is missing or unfinished.
    """
    try:
        record = json.loads(_meta_path(output_path).read_text())
        return record if output_path.stat().st_size == record["n_bytes"] else None
    except (OSError, ValueError, KeyError):
        return None


def export_file(
    source_path: Path,
    contributions: list[tuple[Path, float]],
    output_path: Path,
    seed: int,
    resume: bool = True,
) -> ShardReport:
    """Writes one source file's selected documents to one JSONL shard.

    Args:
        source_path (Path): The source JSONL file.
        contributions (list[tuple[Path, float]]): Index file and repeat factor, one pair per
            manifest row drawing from this source file. A quality curve produces several.
        output_path (Path): The shard to write.
        seed (int): The blend's seed.
        resume (bool): Leave a complete shard alone.

    Returns:
        ShardReport: What was written, or what was already there.

    Raises:
        ExportError: If an index names a document the source file cannot supply.
    """
    if resume:
        record = _completed(output_path)
        if record is not None:
            return ShardReport(
                output_path=output_path,
                source_path=source_path,
                n_documents=record["n_documents"],
                n_lines=record["n_lines"],
                n_bytes=record["n_bytes"],
                skipped=True,
            )

    # Every contributing index is merged and re-sorted by offset, so the shard is written in
    # source order in a single forward pass. Writing bucket by bucket instead would group the
    # output by quality level, which is a strong ordering artifact to hand to a trainer.
    entries: list[tuple[int, int, float]] = []
    for index_path, factor in contributions:
        for offset, length in pickle.loads(Path(index_path).read_bytes()):
            entries.append((offset, length, factor))
    entries.sort()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = output_path.with_name(output_path.name + ".partial")
    n_documents = n_lines = n_bytes = 0

    # Mapped directly rather than through LargeFileLinesReader: that reader addresses a file
    # by line number and insists on an index describing the whole file, whereas this stage
    # already holds the selection's byte offsets and needs nothing but the bytes at them.
    with source_path.open("rb") as raw:
        data = mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            with scratch.open("wb") as out:
                for offset, length, factor in entries:
                    copies = copies_for(factor, seed, str(source_path), offset)
                    n_documents += 1
                    if copies == 0:
                        continue
                    line = data[offset : offset + length]
                    if len(line) != length:
                        raise ExportError(
                            f"{source_path}: index names {length} bytes at offset {offset} but the "
                            f"file supplied {len(line)}. The source has changed since the sidecar "
                            f"was built; re-run 'modalities quality verify-sidecar'."
                        )
                    # The index records the line without its terminator -- checked against the
                    # real corpus, where entries end on '}' with a one-byte gap to the next
                    # offset -- so the newline is added here.
                    record = line + b"\n"
                    for _ in range(copies):
                        out.write(record)
                    n_lines += copies
                    n_bytes += copies * len(record)
        except BaseException:
            scratch.unlink(missing_ok=True)
            raise
        finally:
            data.close()

    os.replace(scratch, output_path)
    # Written only after the shard is in place, so an interrupted export leaves no record and
    # is redone rather than trusted.
    _meta_path(output_path).write_text(
        json.dumps({"n_documents": n_documents, "n_lines": n_lines, "n_bytes": n_bytes})
    )
    return ShardReport(
        output_path=output_path,
        source_path=source_path,
        n_documents=n_documents,
        n_lines=n_lines,
        n_bytes=n_bytes,
    )


# The per-dataset record each array task writes. The blend-wide manifest is merged from these
# afterwards rather than written by every task: eighteen tasks rewriting one shared file would
# race, and the last writer would erase the rest.
DATASET_RECORD = "_export.yaml"


def _contributions_by_dataset(manifest: dict, registry: CorpusRegistry) -> dict[str, dict[Path, list]]:
    """Groups the manifest's rows into work, keyed by dataset and source file.

    A quality curve emits one row per bucket, named ``<dataset>__<level>`` and each carrying
    its own factor. They describe one input dataset and must land in one output directory,
    so they are merged here, and a source file drawn from by several buckets yields one shard
    fed by all of them.

    Args:
        manifest (dict): The parsed mix manifest.
        registry (CorpusRegistry): Resolves a dataset to its source root.

    Returns:
        dict[str, dict[Path, list]]: Dataset name to source path to
            ``[(index_path, factor), ...]``.
    """
    grouped: dict[str, dict[Path, list]] = {}
    for row in manifest["datasets"]:
        name = row.get("source_dataset") or row["name"]
        registry.get(name)  # fails loudly here rather than midway through a terabyte
        for source_path, index_path in row["index_files"].items():
            grouped.setdefault(name, {}).setdefault(Path(source_path), []).append(
                (Path(index_path), float(row["ratio"]))
            )
    return grouped


def export_blend(
    manifest_path: Path,
    registry_path: Path,
    output_root: Path,
    seed: Optional[int] = None,
    only: Optional[list[str]] = None,
    resume: bool = True,
    show_progress: bool = True,
) -> list[DatasetExport]:
    """Writes every dataset of a materialised selection out as JSONL.

    Args:
        manifest_path (Path): The ``mix_manifest.yaml`` written by the apply stage.
        registry_path (Path): The corpus registry YAML.
        output_root (Path): Directory receiving one subdirectory per dataset.
        seed (Optional[int]): Decides which documents get a fractional extra copy. Defaults
            to the seed the selection was applied with, which the mix manifest records.
        only (Optional[list[str]]): Restrict to these dataset names.
        resume (bool): Leave complete shards alone.
        show_progress (bool): Whether to show progress bars.

    Returns:
        list[DatasetExport]: What each dataset contributed.

    Raises:
        ExportError: If the manifest names an index that is not on disk.
    """
    with Path(manifest_path).open() as f:
        manifest = yaml.safe_load(f)
    registry = CorpusRegistry.from_yaml(registry_path)
    output_root = Path(output_root)
    seed = manifest.get("seed", 42) if seed is None else seed

    grouped = _contributions_by_dataset(manifest, registry)
    factors = {}
    for row in manifest["datasets"]:
        factors.setdefault(row.get("source_dataset") or row["name"], {})[row["name"]] = float(row["ratio"])

    exports: list[DatasetExport] = []
    for name in sorted(grouped):
        if only and name not in only:
            continue
        entry = registry.get(name)
        export = DatasetExport(name=name, factors=factors[name])
        sources = sorted(grouped[name])
        for source_path in tqdm(sources, desc=f"export {name}", disable=not show_progress):
            relative = source_path.relative_to(entry.jsonl_root).with_suffix(".jsonl")
            export.shards.append(
                export_file(
                    source_path=source_path,
                    contributions=grouped[name][source_path],
                    output_path=output_root / name / relative,
                    seed=seed,
                    resume=resume,
                )
            )
        _write_dataset_record(output_root, export, seed)
        exports.append(export)
        get_logger(name="main").info(
            f"{name}: {export.n_lines:,} lines from {export.n_documents:,} documents "
            f"({export.n_bytes / 1e9:,.1f} GB) over {len(export.shards):,} shards"
        )
    return exports


def _write_dataset_record(output_root: Path, export: DatasetExport, seed: int) -> Path:
    """Records what one dataset's export produced.

    Args:
        output_root (Path): The export root.
        export (DatasetExport): The dataset's result.
        seed (int): The seed used.

    Returns:
        Path: The written record.
    """
    directory = output_root / export.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_RECORD
    scratch = path.with_suffix(f".yaml.{os.getpid()}.tmp")
    scratch.write_text(
        yaml.safe_dump(
            {
                "name": export.name,
                "seed": seed,
                "n_documents": export.n_documents,
                "n_lines": export.n_lines,
                "n_bytes": export.n_bytes,
                "n_shards": len(export.shards),
                "factors_applied": export.factors,
                # Baked into the bytes, so a training config must not weight this again.
                "repeat_factor_applied": True,
                "ratio": 1.0,
            },
            sort_keys=False,
        )
    )
    os.replace(scratch, path)
    return path


def finalize_export(output_root: Path) -> Path:
    """Merges the per-dataset records into one manifest for the whole export.

    Args:
        output_root (Path): The export root.

    Returns:
        Path: The written ``export_manifest.yaml``.

    Raises:
        ExportError: If no dataset records are present.
    """
    output_root = Path(output_root)
    records = sorted(output_root.glob(f"*/{DATASET_RECORD}"))
    if not records:
        raise ExportError(f"no dataset records under {output_root}; nothing has been exported yet")

    datasets = [yaml.safe_load(p.read_text()) for p in records]
    manifest = {
        # The ratios are already in the bytes. Anyone carrying the mix manifest's ratio into a
        # 'weighted_combined' training config after this stage would apply it a second time --
        # 3.0 becoming 9.0 -- so this manifest states the training-time factor explicitly.
        "repeat_factor_applied": True,
        "training_ratio": 1.0,
        "note": (
            "Up- and downsampling is materialised in these files. The training set is the "
            "concatenation of every dataset's shards; do not apply the mix manifest's ratios again."
        ),
        "n_lines": sum(d["n_lines"] for d in datasets),
        "n_documents": sum(d["n_documents"] for d in datasets),
        "n_bytes": sum(d["n_bytes"] for d in datasets),
        "datasets": datasets,
    }
    path = output_root / EXPORT_MANIFEST
    scratch = path.with_suffix(f".yaml.{os.getpid()}.tmp")
    scratch.write_text(yaml.safe_dump(manifest, sort_keys=False))
    os.replace(scratch, path)
    get_logger(name="main").info(
        f"Export manifest: {len(datasets)} dataset(s), {manifest['n_lines']:,} lines, "
        f"{manifest['n_bytes'] / 1e12:,.2f} TB at {path}"
    )
    return path
