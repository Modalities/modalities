"""Writes a selection out as index files the existing packer already understands.

A modalities index is a pickled ``list[(byte_offset, byte_len)]`` naming the documents
of a JSONL file, and ``PackedDataGenerator`` tokenizes exactly the documents its index
lists. So a selection does not need a filtered copy of the corpus and does not need any
change to the packer: writing an index that lists only the surviving documents is
enough, and packing then reads only those.

The practical consequence is that an ablation costs megabytes rather than terabytes.
The source tree is never written to, and several selections can coexist as several
index directories over the same untouched data.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from modalities.dataloader.preprocessing.quality.file_manifest import FileManifest, ManifestError
from modalities.dataloader.preprocessing.quality.registry import CorpusRegistry, DatasetEntry
from modalities.dataloader.preprocessing.quality.selection import (
    DatasetSelection,
    MissingPolicy,
    SelectionConfig,
    document_mask,
)
from modalities.utils.logger_utils import get_logger


class MaterializationError(RuntimeError):
    """Raised when a selection cannot be written out."""


@dataclass
class MaterializedDataset:
    """Where one dataset's filtered indexes ended up.

    Attributes:
        name (str): Dataset name.
        ratio (float): Up/downsample factor recorded for training.
        n_documents_total (int): Documents before filtering.
        n_documents_kept (int): Documents listed in the written indexes.
        tokens_kept (int): Estimated tokens of the kept documents.
        index_files (dict[str, str]): Source JSONL path to written index path.
    """

    name: str
    ratio: float
    n_documents_total: int
    n_documents_kept: int
    tokens_kept: int
    index_files: dict[str, str]

    def to_dict(self) -> dict:
        """Renders the record for the manifest.

        Returns:
            dict: Plain-data form of this dataset's outcome.
        """
        return {
            "name": self.name,
            "ratio": self.ratio,
            "n_documents_total": self.n_documents_total,
            "n_documents_kept": self.n_documents_kept,
            "row_retention": round(self.n_documents_kept / self.n_documents_total, 6)
            if self.n_documents_total
            else 0.0,
            "est_tokens_kept": self.tokens_kept,
            "index_files": self.index_files,
        }


def config_fingerprint(config: SelectionConfig) -> str:
    """Fingerprints a selection so a manifest can be traced back to it.

    Args:
        config (SelectionConfig): The selection.

    Returns:
        str: Short stable digest of the selection's content.
    """
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def materialize_dataset(
    sidecar_dir: Path,
    dataset_entry: DatasetEntry,
    dataset_selection: DatasetSelection,
    missing_policy: MissingPolicy,
    output_dir: Path,
    show_progress: bool = True,
) -> MaterializedDataset:
    """Writes filtered index files for one dataset.

    Args:
        sidecar_dir (Path): Directory of that dataset's sidecar parts.
        dataset_entry (DatasetEntry): Registry entry, used to map file ids back to
            source paths.
        dataset_selection (DatasetSelection): The rule to apply.
        missing_policy (MissingPolicy): Policy for unannotated documents.
        output_dir (Path): Directory receiving the index files. The source tree's
            directory structure is mirrored below it.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        MaterializedDataset: Counts and the written index paths.

    Raises:
        MaterializationError: If the sidecar is missing, or if the source tree changed
            since the sidecar was built -- the byte offsets then describe documents that
            are no longer at those positions, so the blend would be silently wrong.
    """
    parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    if not parts:
        raise MaterializationError(f"no sidecar parts found in {sidecar_dir}")

    # File ids are positions in a file list, so they only mean anything against the list
    # the sidecar was built from. Resolve through the recorded manifest and refuse if the
    # tree has moved underneath us.
    try:
        source_files = FileManifest.read(sidecar_dir).require_current(dataset_entry)
    except ManifestError as e:
        raise MaterializationError(str(e)) from e
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accumulated per source file, because one file's sidecar may span several parts
    # and an index must list its documents in file order.
    per_file: dict[int, list[tuple[int, int]]] = {}
    n_total = 0
    n_kept = 0
    tokens_kept = 0

    for part in tqdm(parts, desc=f"select {dataset_selection.name}", disable=not show_progress):
        parquet_file = pq.ParquetFile(part)
        for group_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_idx)
            n_total += table.num_rows
            mask = document_mask(table, dataset_selection, missing_policy)
            if not mask.any():
                continue
            file_ids = table.column("file_id").to_numpy(zero_copy_only=False)[mask]
            offsets = table.column("byte_offset").to_numpy(zero_copy_only=False)[mask]
            lengths = table.column("byte_len").to_numpy(zero_copy_only=False)[mask]
            tokens = table.column("est_tokens").to_numpy(zero_copy_only=False)[mask]
            n_kept += int(mask.sum())
            tokens_kept += int(tokens.sum())
            for file_id, offset, length in zip(file_ids, offsets, lengths):
                per_file.setdefault(int(file_id), []).append((int(offset), int(length)))

    index_files: dict[str, str] = {}
    for file_id, entries in sorted(per_file.items()):
        if file_id >= len(source_files):
            raise MaterializationError(
                f"dataset {dataset_selection.name!r}: sidecar references file id {file_id} but its manifest "
                f"records only {len(source_files)} files. The sidecar is internally inconsistent; rebuild it."
            )
        source_path = source_files[file_id]
        # Index entries must be ordered by position, as a freshly generated index is.
        entries.sort()
        relative = source_path.relative_to(dataset_entry.jsonl_root).with_suffix(".idx")
        index_path = output_dir / relative
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(pickle.dumps(entries))
        index_files[str(source_path)] = str(index_path)

    return MaterializedDataset(
        name=dataset_selection.name,
        ratio=dataset_selection.ratio,
        n_documents_total=n_total,
        n_documents_kept=n_kept,
        tokens_kept=tokens_kept,
        index_files=index_files,
    )


def materialize_blend(
    config: SelectionConfig,
    registry: CorpusRegistry,
    sidecar_root: Path,
    output_root: Path,
    show_progress: bool = True,
) -> Path:
    """Writes filtered indexes and a manifest for a whole selection.

    Args:
        config (SelectionConfig): The blend specification.
        registry (CorpusRegistry): Registry resolving dataset names to source files.
        sidecar_root (Path): Directory holding one subdirectory of sidecar parts per
            dataset.
        output_root (Path): Directory receiving per-dataset index trees and the
            manifest.
        show_progress (bool): Whether to show progress bars.

    Returns:
        Path: Path to the written ``mix_manifest.yaml``.

    Raises:
        MaterializationError: If a selected dataset has no sidecar.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    materialized: list[MaterializedDataset] = []

    for dataset_selection in config.enabled_datasets():
        entry = registry.get(dataset_selection.name)
        sidecar_dir = Path(sidecar_root) / dataset_selection.name
        if not sidecar_dir.is_dir():
            raise MaterializationError(
                f"dataset {dataset_selection.name!r} has no sidecar at {sidecar_dir}; "
                "run 'modalities data quality build-sidecar' for it first"
            )
        materialized.append(
            materialize_dataset(
                sidecar_dir=sidecar_dir,
                dataset_entry=entry,
                dataset_selection=dataset_selection,
                missing_policy=config.policy_for(dataset_selection),
                output_dir=output_root / dataset_selection.name,
                show_progress=show_progress,
            )
        )

    total_effective = sum(d.tokens_kept * d.ratio for d in materialized)
    manifest = {
        "selection_fingerprint": config_fingerprint(config),
        "missing_annotation": config.missing_annotation.value,
        "target_tokens": config.target_tokens,
        "est_total_effective_tokens": int(total_effective),
        "datasets": [
            {
                **d.to_dict(),
                "est_effective_tokens": int(d.tokens_kept * d.ratio),
                "blend_share": round(d.tokens_kept * d.ratio / total_effective, 6) if total_effective else 0.0,
                "predicates": [p.describe() for p in next(s for s in config.datasets if s.name == d.name).predicates],
            }
            for d in materialized
        ],
    }
    manifest_path = output_root / "mix_manifest.yaml"
    with manifest_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    get_logger(name="main").info(
        f"Wrote {len(materialized)} filtered index tree(s) and {manifest_path}; "
        f"estimated {_humanise_tokens(total_effective)} effective tokens."
    )
    return manifest_path


def _humanise_tokens(n: float) -> str:
    # Blend totals span from a few thousand tokens in a test to hundreds of billions in
    # a real run, so a fixed unit renders one of those two cases uselessly.
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return f"{n:.0f}"
