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
import os
import pickle
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

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
    exposure_report,
    ordered_quality_levels,
)
from modalities.dataloader.preprocessing.quality.upsampling import (
    UNANNOTATED_BUCKET,
    QualityBucket,
    UpsamplingError,
    solve_curve,
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
        source_dataset (Optional[str]): The registry dataset this came from, when ``name``
            carries a quality bucket suffix and so no longer matches the registry.
        quality_bucket (Optional[str]): The quality level this row covers, when the dataset
            was split by a quality curve.
    """

    name: str
    ratio: float
    n_documents_total: int
    n_documents_kept: int
    tokens_kept: int
    index_files: dict[str, str]
    source_dataset: Optional[str] = None
    quality_bucket: Optional[str] = None

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
            "source_dataset": self.source_dataset or self.name,
            "quality_bucket": self.quality_bucket,
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


def materialize_dataset_buckets(
    sidecar_dir: Path,
    dataset_entry: DatasetEntry,
    dataset_selection: DatasetSelection,
    missing_policy: MissingPolicy,
    output_dir: Path,
    show_progress: bool = True,
) -> list[MaterializedDataset]:
    """Writes one index tree per quality bucket, each with its own repeat factor.

    A quality curve gives a different repeat factor to each quality level, and the packer
    emits one file per source file, so documents of different levels have to end up in
    different indexes for their factors to mean anything. Each bucket therefore becomes its
    own row of the manifest, its own packed output, and its own entry in the training
    blend's repeat factors.

    The curve is solved from the token counts found here rather than from the cube, because
    this stage reads every document anyway: the numbers are exact, so the curve hits its
    token target exactly rather than to within the cube's interpolation error.

    Args:
        sidecar_dir (Path): Directory of that dataset's sidecar parts.
        dataset_entry (DatasetEntry): Registry entry, for mapping file ids to paths.
        dataset_selection (DatasetSelection): The rule to apply. Must carry an
            ``upsampling`` spec.
        missing_policy (MissingPolicy): Policy for unannotated documents.
        output_dir (Path): Directory receiving one subdirectory per bucket.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        list[MaterializedDataset]: One entry per bucket that survived with a non-zero
            factor, worst quality first.

    Raises:
        MaterializationError: If the sidecar is missing, the source tree has drifted, the
            quality field is absent, or the curve cannot be solved.
    """
    spec = dataset_selection.upsampling
    if spec is None:
        raise MaterializationError(f"dataset {dataset_selection.name!r} has no upsampling spec")

    parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    if not parts:
        raise MaterializationError(f"no sidecar parts found in {sidecar_dir}")
    try:
        source_files = FileManifest.read(sidecar_dir).require_current(dataset_entry)
    except ManifestError as e:
        raise MaterializationError(str(e)) from e

    levels = list(ordered_quality_levels(spec.quality_field))
    # Unannotated documents cannot be ordered, so they form the bottom bucket.
    bucket_labels = [UNANNOTATED_BUCKET] + levels
    per_bucket: dict[str, dict[int, list[tuple[int, int]]]] = {label: {} for label in bucket_labels}
    tokens_of: dict[str, int] = dict.fromkeys(bucket_labels, 0)
    documents_of: dict[str, int] = dict.fromkeys(bucket_labels, 0)
    n_total = 0

    for part in tqdm(parts, desc=f"select {dataset_selection.name}", disable=not show_progress):
        parquet_file = pq.ParquetFile(part)
        if spec.quality_field not in parquet_file.schema_arrow.names:
            raise MaterializationError(
                f"dataset {dataset_selection.name!r}: sidecar has no column "
                f"{spec.quality_field!r} to order quality by; join the annotations first"
            )
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
            quality = table.column(spec.quality_field).to_pylist()
            kept_quality = [q for q, keep in zip(quality, mask) if keep]

            for file_id, offset, length, token, level in zip(
                file_ids, offsets, lengths, tokens, kept_quality
            ):
                label = UNANNOTATED_BUCKET if level is None or level not in tokens_of else level
                per_bucket[label].setdefault(int(file_id), []).append((int(offset), int(length)))
                tokens_of[label] += int(token)
                documents_of[label] += 1

    buckets = [
        QualityBucket(
            label=label,
            n_documents=documents_of[label],
            n_tokens=tokens_of[label],
            unannotated=label == UNANNOTATED_BUCKET,
        )
        for label in bucket_labels
        if tokens_of[label] > 0
    ]
    try:
        plan = solve_curve(buckets, spec)
    except UpsamplingError as e:
        raise MaterializationError(f"dataset {dataset_selection.name!r}: {e}") from e

    results: list[MaterializedDataset] = []
    for bucket_plan in plan.buckets:
        label = bucket_plan.bucket.label
        if bucket_plan.factor <= 0:
            continue
        slug = label.strip("<>").replace(" ", "_")
        bucket_dir = Path(output_dir) / slug
        index_files: dict[str, str] = {}
        for file_id, entries in sorted(per_bucket[label].items()):
            if file_id >= len(source_files):
                raise MaterializationError(
                    f"dataset {dataset_selection.name!r}: sidecar references file id {file_id} but its "
                    f"manifest records only {len(source_files)} files. Rebuild it."
                )
            source_path = source_files[file_id]
            entries.sort()
            relative = source_path.relative_to(dataset_entry.jsonl_root).with_suffix(".idx")
            index_path = bucket_dir / relative
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.write_bytes(pickle.dumps(entries))
            index_files[str(source_path)] = str(index_path)

        results.append(
            MaterializedDataset(
                name=f"{dataset_selection.name}__{slug}",
                ratio=bucket_plan.factor,
                n_documents_total=n_total,
                n_documents_kept=bucket_plan.bucket.n_documents,
                tokens_kept=bucket_plan.bucket.n_tokens,
                index_files=index_files,
                source_dataset=dataset_selection.name,
                quality_bucket=label,
            )
        )

    get_logger(name="main").info(
        f"{dataset_selection.name}: quality curve on {spec.quality_field} over "
        f"{len(plan.buckets)} bucket(s), {len(results)} kept, exponent {plan.exponent:.2f}, "
        f"drawing {plan.tokens_drawn:,.0f} of {plan.tokens_available:,} available tokens"
    )
    return results


def _cap_for(config: SelectionConfig, materialized: MaterializedDataset) -> Optional[float]:
    """Finds the repetition cap that applies to one materialised row.

    Args:
        config (SelectionConfig): The selection.
        materialized (MaterializedDataset): The row, which may be one bucket of a dataset.

    Returns:
        Optional[float]: The dataset's own curve cap when it has one, otherwise the
            blend-wide cap, or None if neither was declared.
    """
    source = materialized.source_dataset or materialized.name
    for dataset in config.datasets:
        if dataset.name == source and dataset.upsampling is not None:
            return dataset.upsampling.max_factor
    return config.max_total_exposure


def materialize_blend(
    config: SelectionConfig,
    registry: CorpusRegistry,
    sidecar_root: Path,
    output_root: Path,
    show_progress: bool = True,
    allow_overexposure: bool = False,
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
        MaterializationError: If a selected dataset has no sidecar, or if the run would
            repeat data past its declared cap and ``allow_overexposure`` is not set.
    """
    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    # Everything is built in a sibling directory and moved into place only once the
    # exposure check has passed and the manifest is written. Writing into the destination
    # directly meant a rejected apply left fresh index trees next to the previous run's
    # manifest -- a directory that still looks complete but whose manifest no longer
    # describes the indexes beside it. A sibling keeps the move a rename on one filesystem.
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest_path = _materialize_into(
            staging_root=staging,
            published_root=output_root,
            config=config,
            registry=registry,
            sidecar_root=sidecar_root,
            show_progress=show_progress,
            allow_overexposure=allow_overexposure,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Two renames rather than one: os.replace refuses a non-empty destination directory,
    # so the previous run steps aside first and is only deleted once the new tree is live.
    superseded = output_root.parent / f".{output_root.name}.superseded.{os.getpid()}"
    if output_root.exists():
        os.replace(output_root, superseded)
    try:
        os.replace(staging, output_root)
    except BaseException:
        if superseded.exists():
            os.replace(superseded, output_root)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(superseded, ignore_errors=True)

    final_path = output_root / manifest_path.name
    get_logger(name="main").info(f"Published the blend at {output_root}; manifest {final_path}.")
    return final_path


def _materialize_into(
    staging_root: Path,
    published_root: Path,
    config: SelectionConfig,
    registry: CorpusRegistry,
    sidecar_root: Path,
    show_progress: bool,
    allow_overexposure: bool,
) -> Path:
    """Builds a complete blend inside ``staging_root``.

    Split out from ``materialize_blend`` so that every failure path -- a missing sidecar,
    a bad curve, the exposure cap -- leaves the caller with a directory to discard rather
    than a half-updated destination.

    Args:
        staging_root (Path): Empty directory receiving the index trees and manifest.
        published_root (Path): Where the staged tree will end up. Index paths are recorded
            relative to this, not to the staging directory, which stops existing.
        config (SelectionConfig): The blend specification.
        registry (CorpusRegistry): Registry resolving dataset names to source files.
        sidecar_root (Path): Directory holding one subdirectory of sidecar parts per dataset.
        show_progress (bool): Whether to show progress bars.
        allow_overexposure (bool): Whether to proceed despite exceeded repetition caps.

    Returns:
        Path: Path to the manifest inside ``staging_root``.

    Raises:
        MaterializationError: If a selected dataset has no sidecar, or if the run would
            repeat data past its declared cap and ``allow_overexposure`` is not set.
    """
    output_root = staging_root
    materialized: list[MaterializedDataset] = []

    for dataset_selection in config.enabled_datasets():
        entry = registry.get(dataset_selection.name)
        sidecar_dir = Path(sidecar_root) / dataset_selection.name
        if not sidecar_dir.is_dir():
            raise MaterializationError(
                f"dataset {dataset_selection.name!r} has no sidecar at {sidecar_dir}; "
                "run 'modalities data quality build-sidecar' for it first"
            )
        arguments = dict(
            sidecar_dir=sidecar_dir,
            dataset_entry=entry,
            dataset_selection=dataset_selection,
            missing_policy=config.policy_for(dataset_selection),
            output_dir=output_root / dataset_selection.name,
            show_progress=show_progress,
        )
        # A curve splits one dataset into several rows, one per quality bucket, so that each
        # can carry its own repeat factor through packing and into the training blend.
        if dataset_selection.upsampling is not None:
            materialized.extend(materialize_dataset_buckets(**arguments))
        else:
            materialized.append(materialize_dataset(**arguments))

    # The index writers record the path they physically wrote to, which is inside the
    # staging directory. That directory is renamed away on publication, so a manifest
    # holding those paths would name files that no longer exist and every packing config
    # built from it would point at nothing.
    materialized = [_rebase_index_files(d, staging_root, published_root) for d in materialized]

    total_effective = sum(d.tokens_kept * d.ratio for d in materialized)

    # Ratios are per pass. If the run consumes more than one pass, every factor is
    # multiplied, so a bucket set to 7x becomes 14x on a second pass -- well past where
    # repetition pays. Checked here rather than at preview because this is the point of
    # commitment: preview reports it, apply refuses.
    exposure = exposure_report(
        entries=[
            (
                d.name,
                d.ratio,
                _cap_for(config, d),
            )
            for d in materialized
        ],
        effective_tokens=total_effective,
        training_tokens=config.target_tokens,
    )
    if exposure.exceeded and not allow_overexposure:
        offenders = "\n".join(
            f"  {row.label}: {row.factor:.2f}x requested, seen {row.exposure:.2f}x over "
            f"{exposure.passes:.2f} passes, cap {row.cap:g}x"
            for row in exposure.exceeded
        )
        raise MaterializationError(
            f"the run would repeat data past its declared cap:\n{offenders}\n"
            f"The blend yields {total_effective:,.0f} effective tokens and the run consumes "
            f"{config.target_tokens:,.0f}, so it wraps {exposure.passes:.2f} times. Raise the "
            f"blend's yield, lower target_tokens, relax the cap, or pass allow_overexposure "
            f"to proceed anyway."
        )
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
                "predicates": [
                    p.describe()
                    for p in next(s for s in config.datasets if s.name == (d.source_dataset or d.name)).predicates
                ],
            }
            for d in materialized
        ],
    }
    manifest_path = output_root / "mix_manifest.yaml"
    with manifest_path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    get_logger(name="main").info(
        f"Staged {len(materialized)} filtered index tree(s); "
        f"estimated {_humanise_tokens(total_effective)} effective tokens."
    )
    return manifest_path


def _rebase_index_files(
    materialized: MaterializedDataset, staging_root: Path, published_root: Path
) -> MaterializedDataset:
    """Rewrites a row's index paths from where they were written to where they will live.

    Args:
        materialized (MaterializedDataset): A row carrying staging-relative index paths.
        staging_root (Path): The directory the indexes were written under.
        published_root (Path): The directory they will be renamed into.

    Returns:
        MaterializedDataset: The same row with index paths under ``published_root``.
    """
    return replace(
        materialized,
        index_files={
            source: str(published_root / Path(index).relative_to(staging_root))
            for source, index in materialized.index_files.items()
        },
    )


def _humanise_tokens(n: float) -> str:
    # Blend totals span from a few thousand tokens in a test to hundreds of billions in
    # a real run, so a fixed unit renders one of those two cases uselessly.
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= size:
            return f"{n / size:.2f}{unit}"
    return f"{n:.0f}"
