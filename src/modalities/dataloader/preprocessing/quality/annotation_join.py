"""Attaches external per-document annotations to a dataset's sidecar.

The two sides of this join are both far too large to hold in memory -- one annotation
split alone runs to billions of rows -- so neither can be turned into a hash table. Both
sides are instead partitioned by a hash of the join key into buckets small enough to
join one at a time. Documents whose key appears in no annotation shard keep null
labels; that is the normal outcome for a split that has only been partly downloaded,
and the join reports how often it happens rather than hiding it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from modalities.utils.logger_utils import get_logger

# Annotation columns worth carrying into the sidecar. The annotation corpora also hold
# free-text and list-valued columns; those are read on demand rather than copied onto
# every document, because they cannot be aggregated into the cube anyway.
DEFAULT_LABEL_COLUMNS: tuple[str, ...] = (
    "content_integrity",
    "content_quality",
    "information_density",
    "reasoning_indicators",
    "educational_value",
    "content_safety",
    "pii_presence",
    "audience_level",
    "commercial_bias",
    "time_sensitivity",
    "content_ratio",
    "content_length",
)

KEY_COLUMN = "id"


class AnnotationJoinError(RuntimeError):
    """Raised when a join cannot be carried out as specified."""


@dataclass
class JoinReport:
    """What the join did, in numbers worth acting on.

    Attributes:
        dataset (str): Dataset that was joined.
        split (str): Annotation split used.
        n_documents (int): Documents in the sidecar.
        n_matched (int): Documents that received labels.
        n_annotation_rows (int): Annotation rows read.
        n_duplicate_keys (int): Annotation keys seen more than once. Duplicates are
            real in at least one published split, so they are counted rather than
            assumed away.
        n_missing_key (int): Documents whose sidecar row had no join key at all.
        label_columns (list[str]): Columns actually copied across.
    """

    dataset: str
    split: str
    n_documents: int = 0
    n_matched: int = 0
    n_annotation_rows: int = 0
    n_duplicate_keys: int = 0
    n_missing_key: int = 0
    label_columns: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of documents that received labels.

        Returns:
            float: Matched documents over total documents; 0.0 for an empty sidecar.
        """
        return self.n_matched / self.n_documents if self.n_documents else 0.0

    def to_dict(self) -> dict:
        """Renders the report as a plain dictionary.

        Returns:
            dict: Report fields plus the derived coverage.
        """
        return {
            "dataset": self.dataset,
            "split": self.split,
            "n_documents": self.n_documents,
            "n_matched": self.n_matched,
            "coverage": round(self.coverage, 6),
            "n_annotation_rows": self.n_annotation_rows,
            "n_duplicate_keys": self.n_duplicate_keys,
            "n_missing_key": self.n_missing_key,
            "label_columns": self.label_columns,
        }

    def summary(self) -> str:
        """One-line human-readable summary.

        Returns:
            str: Dataset, coverage and the counts a reader should notice.
        """
        return (
            f"{self.dataset}: {self.n_matched:,}/{self.n_documents:,} documents annotated "
            f"({self.coverage:.1%}) from {self.n_annotation_rows:,} annotation rows"
            + (f", {self.n_duplicate_keys:,} duplicate keys" if self.n_duplicate_keys else "")
            + (f", {self.n_missing_key:,} without a key" if self.n_missing_key else "")
        )


def bucket_of(key: str, n_buckets: int) -> int:
    """Assigns a join key to a bucket.

    Args:
        key (str): The join key.
        n_buckets (int): Number of buckets.

    Returns:
        int: Bucket index in ``[0, n_buckets)``.

    Note:
        Uses blake2b rather than the built-in ``hash``, whose string seed varies per
        process. Both sides of the join are bucketed in separate runs, so an unstable
        hash would silently send matching keys to different buckets.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_buckets


class _BucketWriter:
    # Keeps one open parquet writer per bucket so each row is written exactly once,
    # without buffering a whole side of the join in memory.
    def __init__(self, out_dir: Path, schema: pa.Schema, n_buckets: int, flush_rows: int = 100_000):
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._n_buckets = n_buckets
        self._flush_rows = flush_rows
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._buffers: dict[int, list[dict]] = {}

    def add(self, bucket: int, row: dict) -> None:
        buffer = self._buffers.setdefault(bucket, [])
        buffer.append(row)
        if len(buffer) >= self._flush_rows:
            self._flush(bucket)

    def _flush(self, bucket: int) -> None:
        buffer = self._buffers.get(bucket)
        if not buffer:
            return
        if bucket not in self._writers:
            path = self._out_dir / f"bucket-{bucket:04d}.parquet"
            self._writers[bucket] = pq.ParquetWriter(path, self._schema, compression="zstd")
        self._writers[bucket].write_table(pa.Table.from_pylist(buffer, schema=self._schema))
        self._buffers[bucket] = []

    def close(self) -> None:
        for bucket in list(self._buffers):
            self._flush(bucket)
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def bucket_annotations(
    shard_paths: list[Path],
    out_dir: Path,
    n_buckets: int,
    label_columns: Optional[list[str]] = None,
    key_column: str = KEY_COLUMN,
    normalize_key: Optional[str] = None,
    show_progress: bool = True,
) -> tuple[int, list[str]]:
    """Partitions annotation shards by a hash of their key.

    Args:
        shard_paths (list[Path]): Annotation parquet shards of one split.
        out_dir (Path): Directory receiving the bucket files. Cleared first, so a
            partial previous run cannot contribute stale rows.
        n_buckets (int): Number of buckets. Higher means smaller working set per join
            step; a split of billions of rows wants at least 1024.
        label_columns (Optional[list[str]]): Columns to carry. Defaults to
            ``DEFAULT_LABEL_COLUMNS``, intersected with what the shards actually have.
        key_column (str): Column holding the annotation key.
        normalize_key (Optional[str]): Set to ``"urn_uuid"`` to strip
            ``<urn:uuid:...>`` wrappers, which occur mixed with bare UUIDs on both
            sides of some joins.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        tuple[int, list[str]]: Rows written, and the label columns actually carried.

    Raises:
        AnnotationJoinError: If no shards are given or the key column is absent.
    """
    if not shard_paths:
        raise AnnotationJoinError("no annotation shards to bucket")

    available = set(pq.ParquetFile(shard_paths[0]).schema_arrow.names)
    if key_column not in available:
        raise AnnotationJoinError(f"annotation shards have no {key_column!r} column; found {sorted(available)}")
    wanted = list(label_columns) if label_columns is not None else list(DEFAULT_LABEL_COLUMNS)
    carried = [c for c in wanted if c in available]
    if not carried:
        raise AnnotationJoinError(
            f"none of the requested label columns exist in the shards; available: {sorted(available)}"
        )

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    schema = pa.schema([pa.field("key", pa.large_string())] + [pa.field(c, pa.large_string()) for c in carried])
    writer = _BucketWriter(out_dir, schema, n_buckets)

    from modalities.dataloader.preprocessing.quality.registry import strip_urn_uuid

    n_rows = 0
    try:
        for shard in tqdm(shard_paths, desc="bucketing annotations", disable=not show_progress):
            parquet_file = pq.ParquetFile(shard)
            for group_idx in range(parquet_file.metadata.num_row_groups):
                table = parquet_file.read_row_group(group_idx, columns=[key_column] + carried)
                keys = table.column(key_column).to_pylist()
                columns = {c: table.column(c).to_pylist() for c in carried}
                for i, key in enumerate(keys):
                    if key is None:
                        continue
                    key = str(key)
                    if normalize_key == "urn_uuid":
                        key = strip_urn_uuid(key)
                    row = {"key": key}
                    for c in carried:
                        value = columns[c][i]
                        row[c] = None if value is None else str(value)
                    writer.add(bucket_of(key, n_buckets), row)
                    n_rows += 1
    finally:
        writer.close()

    (out_dir / "_meta.json").write_text(
        json.dumps({"n_buckets": n_buckets, "label_columns": carried, "n_rows": n_rows})
    )
    return n_rows, carried


def _iter_sidecar_parts(sidecar_dir: Path) -> list[Path]:
    parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    if not parts:
        raise AnnotationJoinError(f"no sidecar parts found in {sidecar_dir}")
    return parts


def join_annotations(
    sidecar_dir: Path,
    annotation_bucket_dir: Path,
    dataset_name: str,
    split_name: str,
    duplicate_policy: str = "first",
    show_progress: bool = True,
) -> JoinReport:
    """Copies annotation labels onto a dataset's sidecar, in place.

    Args:
        sidecar_dir (Path): Directory of sidecar parts to enrich.
        annotation_bucket_dir (Path): Output of :func:`bucket_annotations`.
        dataset_name (str): Dataset name, for the report.
        split_name (str): Annotation split name, for the report.
        duplicate_policy (str): What to do when one key carries several annotation
            rows. ``"first"`` keeps the first row seen; ``"error"`` refuses to join.
        show_progress (bool): Whether to show progress bars.

    Returns:
        JoinReport: Coverage and the counts needed to judge whether a selection built
            on these labels is meaningful.

    Raises:
        AnnotationJoinError: If the bucket directory is unusable, or duplicates are
            found under ``duplicate_policy="error"``.
    """
    annotation_bucket_dir = Path(annotation_bucket_dir)
    meta_path = annotation_bucket_dir / "_meta.json"
    if not meta_path.is_file():
        raise AnnotationJoinError(f"{annotation_bucket_dir} has no _meta.json; run bucket_annotations first")
    meta = json.loads(meta_path.read_text())
    n_buckets = meta["n_buckets"]
    label_columns: list[str] = meta["label_columns"]

    parts = _iter_sidecar_parts(sidecar_dir)
    report = JoinReport(dataset=dataset_name, split=split_name, label_columns=label_columns)
    report.n_annotation_rows = meta.get("n_rows", 0)

    # Which buckets this dataset actually needs. A dataset is usually far smaller than
    # the split it joins against, so most buckets still have to be read, but only once.
    for part in tqdm(parts, desc=f"join {dataset_name}", disable=not show_progress):
        table = pq.read_table(part)
        keys = table.column("join_key").to_pylist()
        report.n_documents += len(keys)
        report.n_missing_key += sum(1 for k in keys if k is None)

        needed_buckets: dict[int, list[int]] = {}
        for row_idx, key in enumerate(keys):
            if key is None:
                continue
            needed_buckets.setdefault(bucket_of(key, n_buckets), []).append(row_idx)

        resolved: list[dict[str, Optional[str]]] = [{} for _ in keys]
        for bucket, row_indices in needed_buckets.items():
            bucket_path = annotation_bucket_dir / f"bucket-{bucket:04d}.parquet"
            if not bucket_path.is_file():
                continue
            lookup: dict[str, dict[str, Optional[str]]] = {}
            bucket_table = pq.read_table(bucket_path)
            bucket_keys = bucket_table.column("key").to_pylist()
            bucket_columns = {c: bucket_table.column(c).to_pylist() for c in label_columns}
            for i, bucket_key in enumerate(bucket_keys):
                if bucket_key in lookup:
                    report.n_duplicate_keys += 1
                    if duplicate_policy == "error":
                        raise AnnotationJoinError(
                            f"annotation key {bucket_key!r} appears more than once in split {split_name!r}; "
                            "choose duplicate_policy='first' to keep the first occurrence"
                        )
                    continue
                lookup[bucket_key] = {c: bucket_columns[c][i] for c in label_columns}
            for row_idx in row_indices:
                labels = lookup.get(keys[row_idx])
                if labels is not None:
                    resolved[row_idx] = labels

        n_matched_here = sum(1 for r in resolved if r)
        report.n_matched += n_matched_here

        for column in label_columns:
            values = [r.get(column) if r else None for r in resolved]
            array = pa.array(values, type=pa.large_string())
            existing = table.schema.get_field_index(column)
            if existing >= 0:
                table = table.set_column(existing, pa.field(column, pa.large_string()), array)
            else:
                table = table.append_column(pa.field(column, pa.large_string()), array)
        pq.write_table(table, part, compression="zstd")

    get_logger(name="main").info(report.summary())
    return report
