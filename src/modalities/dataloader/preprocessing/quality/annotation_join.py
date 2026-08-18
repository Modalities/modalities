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
    """Streams rows out to one parquet file per bucket, with bounded memory.

    The shard suffix in the filename lets many tasks bucket one split at once: each
    writes its own file per bucket, and the join reads every file belonging to a bucket.

    Memory is capped on the **total** rows held across all buckets, not per bucket. A
    per-bucket threshold does not bound anything: with 1024 buckets a 50 M-row input
    puts only ~49 k rows in each, so no bucket ever reaches a 100 k threshold and the
    whole input ends up buffered as Python dicts. That is how this OOM-killed all 64
    tasks of a real run at 24 GB each.
    """

    def __init__(
        self,
        out_dir: Path,
        schema: pa.Schema,
        n_buckets: int,
        shard_id: int = 0,
        max_buffered_rows: int = 500_000,
    ):
        """
        Args:
            out_dir (Path): Directory receiving the bucket files.
            schema (pa.Schema): Schema of the rows being written.
            n_buckets (int): Number of buckets.
            shard_id (int): This task's index, used to name its files.
            max_buffered_rows (int): Rows held across all buckets before everything is
                flushed. Bounds memory regardless of the bucket count.
        """
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._n_buckets = n_buckets
        self._shard_id = shard_id
        self._max_buffered_rows = max_buffered_rows
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._buffers: dict[int, list[dict]] = {}
        self._buffered_rows = 0

    def add(self, bucket: int, row: dict) -> None:
        """Queues one row for its bucket, flushing everything if memory is up.

        Args:
            bucket (int): Bucket the row belongs to.
            row (dict): The row, matching the writer's schema.
        """
        self._buffers.setdefault(bucket, []).append(row)
        self._buffered_rows += 1
        if self._buffered_rows >= self._max_buffered_rows:
            self.flush_all()

    def flush_all(self) -> None:
        """Writes every buffered row out and releases the memory."""
        for bucket in list(self._buffers):
            self._flush(bucket)
        self._buffered_rows = 0

    def _flush(self, bucket: int) -> None:
        buffer = self._buffers.get(bucket)
        if not buffer:
            return
        if bucket not in self._writers:
            path = self._out_dir / f"bucket-{bucket:04d}.{self._shard_id:04d}.parquet"
            self._writers[bucket] = pq.ParquetWriter(path, self._schema, compression="zstd")
        self._writers[bucket].write_table(pa.Table.from_pylist(buffer, schema=self._schema))
        self._buffers[bucket] = []

    def close(self) -> None:
        """Flushes what is left and closes every open parquet writer."""
        self.flush_all()
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
    shard_id: int = 0,
    num_shards: int = 1,
    max_buffered_rows: int = 500_000,
    show_progress: bool = True,
) -> tuple[int, list[str]]:
    """Partitions annotation shards by a hash of their key.

    This is the expensive half of the join, since a split can run to billions of rows,
    so it is shardable: run it as an array of ``num_shards`` tasks, each taking a subset
    of the input shards. The result is identical to a single-task run.

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
        shard_id (int): This task's index in ``[0, num_shards)``.
        num_shards (int): How many tasks are bucketing this split.
        max_buffered_rows (int): Rows held in memory across all buckets before being
            flushed. Raise it for throughput, lower it under a tight memory limit.
        show_progress (bool): Whether to show a progress bar.

    Returns:
        tuple[int, list[str]]: Rows written by this task, and the label columns carried.

    Raises:
        AnnotationJoinError: If no shards are given, the key column is absent, or the
            shard selection is out of range.
    """
    if not shard_paths:
        raise AnnotationJoinError("no annotation shards to bucket")
    if not 0 <= shard_id < num_shards:
        raise AnnotationJoinError(f"shard_id {shard_id} is not in [0, {num_shards})")

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
    # Only a single-task run may clear the directory; sibling tasks are writing into it.
    if num_shards == 1 and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A sharded run cannot clear the directory, so leftovers from a previous run with a
    # different array size would be silently mixed in and the join would read a bucket
    # made of rows from two incompatible runs.
    for stale in out_dir.glob("_meta.*.json"):
        previous = json.loads(stale.read_text()).get("num_shards", 1)
        if previous != num_shards:
            raise AnnotationJoinError(
                f"{out_dir} holds output from a run with num_shards={previous}, but this task has "
                f"num_shards={num_shards}. Delete the directory and re-bucket the split from scratch."
            )
    # Strided rather than contiguous, so tasks stay balanced when shard sizes trend
    # across a split.
    my_shards = [path for i, path in enumerate(sorted(shard_paths)) if i % num_shards == shard_id]
    schema = pa.schema([pa.field("key", pa.large_string())] + [pa.field(c, pa.large_string()) for c in carried])
    writer = _BucketWriter(out_dir, schema, n_buckets, shard_id=shard_id, max_buffered_rows=max_buffered_rows)

    from modalities.dataloader.preprocessing.quality.registry import strip_urn_uuid

    n_rows = 0
    try:
        for shard in tqdm(my_shards, desc="bucketing annotations", disable=not show_progress):
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

    # One metadata file per task, so concurrent tasks never overwrite each other's.
    (out_dir / f"_meta.{shard_id:04d}.json").write_text(
        json.dumps(
            {
                "n_buckets": n_buckets,
                "label_columns": carried,
                "n_rows": n_rows,
                "shard_id": shard_id,
                "num_shards": num_shards,
                "n_input_shards": len(my_shards),
            }
        )
    )
    return n_rows, carried


def read_bucket_metadata(annotation_bucket_dir: Path) -> dict:
    """Merges the metadata a bucketing run left behind, checking it is complete.

    Args:
        annotation_bucket_dir (Path): Directory written by :func:`bucket_annotations`.

    Returns:
        dict: ``n_buckets``, ``label_columns`` and the total ``n_rows`` bucketed.

    Raises:
        AnnotationJoinError: If no metadata is present, the tasks disagree on the bucket
            count or label columns, or some announced task never finished. Joining an
            incomplete run would silently drop the annotations that task was carrying,
            which looks exactly like a corpus that was never annotated.
    """
    metadata_paths = sorted(Path(annotation_bucket_dir).glob("_meta.*.json"))
    if not metadata_paths:
        raise AnnotationJoinError(
            f"{annotation_bucket_dir} holds no bucketing metadata; run 'modalities quality bucket-annotations' first"
        )
    merged: Optional[dict] = None
    total_rows = 0
    seen_shards: set[int] = set()
    for path in metadata_paths:
        meta = json.loads(path.read_text())
        if merged is None:
            merged = meta
        elif (meta["n_buckets"], meta["label_columns"]) != (merged["n_buckets"], merged["label_columns"]):
            raise AnnotationJoinError(
                f"{annotation_bucket_dir} mixes incompatible bucketing runs: "
                f"{meta['n_buckets']} buckets / {meta['label_columns']} vs "
                f"{merged['n_buckets']} / {merged['label_columns']}. Re-bucket the split from scratch."
            )
        total_rows += meta.get("n_rows", 0)
        seen_shards.add(meta.get("shard_id", 0))

    expected = merged.get("num_shards", 1)
    if len(seen_shards) != expected:
        missing = sorted(set(range(expected)) - seen_shards)
        raise AnnotationJoinError(
            f"{annotation_bucket_dir} is incomplete: {len(seen_shards)} of {expected} bucketing task(s) "
            f"finished, missing shard id(s) {missing}. Joining now would lose their annotations."
        )
    return {"n_buckets": merged["n_buckets"], "label_columns": merged["label_columns"], "n_rows": total_rows}


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
    meta = read_bucket_metadata(annotation_bucket_dir)
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
            # A bucket is spread over one file per bucketing task, so all are read
            # together; a bucket no task wrote to simply has no files.
            bucket_paths = sorted(annotation_bucket_dir.glob(f"bucket-{bucket:04d}.*.parquet"))
            if not bucket_paths:
                continue
            lookup: dict[str, dict[str, Optional[str]]] = {}
            bucket_table = pa.concat_tables([pq.read_table(path) for path in bucket_paths])
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
