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
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
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
        n_parts_resumed (int): Parts that already carried labels and were not re-joined.
            Their documents still count towards the totals above, read back from the
            labels already on disk: coverage describes the sidecar, not the run, and a
            resumed join that reported 0% would read as a failed one.
        label_columns (list[str]): Columns actually copied across.
    """

    dataset: str
    split: str
    n_documents: int = 0
    n_matched: int = 0
    n_annotation_rows: int = 0
    n_duplicate_keys: int = 0
    n_missing_key: int = 0
    n_parts_resumed: int = 0
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
            "n_parts_resumed": self.n_parts_resumed,
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
            + (f", {self.n_parts_resumed:,} parts already joined" if self.n_parts_resumed else "")
        )


def _metadata_paths(annotation_bucket_dir: Path) -> list[Path]:
    # Excludes the `.tmp` files an interrupted write can leave behind, so a half-written
    # file is never mistaken for a task's metadata.
    return sorted(p for p in Path(annotation_bucket_dir).glob("_meta.*.json") if p.suffix == ".json")


def _read_metadata(path: Path) -> Optional[dict]:
    """Reads one bucketing metadata file, returning None if it cannot be used.

    Args:
        path (Path): Path to a ``_meta.<shard>.json`` file.

    Returns:
        Optional[dict]: The parsed metadata, or None if the file is absent or unparseable.

    Note:
        Metadata is written atomically, so an unparseable file should not occur. It is
        tolerated anyway because the alternative is a fifteen-hour pipeline dying on one
        unreadable sidecar file -- which is exactly what happened when this was a bare
        ``json.loads``: tasks read a sibling's file mid-write and 12 of 64 crashed.
        Skipping is also the safe direction for :func:`read_bucket_metadata`, where a
        missing entry makes the run look incomplete rather than joinable.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_metadata(path: Path, payload: dict) -> None:
    """Writes bucketing metadata so a concurrent reader never sees a partial file.

    Args:
        path (Path): Final path of the metadata file.
        payload (dict): Content to write.

    Note:
        ``Path.write_text`` truncates before writing, leaving a window in which the file
        is empty. Sibling tasks of the same array read these files, so that window is a
        real race. Writing to a per-task temporary name and renaming makes the swap
        atomic: a reader sees either the old file or the complete new one.
    """
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


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
    for stale in _metadata_paths(out_dir):
        meta = _read_metadata(stale)
        if meta is None:
            continue
        previous = meta.get("num_shards", 1)
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

    # One metadata file per task, so concurrent tasks never overwrite each other's, and
    # written atomically so a sibling reading the directory cannot catch it half-written.
    _write_metadata(
        out_dir / f"_meta.{shard_id:04d}.json",
        {
            "n_buckets": n_buckets,
            "label_columns": carried,
            "n_rows": n_rows,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "n_input_shards": len(my_shards),
        },
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
    metadata_paths = _metadata_paths(annotation_bucket_dir)
    if not metadata_paths:
        raise AnnotationJoinError(
            f"{annotation_bucket_dir} holds no bucketing metadata; run 'modalities quality bucket-annotations' first"
        )
    merged: Optional[dict] = None
    total_rows = 0
    seen_shards: set[int] = set()
    for path in metadata_paths:
        meta = _read_metadata(path)
        if meta is None:
            # Leaving the shard id unseen makes the run report as incomplete, which is the
            # safe outcome: better to refuse the join than to join missing annotations.
            continue
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

    if merged is None:
        raise AnnotationJoinError(
            f"{annotation_bucket_dir} has {len(metadata_paths)} metadata file(s) but none could be read; "
            "the bucketing run did not complete. Delete the directory and re-bucket the split."
        )

    expected = merged.get("num_shards", 1)
    if len(seen_shards) != expected:
        missing = sorted(set(range(expected)) - seen_shards)
        raise AnnotationJoinError(
            f"{annotation_bucket_dir} is incomplete: {len(seen_shards)} of {expected} bucketing task(s) "
            f"finished, missing shard id(s) {missing}. Joining now would lose their annotations."
        )
    return {"n_buckets": merged["n_buckets"], "label_columns": merged["label_columns"], "n_rows": total_rows}


def _part_has_labels(part: Path, label_columns: list[str]) -> bool:
    """Whether a sidecar part has already been written back by a join.

    Args:
        part (Path): The sidecar part.
        label_columns (list[str]): Columns the join adds.

    Returns:
        bool: True if every label column is present. Reads only the parquet footer, so
            this is cheap enough to check for every part of a large dataset.
    """
    try:
        names = set(pq.ParquetFile(part).schema_arrow.names)
    except OSError:
        return False
    return all(column in names for column in label_columns)


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
    max_batch_keys: int = 20_000_000,
    resume: bool = False,
    show_progress: bool = True,
) -> JoinReport:
    """Copies annotation labels onto a dataset's sidecar, in place.

    Sidecar parts are processed in batches, and each annotation bucket is read once per
    batch rather than once per part. That distinction decides whether this finishes:
    reading the bucketed split per part meant 454 TB of reads over the real blend --
    5,319 parts for Nemotron-CC against a 23 GB split -- because every part's documents
    hash across every bucket. Batching turns the read amplification from the part count
    into the far smaller number of batches.

    Within a batch each bucket is filtered to the keys the batch actually wants before
    anything is materialised in Python, so memory is bounded by the batch rather than by
    the bucket -- a bucket of a billion-row split holds millions of rows, of which a batch
    typically wants a few thousand.

    Args:
        sidecar_dir (Path): Directory of sidecar parts to enrich.
        annotation_bucket_dir (Path): Output of :func:`bucket_annotations`.
        dataset_name (str): Dataset name, for the report.
        split_name (str): Annotation split name, for the report.
        duplicate_policy (str): What to do when one key carries several annotation rows.
            ``"first"`` keeps the first row seen; ``"error"`` refuses to join.
        max_batch_keys (int): Documents to hold per batch. Larger batches read the
            annotation side fewer times but hold more keys in memory.
        resume (bool): Skip parts that already carry the label columns, to continue an
            interrupted run. The write-back adds columns and values together, so a part
            having them means it was processed. Off by default: re-bucketing the
            annotations and then resuming would silently keep the old labels, so
            continuing has to be asked for.
        show_progress (bool): Whether to show progress bars.

    Returns:
        JoinReport: Coverage and the counts needed to judge whether a selection built on
            these labels is meaningful.

    Raises:
        AnnotationJoinError: If the bucket directory is unusable, or duplicates are found
            under ``duplicate_policy="error"``.
    """
    annotation_bucket_dir = Path(annotation_bucket_dir)
    # Read for its own sake as well as for the label columns: it refuses an incomplete
    # bucketing run rather than letting the join silently drop a missing task's annotations.
    meta = read_bucket_metadata(annotation_bucket_dir)
    label_columns: list[str] = meta["label_columns"]

    parts = _iter_sidecar_parts(sidecar_dir)
    report = JoinReport(dataset=dataset_name, split=split_name, label_columns=label_columns)
    report.n_annotation_rows = meta.get("n_rows", 0)

    # Globbed once for the whole split rather than per bucket. Routing keys to buckets is
    # gone: a batch holds millions of keys, which hash across every bucket, so every bucket
    # file was read anyway -- the profile confirmed 1024 of 1024 -- and the routing cost a
    # blake2b call per key in Python to decide nothing.
    all_bucket_files = sorted(annotation_bucket_dir.glob("bucket-*.parquet"))
    if not all_bucket_files:
        raise AnnotationJoinError(
            f"no bucket files in {annotation_bucket_dir}; run 'quality bucket-annotations' first"
        )

    def flush(batch: list[tuple[Path, pa.Table]]) -> None:
        """Resolves one batch of parts and writes their label columns back.

        Resolution stays in Arrow from end to end. The obvious implementation --
        materialise the keys, build a dict from key to row, look every document up, emit a
        list per label column -- costs a handful of Python objects per document, and at
        1.7 bn documents for Nemotron-CC that was twelve hours. Here each part's labels come
        from one ``index_in`` against the batch's lookup table followed by one ``take`` per
        label column, so the per-document work happens in Arrow's kernels.

        The single remaining Python loop is over the batch's *unique* keys, to route them to
        buckets. That one cannot be vectorised: both sides of the join are bucketed in
        separate runs, so the hash has to be stable across processes, and blake2b is not
        something Arrow's compute layer can do. It is per unique key rather than per
        document, and it is cheap relative to reading the buckets.
        """
        if not batch:
            return

        chunks: list[pa.Array] = []
        for _, table in batch:
            for chunk in table.column("join_key").chunks:
                if len(chunk) > 0:
                    chunks.append(chunk.cast(pa.large_string()))
        wanted_keys = (
            pc.drop_null(pc.unique(pa.chunked_array(chunks, type=pa.large_string())))
            if chunks
            else pa.array([], type=pa.large_string())
        )

        # One scan over the split with the key filter pushed into it, instead of a read and
        # an is_in per bucket file. The split is partitioned into a thousand files of a few
        # hundred kilobytes, so opening and scanning them individually cost more than the
        # data itself: 22 s of reads and 14 s of a thousand separate is_in calls, against
        # 47 s total. Arrow applies the filter per row group while scanning and reads the
        # files in parallel, so memory stays bounded by the rows that match rather than by
        # the size of the split.
        pieces: list[pa.Table] = []
        if len(wanted_keys) > 0:
            dataset = ds.dataset(all_bucket_files, format="parquet")
            matched = dataset.to_table(
                columns=["key"] + label_columns,
                filter=ds.field("key").isin(wanted_keys),
            )
            if matched.num_rows:
                pieces.append(matched)

        lookup_keys: Optional[pa.Array] = None
        lookup: Optional[pa.Table] = None
        if pieces:
            lookup = pa.concat_tables(pieces, promote_options="permissive")
            keys_column = lookup.column("key").cast(pa.large_string()).combine_chunks()
            unique_keys = pc.unique(keys_column)
            n_duplicates = len(keys_column) - len(unique_keys)
            if n_duplicates:
                report.n_duplicate_keys += n_duplicates
                if duplicate_policy == "error":
                    counts = pc.value_counts(keys_column)
                    repeated = counts.field("values").filter(pc.greater(counts.field("counts"), 1))
                    example = repeated[0].as_py() if len(repeated) else "<unknown>"
                    raise AnnotationJoinError(
                        f"annotation key {example!r} appears more than once in split "
                        f"{split_name!r}; choose duplicate_policy='first' to keep the first"
                    )
                # index_in reports the first position of each value, which is exactly the
                # "keep the first row seen" policy, done in one pass instead of a loop.
                lookup = lookup.take(pc.index_in(unique_keys, value_set=keys_column))
                lookup_keys = unique_keys
            else:
                lookup_keys = keys_column

        for part, table in batch:
            keys = table.column("join_key").cast(pa.large_string())
            if lookup is None or lookup_keys is None or len(lookup_keys) == 0:
                indices = None
            else:
                indices = pc.index_in(keys, value_set=lookup_keys)
                report.n_matched += len(indices) - indices.null_count

            for column in label_columns:
                if indices is None:
                    array = pa.nulls(table.num_rows, type=pa.large_string())
                else:
                    # A null index yields a null label, so unmatched documents fall out
                    # correctly without being special-cased.
                    array = pc.take(lookup.column(column), indices).cast(pa.large_string())
                existing = table.schema.get_field_index(column)
                if existing >= 0:
                    table = table.set_column(existing, pa.field(column, pa.large_string()), array)
                else:
                    table = table.append_column(pa.field(column, pa.large_string()), array)
            pq.write_table(table, part, compression="zstd")

    batch: list[tuple[Path, pa.Table]] = []
    batch_keys = 0
    n_skipped = 0
    for part in tqdm(parts, desc=f"join {dataset_name}", disable=not show_progress):
        if resume and label_columns and _part_has_labels(part, label_columns):
            # Count what this part already holds instead of ignoring it, so a resumed run
            # reports the sidecar's real coverage rather than only what it happened to do.
            n_skipped += 1
            existing = pq.read_table(part, columns=["join_key", label_columns[0]])
            report.n_documents += existing.num_rows
            report.n_matched += existing.num_rows - existing.column(label_columns[0]).null_count
            report.n_missing_key += existing.column("join_key").null_count
            continue
        table = pq.read_table(part)
        # Kept as Arrow: materialising this column was 1.7 bn Python strings for the
        # largest dataset, before any joining had happened at all.
        report.n_documents += table.num_rows
        report.n_missing_key += table.column("join_key").null_count
        batch.append((part, table))
        batch_keys += table.num_rows
        if batch_keys >= max_batch_keys:
            flush(batch)
            batch, batch_keys = [], 0
    flush(batch)

    report.n_parts_resumed = n_skipped
    if n_skipped:
        get_logger(name="main").info(
            f"{dataset_name}: resumed, skipped {n_skipped:,} of {len(parts):,} parts that already carried labels"
        )
    get_logger(name="main").info(report.summary())
    return report
