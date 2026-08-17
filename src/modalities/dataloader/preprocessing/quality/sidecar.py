"""Builds the per-document table a blend's selection is computed from.

This is the one pass that has to read the raw data. For every document it records
where the document lives, how many tokens it is expected to contribute, the key that
joins it to external annotations, and whatever quality signals its own record already
carries. Everything downstream -- annotation join, aggregation, previewing a selection,
writing a filtered index -- works from this table and never reads the JSONL again.

The position columns are what make the later steps cheap: a selection is materialised
by writing out the ``(byte_offset, byte_len)`` pairs of the documents that survived,
which is exactly the on-disk format of a modalities index file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import jq
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from modalities.dataloader.create_index import IndexGenerator
from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.preprocessing.quality.registry import DatasetEntry, KeyKind, SourcePointerResolver
from modalities.dataloader.preprocessing.quality.tokens import TokenCalibration
from modalities.utils.logger_utils import get_logger

# Columns every sidecar carries, whatever dataset it describes. Native metrics and
# annotation columns are added alongside these.
BASE_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("file_id", pa.uint32()),
    ("line_no", pa.uint32()),
    ("byte_offset", pa.uint64()),
    ("byte_len", pa.uint32()),
    ("text_bytes", pa.uint32()),
    ("est_tokens", pa.uint32()),
    ("join_key", pa.large_string()),
)


class SidecarWriteError(RuntimeError):
    """Raised when a sidecar cannot be produced for a dataset."""


# One path segment of a jq expression: a bare identifier, or a quoted key for names that
# are not valid identifiers (``."openlid-v3"``).
_PATH_SEGMENT = re.compile(r'\.(?:([A-Za-z_][A-Za-z0-9_]*)|"((?:[^"\\]|\\.)*)")')


def parse_simple_path(jq_pattern: str) -> Optional[list[str]]:
    """Recognises a jq pattern that is nothing more than a chain of field lookups.

    Args:
        jq_pattern (str): The pattern from a native-metric declaration.

    Returns:
        Optional[list[str]]: The field names to walk, or None if the pattern uses
            anything beyond plain field access -- filters, pipes, indexing, functions.

    Note:
        This exists for speed, and the speed difference is not marginal.
        ``jq.compile(...).input_value(record)`` converts the *whole* record into jq's
        own representation on every call, so on a 21 KB document two such calls cost
        more than twenty times the rest of building a sidecar row. Documents are read
        by the billion here, so plain field access is used wherever the pattern allows
        it and jq is kept only for patterns that genuinely need it.
    """
    pattern = jq_pattern.strip()
    if not pattern.startswith("."):
        return None
    keys: list[str] = []
    position = 0
    while position < len(pattern):
        match = _PATH_SEGMENT.match(pattern, position)
        if match is None:
            return None
        bare, quoted = match.group(1), match.group(2)
        keys.append(bare if bare is not None else quoted.replace('\\"', '"').replace("\\\\", "\\"))
        position = match.end()
    return keys or None


def _lookup_path(record: Any, keys: list[str]) -> Any:
    # Mirrors jq's behaviour for a field chain: a missing key, or a non-object where an
    # object is needed, yields no value rather than an error.
    current = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def build_metric_extractor(jq_pattern: str) -> tuple[Callable[[dict[str, Any]], Any], bool]:
    """Builds the fastest available extractor for a native-metric pattern.

    Args:
        jq_pattern (str): The pattern from a native-metric declaration.

    Returns:
        tuple[Callable[[dict[str, Any]], Any], bool]: A function pulling the value out
            of a decoded record, and whether it took the plain-path route. The flag is
            reported so a pattern that silently fell back to jq -- and therefore costs
            twenty times more per document -- is visible rather than a mystery.
    """
    keys = parse_simple_path(jq_pattern)
    if keys is not None:
        return (lambda record: _lookup_path(record, keys)), True

    program = jq.compile(jq_pattern)

    def extract_with_jq(record: dict[str, Any]) -> Any:
        try:
            return program.input_value(record).first()
        except (ValueError, StopIteration):
            return None

    return extract_with_jq, False


def _aggregate(values: Any, aggregation: Optional[str]) -> Optional[float]:
    # Several corpora store a per-page array of scores rather than one document score.
    # Without an aggregation the array cannot become a column, so the first element is
    # used, which is the common case of a single-page document.
    if values is None:
        return None
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return float(values)
    if isinstance(values, list):
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not numeric:
            return None
        if aggregation in (None, "first"):
            return numeric[0]
        if aggregation == "min":
            return min(numeric)
        if aggregation == "max":
            return max(numeric)
        if aggregation == "mean":
            return sum(numeric) / len(numeric)
        raise ValueError(f"unknown aggregation {aggregation!r}")
    return None


def ensure_index(jsonl_path: Path, index_path: Optional[Path] = None) -> Path:
    """Returns the index for a JSONL file, creating it if it does not exist.

    Args:
        jsonl_path (Path): The JSONL file.
        index_path (Optional[Path]): Where the index lives. Defaults to the file's
            own ``.idx`` sibling, matching ``LargeFileLinesReader``'s convention.

    Returns:
        Path: Path to a usable index file.
    """
    index_path = LargeFileLinesReader.default_index_path(jsonl_path, index_path)
    if not index_path.is_file():
        get_logger(name="main").info(f"Creating missing index for {jsonl_path} ...")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        IndexGenerator(jsonl_path).create_index(index_path)
    return index_path


class SidecarBuilder:
    """Produces one dataset's per-document table.

    The builder is deliberately per-file: each JSONL file yields an independent
    parquet part, so a dataset of thousands of files can be built by as many parallel
    tasks, and a failed shard can be rebuilt on its own.
    """

    def __init__(
        self,
        dataset: DatasetEntry,
        calibration: TokenCalibration,
        index_root: Optional[Path] = None,
        row_group_size: int = 200_000,
    ):
        """
        Args:
            dataset (DatasetEntry): The dataset being described.
            calibration (TokenCalibration): Token estimator for this dataset.
            index_root (Optional[Path]): Directory holding index files, if they are not
                kept next to the JSONL. Source trees are often read-only.
            row_group_size (int): Parquet row group size for the output.
        """
        self._dataset = dataset
        self._calibration = calibration
        self._index_root = index_root
        self._row_group_size = row_group_size
        self._native_programs = []
        slow_patterns: list[str] = []
        for metric in dataset.native_metrics:
            extractor, is_fast = build_metric_extractor(metric.jq_pattern)
            self._native_programs.append((metric.name, extractor, metric.aggregation))
            if not is_fast:
                slow_patterns.append(f"{metric.name}={metric.jq_pattern}")
        if slow_patterns:
            get_logger(name="main").warning(
                f"{dataset.name}: {len(slow_patterns)} native metric(s) need jq and will dominate the pass "
                f"({', '.join(slow_patterns)}). Rewrite as a plain field path if possible."
            )

    def _index_path_for(self, jsonl_path: Path) -> Path:
        if self._index_root is None:
            return LargeFileLinesReader.default_index_path(jsonl_path, None)
        relative = jsonl_path.relative_to(self._dataset.jsonl_root)
        return Path(self._index_root) / relative.with_suffix(".idx")

    def schema(self) -> pa.Schema:
        """Builds the parquet schema for this dataset's sidecar.

        Returns:
            pa.Schema: Base position/token columns plus one column per native metric.
        """
        fields = [pa.field(name, dtype) for name, dtype in BASE_FIELDS]
        fields += [pa.field(f"native_{name}", pa.float64()) for name, _, _ in self._native_programs]
        return pa.schema(fields)

    def _rows_for_file(self, jsonl_path: Path, file_id: int) -> Iterator[dict[str, Any]]:
        index_path = ensure_index(jsonl_path, self._index_path_for(jsonl_path))
        reader = LargeFileLinesReader(jsonl_path, index_path=index_path)
        key_spec = self._dataset.key
        text_field = self._dataset.text_field
        try:
            for line_no, (byte_offset, byte_len) in enumerate(reader.index):
                try:
                    record = json.loads(reader[line_no])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # A single malformed line must not abort a multi-terabyte pass; it
                    # is dropped, and its absence shows up as a row-count mismatch
                    # against the index.
                    continue

                text = record.get(text_field)
                text_bytes = len(text.encode("utf-8")) if isinstance(text, str) else 0

                row: dict[str, Any] = {
                    "file_id": file_id,
                    "line_no": line_no,
                    "byte_offset": byte_offset,
                    "byte_len": byte_len,
                    "text_bytes": text_bytes,
                    "est_tokens": self._calibration.estimate(record, text_bytes),
                    "join_key": key_spec.derive(record) if key_spec is not None else None,
                }
                for name, extract, aggregation in self._native_programs:
                    row[f"native_{name}"] = _aggregate(extract(record), aggregation)
                yield row
        finally:
            reader.close()

    def build_file(self, jsonl_path: Path, file_id: int, output_path: Path) -> int:
        """Builds the sidecar part for one JSONL file.

        Args:
            jsonl_path (Path): The JSONL file to describe.
            file_id (int): Index of this file within the dataset's sorted file list.
                Stored per row so a selection can be mapped back to its source file.
            output_path (Path): Destination parquet path. Parents are created.

        Returns:
            int: Number of documents written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        schema = self.schema()
        n_rows = 0
        batch: list[dict[str, Any]] = []
        writer = pq.ParquetWriter(output_path, schema, compression="zstd")
        try:
            for row in self._rows_for_file(jsonl_path, file_id):
                batch.append(row)
                if len(batch) >= self._row_group_size:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    n_rows += len(batch)
                    batch = []
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                n_rows += len(batch)
        finally:
            writer.close()
        return n_rows

    def build(
        self,
        output_dir: Path,
        file_ids: Optional[list[int]] = None,
        show_progress: bool = True,
    ) -> dict[str, int]:
        """Builds sidecar parts for the dataset's files.

        Args:
            output_dir (Path): Directory receiving one parquet part per file.
            file_ids (Optional[list[int]]): Restrict the build to these file ids, for
                sharding the work across tasks. None builds every file.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            dict[str, int]: Maps output part name to documents written.

        Raises:
            SidecarWriteError: If the dataset matches no files, which usually means a
                wrong root or glob rather than an empty corpus.
        """
        files = self._dataset.iter_files()
        if not files:
            raise SidecarWriteError(
                f"dataset {self._dataset.name!r}: no files matched {self._dataset.glob!r} "
                f"under {self._dataset.jsonl_root}"
            )
        selected = [(i, p) for i, p in enumerate(files) if file_ids is None or i in set(file_ids)]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, int] = {}
        iterator = tqdm(selected, desc=f"sidecar {self._dataset.name}", disable=not show_progress)
        for file_id, jsonl_path in iterator:
            part_name = f"part-{file_id:06d}.parquet"
            written[part_name] = self.build_file(jsonl_path, file_id, output_dir / part_name)
        return written


def resolve_source_pointers(
    sidecar_dir: Path,
    dataset: DatasetEntry,
    batch_size: int = 500_000,
    only_parts: Optional[list[int]] = None,
) -> int:
    """Rewrites pointer join keys into the annotation keys they stand for.

    A translated corpus stores a ``<file>/<line>`` pointer back to the document it was
    translated from. The annotation belongs to that original, so the pointer has to be
    exchanged for a hash of the original's text before the join can happen.

    Args:
        sidecar_dir (Path): Directory of sidecar parts to rewrite in place.
        dataset (DatasetEntry): The dataset, whose key spec supplies the source root.
        batch_size (int): How many pointers to resolve per pass over the source files.
        only_parts (Optional[list[int]]): Restrict to the parts of these file ids. Lets a
            sharded build resolve only the parts it wrote, leaving the rest to the tasks
            that own them.

    Returns:
        int: Number of rows whose key was resolved.

    Raises:
        ValueError: If the dataset's key is not a source pointer.
    """
    if dataset.key is None or dataset.key.kind != KeyKind.SOURCE_POINTER:
        raise ValueError(f"dataset {dataset.name!r} does not use a source-pointer key")

    resolver = SourcePointerResolver(
        source_root=dataset.key.source_root,
        text_field=dataset.key.text_field,
        line_offset=dataset.key.source_line_offset,
    )
    if only_parts is None:
        parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    else:
        candidates = (Path(sidecar_dir) / f"part-{file_id:06d}.parquet" for file_id in only_parts)
        parts = [path for path in candidates if path.is_file()]
    n_resolved = 0
    for part in tqdm(parts, desc=f"resolve pointers {dataset.name}"):
        table = pq.read_table(part)
        pointers = [p for p in table.column("join_key").to_pylist() if p is not None]
        if not pointers:
            continue
        mapping: dict[str, str] = {}
        unique_pointers = sorted(set(pointers))
        for start in range(0, len(unique_pointers), batch_size):
            mapping.update(resolver.resolve(unique_pointers[start : start + batch_size]))
        resolved = [mapping.get(p) if p is not None else None for p in table.column("join_key").to_pylist()]
        n_resolved += sum(1 for r in resolved if r is not None)
        table = table.set_column(
            table.schema.get_field_index("join_key"),
            pa.field("join_key", pa.large_string()),
            pa.array(resolved, type=pa.large_string()),
        )
        pq.write_table(table, part, compression="zstd")
    return n_resolved
