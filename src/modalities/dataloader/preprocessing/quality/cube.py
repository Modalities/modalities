"""Aggregates a sidecar so any threshold combination can be costed instantly.

Tuning a blend means trying many threshold combinations and asking what each one costs
in tokens. Answering that from the per-document table would mean re-reading billions of
rows for every edit. Answering it from a cube does not: documents are grouped once by
the fields a selection may threshold on, and each group records how many documents and
tokens it holds. Any conjunction of predicates over those fields is then a sum over the
groups that satisfy it -- microseconds, and exact rather than sampled.

The cube is exact for the dimensions it was built over. Predicates over anything else
cannot be answered from it, and the selection engine says so and falls back to the
sidecar rather than quietly approximating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# The propella label columns worth grouping on. Ordinal scales first, then the two
# binary-ish safety fields; together these are what selections actually threshold.
DEFAULT_LABEL_DIMENSIONS: tuple[str, ...] = (
    "educational_value",
    "content_quality",
    "information_density",
    "reasoning_indicators",
    "content_integrity",
    "content_safety",
    "pii_presence",
)

# Marker for documents that carry no annotation. Kept as an explicit dimension value
# rather than dropped, because on a partly downloaded split the unannotated documents
# can be the majority and the policy for them changes the answer completely.
MISSING = "__missing__"

N_SCORE_BINS = 10


class CubeError(RuntimeError):
    """Raised when a cube cannot be built or used as requested."""


@dataclass(frozen=True)
class ScoreBinning:
    """Bin edges turning a continuous native metric into a cube dimension.

    Attributes:
        column (str): Sidecar column the edges were computed from.
        edges (tuple[float, ...]): Ascending bin edges. Bin ``i`` covers
            ``[edges[i], edges[i + 1])``, with the last bin closed at the top.
    """

    column: str
    edges: tuple[float, ...]

    def bin_index(self, values: np.ndarray) -> np.ndarray:
        """Assigns values to bins.

        Args:
            values (np.ndarray): Metric values, possibly containing NaN.

        Returns:
            np.ndarray: Bin index per value; ``-1`` where the value is missing.
        """
        edges = np.asarray(self.edges, dtype=np.float64)
        index = np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)
        return np.where(np.isnan(values), -1, index)

    def lower_bound_of(self, bin_index: int) -> float:
        """Smallest value that can fall in a bin.

        Args:
            bin_index (int): The bin.

        Returns:
            float: The bin's lower edge.
        """
        return self.edges[bin_index]

    def upper_bound_of(self, bin_index: int) -> float:
        """Smallest value above a bin.

        Args:
            bin_index (int): The bin.

        Returns:
            float: The bin's upper edge.
        """
        return self.edges[bin_index + 1]


@dataclass
class Cube:
    """Grouped document and token counts for one dataset.

    Attributes:
        dataset (str): Dataset the cube describes.
        label_dimensions (list[str]): Annotation columns grouped on.
        score_binnings (dict[str, ScoreBinning]): Native metrics grouped on, by name.
        table (pa.Table): One row per non-empty group: the dimension values, plus
            ``n_documents`` and ``n_tokens``.
        n_documents (int): Documents represented.
        n_tokens (int): Estimated tokens represented.
    """

    dataset: str
    label_dimensions: list[str]
    score_binnings: dict[str, ScoreBinning]
    table: pa.Table
    n_documents: int
    n_tokens: int

    @property
    def dimensions(self) -> list[str]:
        """All groupable dimension names.

        Returns:
            list[str]: Label dimensions followed by binned score dimensions.
        """
        return list(self.label_dimensions) + [f"native_{name}" for name in self.score_binnings]

    def write(self, path: Path) -> None:
        """Writes the cube to a parquet file with its metadata embedded.

        Args:
            path (Path): Destination path. Parents are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            b"quality_cube": json.dumps(
                {
                    "dataset": self.dataset,
                    "label_dimensions": self.label_dimensions,
                    "score_binnings": {k: list(v.edges) for k, v in self.score_binnings.items()},
                    "n_documents": self.n_documents,
                    "n_tokens": self.n_tokens,
                }
            ).encode()
        }
        table = self.table.replace_schema_metadata({**(self.table.schema.metadata or {}), **meta})
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def read(cls, path: Path) -> "Cube":
        """Reads a cube written by :meth:`write`.

        Args:
            path (Path): The cube parquet file.

        Returns:
            Cube: The loaded cube.

        Raises:
            CubeError: If the file carries no cube metadata.
        """
        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(b"quality_cube")
        if raw is None:
            raise CubeError(f"{path} is not a quality cube (no metadata)")
        meta = json.loads(raw)
        return cls(
            dataset=meta["dataset"],
            label_dimensions=meta["label_dimensions"],
            score_binnings={
                k: ScoreBinning(column=f"native_{k}", edges=tuple(v)) for k, v in meta["score_binnings"].items()
            },
            table=table,
            n_documents=meta["n_documents"],
            n_tokens=meta["n_tokens"],
        )


def _quantile_edges(values: np.ndarray, n_bins: int) -> Optional[tuple[float, ...]]:
    # Quantile edges keep every bin populated, which matters because native scores are
    # heavily skewed: fixed-width bins would leave most of the range nearly empty and
    # crowd almost all documents into one or two cells.
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(finite, quantiles))
    if edges.size < 2:
        # A metric with a single distinct value cannot be binned, but must still be
        # thresholdable, so it gets one bin wide enough to hold it.
        return (float(edges[0]), float(np.nextafter(edges[0], np.inf)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return tuple(float(e) for e in edges)


def _aggregate_cells(table: pa.Table, dimension_names: list[str]) -> pa.Table:
    # Sums the two count columns per distinct combination of dimension values. Arrow
    # suffixes aggregated columns, and does not promise where it puts them relative to
    # the group keys, so the result is reassembled by name.
    aggregated = table.group_by(dimension_names).aggregate([("n_documents", "sum"), ("n_tokens", "sum")])
    return pa.table(
        {
            **{name: aggregated.column(name) for name in dimension_names},
            "n_documents": aggregated.column("n_documents_sum"),
            "n_tokens": aggregated.column("n_tokens_sum"),
        }
    )


def _sidecar_parts(sidecar_dir: Path) -> list[Path]:
    parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    if not parts:
        raise CubeError(f"no sidecar parts found in {sidecar_dir}")
    return parts


def _sample_scores(parts: list[Path], column: str, max_rows: int) -> np.ndarray:
    collected: list[np.ndarray] = []
    total = 0
    for part in parts:
        parquet_file = pq.ParquetFile(part)
        if column not in parquet_file.schema_arrow.names:
            continue
        for group_idx in range(parquet_file.metadata.num_row_groups):
            chunk = (
                parquet_file.read_row_group(group_idx, columns=[column]).column(column).to_numpy(zero_copy_only=False)
            )
            chunk = chunk.astype(np.float64)
            collected.append(chunk)
            total += chunk.size
            if total >= max_rows:
                return np.concatenate(collected)[:max_rows]
    return np.concatenate(collected) if collected else np.empty(0, dtype=np.float64)


def build_cube(
    sidecar_dir: Path,
    dataset_name: str,
    label_dimensions: Iterable[str] = DEFAULT_LABEL_DIMENSIONS,
    score_columns: Optional[Iterable[str]] = None,
    n_score_bins: int = N_SCORE_BINS,
    binning_sample_rows: int = 2_000_000,
    aggregate_batch_rows: int = 8_000_000,
) -> Cube:
    """Groups a sidecar into a cube.

    Args:
        sidecar_dir (Path): Directory of sidecar parts, after the annotation join.
        dataset_name (str): Dataset name recorded in the cube.
        label_dimensions (Iterable[str]): Annotation columns to group on. Columns
            absent from the sidecar are skipped, so an unannotated dataset still gets
            a usable cube over its native metrics.
        score_columns (Optional[Iterable[str]]): Native metric columns to bin and group
            on, named without the ``native_`` prefix. None uses every native column
            present.
        n_score_bins (int): Quantile bins per native metric.
        binning_sample_rows (int): Rows sampled to compute bin edges.
        aggregate_batch_rows (int): How many rows to accumulate before running a
            grouping pass. Larger batches group more efficiently but hold more rows in
            memory; 8 million costs roughly a gigabyte for a typical dimension set.

    Returns:
        Cube: The aggregated cube.

    Raises:
        CubeError: If the sidecar directory holds no parts.
    """
    parts = _sidecar_parts(sidecar_dir)
    available = set(pq.ParquetFile(parts[0]).schema_arrow.names)

    used_labels = [c for c in label_dimensions if c in available]
    if score_columns is None:
        used_scores = sorted(c[len("native_") :] for c in available if c.startswith("native_"))
    else:
        used_scores = [c for c in score_columns if f"native_{c}" in available]

    binnings: dict[str, ScoreBinning] = {}
    for name in used_scores:
        edges = _quantile_edges(_sample_scores(parts, f"native_{name}", binning_sample_rows), n_score_bins)
        if edges is not None:
            binnings[name] = ScoreBinning(column=f"native_{name}", edges=edges)

    columns = used_labels + [f"native_{n}" for n in binnings] + ["est_tokens"]
    dimension_names = used_labels + [f"native_{n}" for n in binnings]
    schema = pa.schema(
        [pa.field(c, pa.large_string()) for c in used_labels]
        + [pa.field(f"native_{n}", pa.int16()) for n in binnings]
        + [pa.field("n_documents", pa.int64()), pa.field("n_tokens", pa.int64())]
    )

    # Grouping stays inside Arrow's C++ kernels rather than a Python loop over documents,
    # which is what makes this feasible over a whole blend.
    #
    # Row groups are batched before being aggregated. Aggregating each one alone barely
    # compresses -- at these cardinalities a row group of a million rows yields nearly a
    # million cells -- so the work would be done twice for no gain. Batching lets the
    # cardinality saturate first, which is the whole reason a cube is small.
    aggregates: list[pa.Table] = []
    pending: list[pa.Table] = []
    pending_rows = 0
    n_documents = 0
    n_tokens = 0

    def flush() -> None:
        nonlocal pending, pending_rows
        if pending:
            aggregates.append(_aggregate_cells(pa.concat_tables(pending), dimension_names))
            pending, pending_rows = [], 0

    for part in parts:
        parquet_file = pq.ParquetFile(part)
        for group_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_idx, columns=columns)
            if table.num_rows == 0:
                continue
            tokens = table.column("est_tokens").to_numpy(zero_copy_only=False).astype(np.int64)

            grouped: dict[str, Any] = {c: pc.fill_null(table.column(c), MISSING) for c in used_labels}
            for name, binning in binnings.items():
                bins = binning.bin_index(
                    table.column(f"native_{name}").to_numpy(zero_copy_only=False).astype(np.float64)
                )
                grouped[f"native_{name}"] = pa.array(bins, type=pa.int16())
            grouped["n_documents"] = pa.array(np.ones(table.num_rows, dtype=np.int64))
            grouped["n_tokens"] = pa.array(tokens)

            pending.append(pa.table(grouped))
            pending_rows += table.num_rows
            n_documents += table.num_rows
            n_tokens += int(tokens.sum())
            if pending_rows >= aggregate_batch_rows:
                flush()
    flush()

    if aggregates:
        table = _aggregate_cells(pa.concat_tables(aggregates), dimension_names).cast(schema)
    else:
        table = schema.empty_table()

    return Cube(
        dataset=dataset_name,
        label_dimensions=used_labels,
        score_binnings=binnings,
        table=table,
        n_documents=n_documents,
        n_tokens=n_tokens,
    )
