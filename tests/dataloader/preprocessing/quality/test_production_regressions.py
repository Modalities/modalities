"""Regressions for failures that surfaced only on the full production run.

Each of these cost hours on a 20 TB blend and none of them had a test. They share a shape:
a stage fails, reports the failure somewhere quiet, and destroys or hides the evidence, so
the symptom appears much later and much further downstream.

  * pointer resolution wrote nulls over the pointers it could not resolve, making a retry
    impossible without rebuilding the sidecar from source;
  * the annotation scan held a fragment per bucket file, so a split with many small files
    exceeded 160 GiB while a split with fifty times the data did not.

A third belonged to the packing stage -- its skip-if-exists treated a truncated output as
finished. That stage is gone, but the lesson outlived it: `export.py` records a shard's line
and byte counts only after the file is in place, and treats a size mismatch as unfinished.
See `test_export.py::test_resume_rewrites_a_shard_that_never_finished`.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from modalities.dataloader.preprocessing.quality import annotation_join
from modalities.dataloader.preprocessing.quality.annotation_join import (
    bucket_annotations,
    join_annotations,
)
from modalities.dataloader.preprocessing.quality.registry import DatasetEntry, KeyKind, KeySpec
from modalities.dataloader.preprocessing.quality.sidecar import (
    SidecarBuilder,
    SidecarWriteError,
    resolve_source_pointers,
)
from modalities.dataloader.preprocessing.quality.tokens import TokenCalibration


EDUCATIONAL_LEVELS = ["none", "minimal", "basic", "moderate", "high"]


@pytest.fixture
def toy_corpus(tmp_path: Path) -> Path:
    """Two shards of documents. Local rather than shared, so this file stands alone."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for shard in range(2):
        with (corpus / f"shard_{shard}.jsonl").open("w") as f:
            for i in range(100):
                f.write(json.dumps({"id": f"doc-{shard}-{i}", "text": "word " * (10 + i)}) + "\n")
    return corpus


@pytest.fixture
def toy_entry(toy_corpus: Path) -> DatasetEntry:
    return DatasetEntry(
        name="toy",
        jsonl_root=toy_corpus,
        glob="*.jsonl",
        annotation_split="toy",
        key=KeySpec(kind=KeyKind.FIELD, field="id"),
    )


@pytest.fixture
def toy_annotations(tmp_path: Path, toy_corpus: Path) -> Path:
    """Annotations for the first 150 of the 200 documents."""
    import pyarrow as pa

    rows = {"id": [], "educational_value": []}
    for shard in range(2):
        with (toy_corpus / f"shard_{shard}.jsonl").open() as f:
            for line in f:
                if len(rows["id"]) >= 150:
                    break
                record = json.loads(line)
                rows["id"].append(record["id"])
                rows["educational_value"].append(EDUCATIONAL_LEVELS[len(rows["id"]) % 5])
    out = tmp_path / "annotations"
    out.mkdir()
    pq.write_table(pa.table(rows), out / "shard0.parquet")
    return out


def _pointer_corpus(tmp_path: Path, source_root: Path) -> DatasetEntry:
    """A translated corpus whose ids point into another corpus, as KletterMix does."""
    corpus = tmp_path / "translated"
    corpus.mkdir()
    with (corpus / "part.jsonl").open("w") as f:
        for i in range(20):
            f.write(json.dumps({"id": f"part_0.jsonl/{i}", "text": f"uebersetzt {i}"}) + "\n")
    return DatasetEntry(
        name="translated",
        jsonl_root=corpus,
        glob="*.jsonl",
        annotation_split="src",
        key=KeySpec(
            kind=KeyKind.SOURCE_POINTER, field="id", source_root=source_root, source_line_offset=0
        ),
    )


def test_pointer_resolution_refuses_to_overwrite_pointers_it_cannot_resolve(tmp_path: Path):
    """The write-back replaces the pointer with the resolved key, so writing nulls destroys
    the only copy. In production the source corpus had been moved and 225 M pointers were
    overwritten, logged as INFO, and only noticed two hours later as 0% join coverage."""
    missing = tmp_path / "not_where_it_used_to_be"
    missing.mkdir()
    entry = _pointer_corpus(tmp_path, missing)
    sidecar = tmp_path / "sidecar"
    SidecarBuilder(
        entry, TokenCalibration(dataset="translated", tokenizer="t", bytes_per_token=4.0),
        index_root=tmp_path / "idx",
    ).build(sidecar, show_progress=False)

    before = pq.read_table(sidecar / "part-000000.parquet").column("join_key").to_pylist()
    assert all(v is not None for v in before), "the builder should have stored the raw pointers"

    with pytest.raises(SidecarWriteError, match="Refusing to write"):
        resolve_source_pointers(sidecar, entry)

    after = pq.read_table(sidecar / "part-000000.parquet").column("join_key").to_pylist()
    assert after == before, "the pointers must survive a failed resolution so a retry is possible"


def test_pointer_resolution_still_writes_when_it_resolves(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    with (source / "part_0.jsonl").open("w") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"original {i}"}) + "\n")
    entry = _pointer_corpus(tmp_path, source)
    sidecar = tmp_path / "sidecar"
    SidecarBuilder(
        entry, TokenCalibration(dataset="translated", tokenizer="t", bytes_per_token=4.0),
        index_root=tmp_path / "idx",
    ).build(sidecar, show_progress=False)

    n = resolve_source_pointers(sidecar, entry)
    assert n == 20
    keys = pq.read_table(sidecar / "part-000000.parquet").column("join_key").to_pylist()
    # Pointers are replaced by content hashes of the source text.
    assert all(k is not None and len(k) == 64 for k in keys)


def test_the_join_is_unchanged_by_how_many_bucket_files_it_scans_at_once(
    tmp_path: Path, monkeypatch, toy_entry: DatasetEntry, toy_annotations: Path
):
    """Chunking the fragment list is what stopped HPLT exceeding 160 GiB -- a split with
    65,536 files and 10.7 M documents died where one with 16,384 files and 504 M did not, so
    the cost is per fragment. Chunking must not change what the join produces."""
    calibration = TokenCalibration(dataset="toy", tokenizer="t", bytes_per_token=4.0)
    buckets = tmp_path / "buckets"
    bucket_annotations(
        shard_paths=sorted(toy_annotations.glob("*.parquet")),
        out_dir=buckets,
        n_buckets=16,
        label_columns=["educational_value"],
        show_progress=False,
    )

    results = {}
    for group_size in (1, 3, 1024):
        sidecar = tmp_path / f"sidecar_{group_size}"
        SidecarBuilder(toy_entry, calibration, index_root=tmp_path / f"idx_{group_size}").build(
            sidecar, show_progress=False
        )
        monkeypatch.setattr(annotation_join, "SCAN_FILE_GROUP", group_size)
        report = join_annotations(sidecar, buckets, "toy", "toy", show_progress=False)
        labels = []
        for part in sorted(sidecar.glob("part-*.parquet")):
            labels.extend(pq.read_table(part, columns=["educational_value"]).column(0).to_pylist())
        results[group_size] = (report.n_matched, report.n_documents, labels)

    reference = results[1024]
    for group_size, outcome in results.items():
        assert outcome == reference, f"SCAN_FILE_GROUP={group_size} changed the join result"
    assert reference[0] == 150, "the fixture annotates 150 of 200 documents"

