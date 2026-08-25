"""Regressions for the three failures that surfaced only on the full production run.

Each of these cost hours on a 20 TB blend and none of them had a test. They share a shape:
a stage fails, reports the failure somewhere quiet, and destroys or hides the evidence, so
the symptom appears much later and much further downstream.

  * pointer resolution wrote nulls over the pointers it could not resolve, making a retry
    impossible without rebuilding the sidecar from source;
  * the annotation scan held a fragment per bucket file, so a split with many small files
    exceeded 160 GiB while a split with fifty times the data did not;
  * the packer's skip-if-exists treated a truncated output as finished.
"""

import importlib.util
import json
import pickle
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


def _load_pack_many():
    """Loads the packing driver, which lives under config_files rather than in the package."""
    path = (
        Path(__file__).resolve().parents[4]
        / "config_files/data_preparation/quality/slurm/pack_many.py"
    )
    spec = importlib.util.spec_from_file_location("pack_many", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skip_existing_rejects_a_truncated_packed_file(tmp_path: Path):
    """Existence is not health. One .pbin in 54,738 was 151 MB on disk reporting data_len=0;
    skipping on existence left it in the blend, and the dataset would have trained 567 M
    tokens short had the final check counted files instead of reading headers."""
    pack_many = _load_pack_many()

    absent = tmp_path / "never_written.pbin"
    assert not pack_many._is_usable(absent, None)

    # A header claiming an empty data section, which is what an interrupted write leaves.
    truncated = tmp_path / "truncated.pbin"
    truncated.write_bytes((0).to_bytes(8, "little") + (4).to_bytes(4, "little") + b"\x00" * 4096)
    assert not pack_many._is_usable(truncated, None), "a zero-length data section is not a finished pack"

    garbage = tmp_path / "garbage.pbin"
    garbage.write_bytes(b"not a pbin")
    assert not pack_many._is_usable(garbage, None)


def test_a_healthy_packed_file_is_skipped(tmp_path: Path):
    pack_many = _load_pack_many()
    healthy = tmp_path / "healthy.pbin"
    payload = b"\x01\x00\x00\x00" * 32
    index = pickle.dumps([(0, len(payload))])
    healthy.write_bytes(
        len(payload).to_bytes(8, "little") + (4).to_bytes(4, "little") + payload + index
    )
    assert pack_many._is_usable(healthy, None), "with no fingerprint to check, a healthy header is enough"


def test_skip_existing_rejects_an_output_packed_from_a_different_selection(tmp_path: Path):
    """Health is not currency. A changed predicate rewrites the index under the same path,
    so the .pbin beside it stays structurally perfect while holding the documents the
    previous selection chose. Skipping it would train on tokens no current predicate picked."""
    pack_many = _load_pack_many()
    healthy = tmp_path / "healthy.pbin"
    payload = b"\x01\x00\x00\x00" * 32
    index = pickle.dumps([(0, len(payload))])
    healthy.write_bytes(len(payload).to_bytes(8, "little") + (4).to_bytes(4, "little") + payload + index)
    marker = healthy.with_name(healthy.name + ".fingerprint")

    assert not pack_many._is_usable(healthy, "abc123"), "no record means it cannot be shown to be current"

    marker.write_text("stale-digest")
    assert not pack_many._is_usable(healthy, "abc123"), "a mismatched record must force a repack"

    marker.write_text("abc123\n")
    assert pack_many._is_usable(healthy, "abc123"), "a matching record must still be skipped"


def _healthy_pbin_bytes() -> bytes:
    payload = b"\x01\x00\x00\x00" * 32
    return len(payload).to_bytes(8, "little") + (4).to_bytes(4, "little") + payload + pickle.dumps(
        [(0, len(payload))]
    )


def test_a_failed_repack_does_not_leave_a_marker_vouching_for_the_wreckage(tmp_path: Path):
    """The dangerous interleaving: an output and a matching record exist, the health check
    rejects the output, the rebuild is interrupted after writing a non-zero header, and the
    old record still agrees. The next run would then accept a half-written file, because
    the header reads and the fingerprint matches."""
    pack_many = _load_pack_many()

    destination = tmp_path / "shard_0.pbin"
    destination.write_bytes(b"damaged")
    marker = pack_many._marker_for(destination)
    marker.write_text("abc123")

    def interrupted(target: Path) -> None:
        target.write_bytes(_healthy_pbin_bytes())  # a plausible-looking partial write
        raise KeyboardInterrupt("killed by the scheduler")

    with pytest.raises(KeyboardInterrupt):
        pack_many._pack_to(destination, "abc123", interrupted)

    assert not marker.exists(), "the record must be torn up before the attempt, not after it"
    assert not pack_many._is_usable(destination, "abc123"), "the wreckage must not be skippable"
    assert list(tmp_path.glob("*.partial")) == [], "a failed attempt must not leave its scratch file"


def test_a_damaged_output_can_actually_be_replaced(tmp_path: Path):
    """PackedDataGenerator.run refuses a destination that already exists, so packing
    straight to it meant a damaged .pbin raised 'file already exists' on every retry and
    could never be rebuilt. Packing into a scratch file and moving it in fixes that."""
    pack_many = _load_pack_many()

    destination = tmp_path / "shard_0.pbin"
    destination.write_bytes(b"damaged")
    pack_many._marker_for(destination).write_text("stale")

    def pack(target: Path) -> None:
        if target.exists():
            raise ValueError(f"file already exists at destination path '{target}'.")
        target.write_bytes(_healthy_pbin_bytes())

    pack_many._pack_to(destination, "abc123", pack)

    assert pack_many._is_usable(destination, "abc123")
    assert pack_many._marker_for(destination).read_text().strip() == "abc123"
    assert list(tmp_path.glob("*.partial")) == []
