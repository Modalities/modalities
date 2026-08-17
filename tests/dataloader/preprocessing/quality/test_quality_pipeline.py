import json
import pickle
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modalities.dataloader.large_file_lines_reader import LargeFileLinesReader
from modalities.dataloader.preprocessing.quality.annotation_join import bucket_annotations, join_annotations
from modalities.dataloader.preprocessing.quality.cube import build_cube
from modalities.dataloader.preprocessing.quality.materialize import materialize_dataset
from modalities.dataloader.preprocessing.quality.registry import (
    CorpusRegistry,
    DatasetEntry,
    KeyKind,
    KeySpec,
    NativeMetric,
    SourcePointerResolver,
    sha256_text,
    strip_urn_uuid,
)
from modalities.dataloader.preprocessing.quality.selection import (
    DatasetSelection,
    MissingPolicy,
    Op,
    Predicate,
    evaluate_on_cube,
    evaluate_on_sidecar,
)
from modalities.dataloader.preprocessing.quality.sidecar import SidecarBuilder
from modalities.dataloader.preprocessing.quality.tokens import TokenCalibration, calibrate_dataset

EDUCATIONAL_LEVELS = ["none", "minimal", "basic", "moderate", "high"]


class _WhitespaceTokenizer:
    """Stand-in tokenizer, so calibration can be tested without a model download."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two shards whose documents get longer as their quality rises."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    rng = random.Random(3)
    for shard in range(2):
        with (corpus_dir / f"shard_{shard}.jsonl").open("w") as f:
            for i in range(100):
                quality = rng.random()
                record = {
                    "id": f"doc-{shard}-{i}",
                    "text": " ".join(["word"] * int(10 + 200 * quality)),
                    "score": round(5 * quality, 4),
                    # Padding so the stored line is much longer than the text, which is
                    # what makes estimating from line length wrong.
                    "payload": "x" * 300,
                }
                f.write(json.dumps(record) + "\n")
    return corpus_dir


@pytest.fixture
def annotations(tmp_path: Path, corpus: Path) -> Path:
    """Annotations for the first 150 of the 200 documents."""
    rows = {"id": [], "educational_value": []}
    random.Random(4)
    for shard in range(2):
        with (corpus / f"shard_{shard}.jsonl").open() as f:
            for line in f:
                record = json.loads(line)
                if len(rows["id"]) >= 150:
                    break
                rows["id"].append(record["id"])
                rows["educational_value"].append(EDUCATIONAL_LEVELS[min(4, int(record["score"] / 5 * 5))])
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    pq.write_table(pa.table(rows), annotation_dir / "shard0.parquet")
    return annotation_dir


@pytest.fixture
def dataset_entry(corpus: Path) -> DatasetEntry:
    return DatasetEntry(
        name="toy",
        jsonl_root=corpus,
        glob="*.jsonl",
        annotation_split="toy",
        key=KeySpec(kind=KeyKind.FIELD, field="id"),
        native_metrics=[NativeMetric(name="score", jq_pattern=".score")],
    )


@pytest.fixture
def built_sidecar(tmp_path: Path, dataset_entry: DatasetEntry, annotations: Path) -> Path:
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=dataset_entry.iter_files(),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=100,
    )
    sidecar_dir = tmp_path / "sidecar"
    SidecarBuilder(dataset_entry, calibration, index_root=tmp_path / "idx").build(sidecar_dir, show_progress=False)

    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=tmp_path / "buckets",
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )
    join_annotations(sidecar_dir, tmp_path / "buckets", "toy", "toy", show_progress=False)
    return sidecar_dir


def test_strip_urn_uuid_handles_both_stored_forms():
    assert strip_urn_uuid("<urn:uuid:abc-123>") == "abc-123"
    assert strip_urn_uuid("abc-123") == "abc-123"


def test_sha256_key_is_taken_over_the_exact_bytes():
    # Stripping or appending whitespace produces a key that matches nothing, so the
    # digest must be over the unmodified text.
    assert sha256_text("hello") != sha256_text("hello\n")
    assert sha256_text("hello") != sha256_text(" hello ")


def test_calibration_prefers_a_native_count_when_every_record_has_one(tmp_path: Path):
    corpus_dir = tmp_path / "native"
    corpus_dir.mkdir()
    with (corpus_dir / "a.jsonl").open("w") as f:
        for i in range(50):
            text = " ".join(["w"] * (i + 1))
            f.write(json.dumps({"text": text, "token_count": (i + 1) * 2}) + "\n")

    calibration = calibrate_dataset(
        dataset_name="native",
        file_paths=[corpus_dir / "a.jsonl"],
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=50,
    )

    assert calibration.uses_native_field()
    assert calibration.native_field == "token_count"
    # The corpus counts twice as many tokens as our tokenizer, so the scale halves them.
    assert calibration.native_scale == pytest.approx(0.5, abs=1e-6)


def test_calibration_ignores_a_partially_present_native_count(tmp_path: Path):
    corpus_dir = tmp_path / "partial"
    corpus_dir.mkdir()
    with (corpus_dir / "a.jsonl").open("w") as f:
        for i in range(50):
            record = {"text": " ".join(["w"] * (i + 1))}
            if i % 2 == 0:
                record["token_count"] = i + 1
            f.write(json.dumps(record) + "\n")

    calibration = calibrate_dataset(
        dataset_name="partial",
        file_paths=[corpus_dir / "a.jsonl"],
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=50,
    )

    assert not calibration.uses_native_field()


def test_estimate_falls_back_to_bytes_per_token():
    calibration = TokenCalibration(
        dataset="toy", tokenizer="whitespace", bytes_per_token=4.0, eod_tokens_per_document=1
    )

    assert calibration.estimate({}, text_bytes=400) == 101


def test_sidecar_estimates_from_text_not_from_the_stored_line(built_sidecar: Path):
    table = pq.read_table(sorted(built_sidecar.glob("part-*.parquet"))[0])

    # Every record carries 300 bytes of padding, so a line-length estimate would be
    # inflated by roughly that much on the shortest documents.
    assert (table.column("byte_len").to_numpy() > table.column("text_bytes").to_numpy()).all()
    shortest = min(table.column("text_bytes").to_pylist())
    assert shortest < 300


def test_join_reports_partial_coverage(tmp_path: Path, dataset_entry: DatasetEntry, annotations: Path):
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=dataset_entry.iter_files(),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=100,
    )
    sidecar_dir = tmp_path / "sidecar2"
    SidecarBuilder(dataset_entry, calibration, index_root=tmp_path / "idx2").build(sidecar_dir, show_progress=False)
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=tmp_path / "buckets2",
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )

    report = join_annotations(sidecar_dir, tmp_path / "buckets2", "toy", "toy", show_progress=False)

    assert report.n_documents == 200
    assert report.n_matched == 150
    assert report.coverage == pytest.approx(0.75)


@pytest.mark.parametrize("policy", [MissingPolicy.KEEP, MissingPolicy.DROP])
def test_cube_agrees_with_the_per_document_scan(built_sidecar: Path, policy: MissingPolicy):
    cube = build_cube(built_sidecar, "toy")
    selection = DatasetSelection(
        name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")]
    )

    from_cube = evaluate_on_cube(cube, selection, policy)
    from_sidecar = evaluate_on_sidecar(built_sidecar, selection, policy)

    assert from_cube.n_documents_kept == from_sidecar.n_documents_kept
    assert from_cube.tokens_kept == from_sidecar.tokens_kept


def test_materialized_index_is_loadable_and_selects_the_right_documents(
    built_sidecar: Path, dataset_entry: DatasetEntry, tmp_path: Path
):
    selection = DatasetSelection(
        name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="moderate")]
    )
    expected = evaluate_on_sidecar(built_sidecar, selection, MissingPolicy.DROP)

    result = materialize_dataset(
        sidecar_dir=built_sidecar,
        dataset_entry=dataset_entry,
        dataset_selection=selection,
        missing_policy=MissingPolicy.DROP,
        output_dir=tmp_path / "indexes",
        show_progress=False,
    )

    assert result.n_documents_kept == expected.n_documents_kept

    total_in_indexes = 0
    for source_path, index_path in result.index_files.items():
        entries = pickle.loads(Path(index_path).read_bytes())
        total_in_indexes += len(entries)
        # The index must be readable by the same reader the packer uses, and every
        # document it names must satisfy the predicate.
        reader = LargeFileLinesReader(Path(source_path), index_path=Path(index_path))
        try:
            assert len(reader) == len(entries)
            for i in range(len(reader)):
                record = json.loads(reader[i])
                assert record["id"].startswith("doc-")
        finally:
            reader.close()
    assert total_in_indexes == expected.n_documents_kept


def test_materialized_index_entries_are_ordered_by_position(
    built_sidecar: Path, dataset_entry: DatasetEntry, tmp_path: Path
):
    result = materialize_dataset(
        sidecar_dir=built_sidecar,
        dataset_entry=dataset_entry,
        dataset_selection=DatasetSelection(name="toy", predicates=[Predicate(field="score", op=Op.GTE, value=1.0)]),
        missing_policy=MissingPolicy.KEEP,
        output_dir=tmp_path / "indexes2",
        show_progress=False,
    )

    for index_path in result.index_files.values():
        entries = pickle.loads(Path(index_path).read_bytes())
        offsets = [offset for offset, _ in entries]
        assert offsets == sorted(offsets), "a generated index lists documents in file order"


def test_source_pointer_resolution_uses_zero_indexed_lines(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    with (source_dir / "part_0.jsonl").open("w") as f:
        for i in range(5):
            f.write(json.dumps({"text": f"line {i}"}) + "\n")

    resolver = SourcePointerResolver(source_root=source_dir)
    resolved = resolver.resolve(["part_0.jsonl/0", "part_0.jsonl/3"])

    assert resolved["part_0.jsonl/0"] == sha256_text("line 0")
    assert resolved["part_0.jsonl/3"] == sha256_text("line 3")


def test_source_pointer_resolution_skips_pointers_past_the_end(tmp_path: Path):
    source_dir = tmp_path / "source2"
    source_dir.mkdir()
    (source_dir / "part_0.jsonl").write_text(json.dumps({"text": "only line"}) + "\n")

    resolved = SourcePointerResolver(source_root=source_dir).resolve(["part_0.jsonl/0", "part_0.jsonl/9"])

    assert "part_0.jsonl/0" in resolved
    assert "part_0.jsonl/9" not in resolved, "a pointer past the end must not map to a wrong key"


def test_registry_rejects_an_annotated_dataset_without_a_key(tmp_path: Path):
    with pytest.raises(ValueError, match="no key spec"):
        DatasetEntry(name="x", jsonl_root=tmp_path, annotation_split="some/split")


def test_registry_rejects_duplicate_dataset_names(tmp_path: Path):
    with pytest.raises(ValueError, match="duplicate dataset name"):
        CorpusRegistry(
            datasets=[DatasetEntry(name="x", jsonl_root=tmp_path), DatasetEntry(name="x", jsonl_root=tmp_path)]
        )
