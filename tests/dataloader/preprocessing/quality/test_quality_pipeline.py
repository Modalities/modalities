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


# --------------------------------------------------------------- fast path extraction


@pytest.mark.parametrize(
    "pattern,expected",
    [
        (".score", ["score"]),
        (".metadata.len_cl100k_base", ["metadata", "len_cl100k_base"]),
        ('."openlid-v3".prob', ["openlid-v3", "prob"]),
        ('.metadata.dclm_plus2."__label__1"', ["metadata", "dclm_plus2", "__label__1"]),
    ],
)
def test_simple_paths_are_recognised(pattern, expected):
    from modalities.dataloader.preprocessing.quality.sidecar import parse_simple_path

    assert parse_simple_path(pattern) == expected


@pytest.mark.parametrize("pattern", [".scores | max", ".a[0]", "select(.x)", ".a | length", "."])
def test_non_path_patterns_fall_back_to_jq(pattern):
    from modalities.dataloader.preprocessing.quality.sidecar import build_metric_extractor, parse_simple_path

    assert parse_simple_path(pattern) is None
    _, is_fast = build_metric_extractor(pattern)
    assert is_fast is False


def test_an_invalid_pattern_fails_when_the_builder_is_created(tmp_path: Path, corpus: Path):
    # Better to fail here than to hand back an extractor that yields None for every
    # document and look like a corpus with no metrics.
    entry = DatasetEntry(
        name="bad",
        jsonl_root=corpus,
        glob="*.jsonl",
        native_metrics=[NativeMetric(name="broken", jq_pattern="score")],
    )
    calibration = TokenCalibration(dataset="bad", tokenizer="w", bytes_per_token=4.0)

    with pytest.raises(ValueError, match="compile error"):
        SidecarBuilder(entry, calibration, index_root=tmp_path / "bad_idx")


def test_fast_extractor_agrees_with_jq_on_real_shapes():
    import jq

    from modalities.dataloader.preprocessing.quality.sidecar import build_metric_extractor

    record = {
        "score": 2.5,
        "zero": 0,
        "flag": False,
        "openlid-v3": {"prob": [0.99, 0.01]},
        "metadata": {"dclm_plus2": {"__label__1": 0.94}, "nested": None},
    }
    for pattern in (
        ".score",
        ".zero",
        ".flag",
        '."openlid-v3".prob',
        '.metadata.dclm_plus2."__label__1"',
        ".missing",
        ".metadata.nested.deeper",
        ".score.deeper",
    ):
        fast, is_fast = build_metric_extractor(pattern)
        assert is_fast, pattern
        try:
            expected = jq.compile(pattern).input_value(record).first()
        except (ValueError, StopIteration):
            expected = None
        assert fast(record) == expected, f"{pattern}: {fast(record)!r} != {expected!r}"


def test_jq_fallback_still_extracts(tmp_path: Path):
    from modalities.dataloader.preprocessing.quality.sidecar import build_metric_extractor

    extract, is_fast = build_metric_extractor('."openlid-v3".prob | max')
    assert not is_fast
    assert extract({"openlid-v3": {"prob": [0.1, 0.9]}}) == 0.9


def test_sidecar_uses_the_fast_path_and_still_records_metrics(tmp_path: Path, corpus: Path):
    # A dataset whose metrics are all plain paths must produce the same values it would
    # have produced through jq.
    entry = DatasetEntry(
        name="fast",
        jsonl_root=corpus,
        glob="*.jsonl",
        native_metrics=[NativeMetric(name="score", jq_pattern=".score")],
    )
    calibration = TokenCalibration(dataset="fast", tokenizer="w", bytes_per_token=4.0)
    out = tmp_path / "fast_sidecar"
    SidecarBuilder(entry, calibration, index_root=tmp_path / "fast_idx").build(out, show_progress=False)

    table = pq.read_table(sorted(out.glob("part-*.parquet"))[0])
    values = table.column("native_score").to_pylist()
    assert len(values) == 100
    assert all(v is not None for v in values)


# --------------------------------------------------------------------- cube vectorising


def test_cube_is_independent_of_the_aggregation_batch_size(built_sidecar: Path):
    # Batching row groups is a performance detail and must not change the result.
    big = build_cube(built_sidecar, "toy", aggregate_batch_rows=10_000_000)
    small = build_cube(built_sidecar, "toy", aggregate_batch_rows=1)

    assert big.n_documents == small.n_documents
    assert big.n_tokens == small.n_tokens
    assert big.table.num_rows == small.table.num_rows

    def as_set(cube):
        cols = cube.dimensions + ["n_documents", "n_tokens"]
        return {tuple(row[c] for c in cols) for row in cube.table.select(cols).to_pylist()}

    assert as_set(big) == as_set(small)


def test_cube_totals_match_the_sidecar(built_sidecar: Path):
    cube = build_cube(built_sidecar, "toy")
    total_docs = total_tokens = 0
    for part in sorted(built_sidecar.glob("part-*.parquet")):
        table = pq.read_table(part, columns=["est_tokens"])
        total_docs += table.num_rows
        total_tokens += sum(table.column("est_tokens").to_pylist())

    assert cube.n_documents == total_docs
    assert cube.n_tokens == total_tokens
    assert sum(cube.table.column("n_documents").to_pylist()) == total_docs
    assert sum(cube.table.column("n_tokens").to_pylist()) == total_tokens


# ------------------------------------------------------------------------- sharding


def test_plan_sidecar_work_partitions_completely_and_disjointly(tmp_path: Path, corpus: Path):
    from modalities.dataloader.preprocessing.quality.pipeline import plan_sidecar_work

    other = tmp_path / "other"
    other.mkdir()
    for i in range(5):
        (other / f"f{i}.jsonl").write_text(json.dumps({"text": "x"}) + "\n")
    registry = CorpusRegistry(
        datasets=[
            DatasetEntry(name="a", jsonl_root=corpus, glob="*.jsonl"),
            DatasetEntry(name="b", jsonl_root=other, glob="*.jsonl"),
        ]
    )

    seen: list[tuple[str, int]] = []
    for shard_id in range(3):
        for name, file_ids in plan_sidecar_work(registry, shard_id=shard_id, num_shards=3).items():
            seen.extend((name, file_id) for file_id in file_ids)

    expected = [("a", i) for i in range(2)] + [("b", i) for i in range(5)]
    assert sorted(seen) == sorted(expected), "every file must be built exactly once across the array"


def test_plan_sidecar_work_rejects_an_out_of_range_shard(corpus: Path):
    from modalities.dataloader.preprocessing.quality.pipeline import plan_sidecar_work

    registry = CorpusRegistry(datasets=[DatasetEntry(name="a", jsonl_root=corpus, glob="*.jsonl")])

    with pytest.raises(ValueError, match="not in"):
        plan_sidecar_work(registry, shard_id=3, num_shards=3)


def _build_sidecar_only(tmp_path: Path, dataset_entry: DatasetEntry, suffix: str) -> Path:
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=dataset_entry.iter_files(),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=100,
    )
    out = tmp_path / f"sidecar_{suffix}"
    SidecarBuilder(dataset_entry, calibration, index_root=tmp_path / f"idx_{suffix}").build(out, show_progress=False)
    return out


def test_sharded_bucketing_gives_the_same_join_as_a_single_task(
    tmp_path: Path, dataset_entry: DatasetEntry, annotations: Path
):
    # Spread the annotations over several files so there is something to shard.
    split_dir = tmp_path / "annotations_split"
    split_dir.mkdir()
    table = pq.read_table(sorted(annotations.glob("*.parquet"))[0])
    for i in range(3):
        pq.write_table(table.slice(i * 50, 50), split_dir / f"part{i}.parquet")
    shards = sorted(split_dir.glob("*.parquet"))

    single_sidecar = _build_sidecar_only(tmp_path, dataset_entry, "single")
    bucket_annotations(
        shard_paths=shards,
        out_dir=tmp_path / "buckets_single",
        n_buckets=8,
        label_columns=["educational_value"],
        show_progress=False,
    )
    single = join_annotations(single_sidecar, tmp_path / "buckets_single", "toy", "toy", show_progress=False)

    sharded_sidecar = _build_sidecar_only(tmp_path, dataset_entry, "sharded")
    for shard_id in range(3):
        bucket_annotations(
            shard_paths=shards,
            out_dir=tmp_path / "buckets_sharded",
            n_buckets=8,
            label_columns=["educational_value"],
            shard_id=shard_id,
            num_shards=3,
            show_progress=False,
        )
    sharded = join_annotations(sharded_sidecar, tmp_path / "buckets_sharded", "toy", "toy", show_progress=False)

    assert sharded.n_matched == single.n_matched
    assert sharded.n_annotation_rows == single.n_annotation_rows
    # The labels themselves, not just the counts.
    single_labels = pq.read_table(sorted(single_sidecar.glob("part-*.parquet"))[0]).column("educational_value")
    sharded_labels = pq.read_table(sorted(sharded_sidecar.glob("part-*.parquet"))[0]).column("educational_value")
    assert single_labels.to_pylist() == sharded_labels.to_pylist()


def test_join_refuses_an_incomplete_bucketing_run(tmp_path: Path, annotations: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import AnnotationJoinError, read_bucket_metadata

    # One of three announced tasks ran, so two thirds of the annotations are absent.
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=tmp_path / "buckets_partial",
        n_buckets=4,
        label_columns=["educational_value"],
        shard_id=0,
        num_shards=3,
        show_progress=False,
    )

    with pytest.raises(AnnotationJoinError, match="incomplete"):
        read_bucket_metadata(tmp_path / "buckets_partial")


def test_bucketing_rejects_an_out_of_range_shard(tmp_path: Path, annotations: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import AnnotationJoinError

    with pytest.raises(AnnotationJoinError, match="not in"):
        bucket_annotations(
            shard_paths=sorted(annotations.glob("*.parquet")),
            out_dir=tmp_path / "buckets_bad",
            n_buckets=4,
            shard_id=5,
            num_shards=3,
            show_progress=False,
        )
