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
from modalities.dataloader.preprocessing.quality.materialize import (
    MaterializationError,
    materialize_dataset,
    materialize_dataset_buckets,
)
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
    predicate_breakdown,
)
from modalities.dataloader.preprocessing.quality.upsampling import UNANNOTATED_BUCKET, UpsamplingSpec
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


# ------------------------------------------------------- calibration read is bounded


def test_calibration_read_does_not_scale_with_file_count(tmp_path: Path):
    # The whole point: a dataset with many files must not cost many times more to
    # calibrate. Reading a fixed number of lines from every file made this stage read
    # 30 TB over the real blend.
    from modalities.dataloader.preprocessing.quality import tokens as tokens_module

    corpus_dir = tmp_path / "many_files"
    corpus_dir.mkdir()
    for f in range(200):
        with (corpus_dir / f"shard_{f:04d}.jsonl").open("w") as fh:
            for i in range(500):
                fh.write(json.dumps({"text": " ".join(["w"] * 20)}) + "\n")

    opened: list[Path] = []
    original_open = Path.open

    def counting_open(self, *args, **kwargs):
        opened.append(self)
        return original_open(self, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(Path, "open", counting_open)
    try:
        calibration = tokens_module.calibrate_dataset(
            dataset_name="many",
            file_paths=sorted(corpus_dir.glob("*.jsonl")),
            tokenizer=_WhitespaceTokenizer(),
            tokenizer_name="whitespace",
            sample_size=200,
            max_probe_files=32,
        )
    finally:
        monkey.undo()

    assert calibration.sampled_documents == 200
    jsonl_opened = [p for p in opened if p.suffix == ".jsonl"]
    assert len(jsonl_opened) <= 32, f"opened {len(jsonl_opened)} of 200 files; the read must be bounded"


def test_calibration_probes_files_across_the_whole_dataset(tmp_path: Path):
    # Spread matters: a prefix would calibrate on whatever the corpus happens to be
    # ordered by. Each file here has a distinct text length, so the sample reveals reach.
    corpus_dir = tmp_path / "spread"
    corpus_dir.mkdir()
    for f in range(100):
        with (corpus_dir / f"shard_{f:04d}.jsonl").open("w") as fh:
            for _ in range(20):
                fh.write(json.dumps({"text": " ".join(["w"] * (f + 1))}) + "\n")

    from modalities.dataloader.preprocessing.quality.tokens import _probe_files

    probed = _probe_files(sorted(corpus_dir.glob("*.jsonl")), max_probe_files=10)

    assert len(probed) == 10
    indices = [int(p.stem.split("_")[1]) for p in probed]
    assert indices[0] < 10 and indices[-1] > 80, f"probe files clustered: {indices}"


def test_calibration_is_reproducible_for_a_given_seed(tmp_path: Path, corpus: Path):
    from modalities.dataloader.preprocessing.quality.tokens import calibrate_dataset as calibrate

    kwargs = dict(
        dataset_name="toy",
        file_paths=sorted(corpus.glob("*.jsonl")),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=50,
    )
    first = calibrate(**kwargs, seed=7)
    second = calibrate(**kwargs, seed=7)

    assert first.bytes_per_token == second.bytes_per_token
    assert first.sampled_tokens == second.sampled_tokens


def test_calibration_is_written_after_each_dataset(tmp_path: Path, corpus: Path):
    # Interrupting a long calibration must not throw away what it already measured.
    from modalities.dataloader.preprocessing.quality import pipeline
    from modalities.dataloader.preprocessing.quality.tokens import CalibrationSet

    other = tmp_path / "other_corpus"
    other.mkdir()
    (other / "a.jsonl").write_text(json.dumps({"text": "one two three"}) + "\n")
    registry = CorpusRegistry(
        datasets=[
            DatasetEntry(name="first", jsonl_root=corpus, glob="*.jsonl"),
            DatasetEntry(name="second", jsonl_root=other, glob="*.jsonl"),
        ]
    )

    work_dir = tmp_path / "work"
    seen_after_first: list[list[str]] = []
    original = CalibrationSet.to_yaml

    def recording_to_yaml(self, path):
        original(self, path)
        seen_after_first.append(sorted(self.calibrations))

    monkey = pytest.MonkeyPatch()
    monkey.setattr(CalibrationSet, "to_yaml", recording_to_yaml)
    try:
        pipeline.calibrate_blend(
            registry=registry,
            work_dir=work_dir,
            tokenizer=_WhitespaceTokenizer(),
            tokenizer_name="whitespace",
            sample_size=20,
        )
    finally:
        monkey.undo()

    assert seen_after_first == [["first"], ["first", "second"]], seen_after_first


# ------------------------------------------------------- bucket writer memory bound


def test_bucket_writer_memory_is_bounded_by_total_not_per_bucket(tmp_path: Path):
    # The regression that OOM-killed all 64 tasks of a real run: the flush threshold was
    # per bucket, so with many buckets no single bucket ever reached it and the whole
    # input stayed in memory. The bound must be on the total held across buckets.
    from modalities.dataloader.preprocessing.quality.annotation_join import _BucketWriter

    schema = pa.schema([pa.field("key", pa.large_string()), pa.field("label", pa.large_string())])
    writer = _BucketWriter(tmp_path / "buckets", schema, n_buckets=1024, max_buffered_rows=1000)
    try:
        # Spread 20,000 rows over 1024 buckets: ~20 per bucket, far below any sane
        # per-bucket threshold, so a per-bucket rule would never flush.
        for i in range(20_000):
            writer.add(i % 1024, {"key": f"k{i}", "label": "x"})
            assert writer._buffered_rows < 1000 + 1, "total buffered rows exceeded the cap"
    finally:
        writer.close()

    written = sorted((tmp_path / "buckets").glob("*.parquet"))
    assert written, "nothing was written"
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in written)
    assert total == 20_000, f"rows lost or duplicated across flushes: {total}"


def test_bucket_writer_survives_repeated_flushes_of_the_same_bucket(tmp_path: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import _BucketWriter

    schema = pa.schema([pa.field("key", pa.large_string())])
    writer = _BucketWriter(tmp_path / "b", schema, n_buckets=2, max_buffered_rows=10)
    try:
        for i in range(100):
            writer.add(0, {"key": f"k{i}"})
    finally:
        writer.close()

    files = list((tmp_path / "b").glob("*.parquet"))
    assert len(files) == 1, "one bucket must stay one file across flushes"
    assert pq.ParquetFile(files[0]).metadata.num_rows == 100


def test_bucketing_refuses_to_mix_runs_with_different_array_sizes(tmp_path: Path, annotations: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import AnnotationJoinError

    shards = sorted(annotations.glob("*.parquet"))
    out = tmp_path / "mixed_buckets"
    bucket_annotations(
        shard_paths=shards,
        out_dir=out,
        n_buckets=4,
        label_columns=["educational_value"],
        shard_id=0,
        num_shards=4,
        show_progress=False,
    )

    with pytest.raises(AnnotationJoinError, match="num_shards"):
        bucket_annotations(
            shard_paths=shards,
            out_dir=out,
            n_buckets=4,
            label_columns=["educational_value"],
            shard_id=0,
            num_shards=8,
            show_progress=False,
        )


# --------------------------------------------- bucketing metadata write/read race


def test_a_truncated_metadata_file_does_not_stop_a_run(tmp_path: Path, annotations: Path):
    # The regression: the guard read every _meta file with a bare json.loads, so a task
    # that caught a sibling's file mid-write died. 12 of 64 tasks of a real run crashed
    # this way.
    out = tmp_path / "buckets_truncated"
    out.mkdir()
    (out / "_meta.0005.json").write_text("")

    n_rows, _ = bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=out,
        n_buckets=4,
        label_columns=["educational_value"],
        shard_id=0,
        num_shards=1,
        show_progress=False,
    )

    assert n_rows == 150


def test_read_bucket_metadata_skips_an_unreadable_file(tmp_path: Path, annotations: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import AnnotationJoinError, read_bucket_metadata

    out = tmp_path / "buckets_partial_meta"
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=out,
        n_buckets=4,
        label_columns=["educational_value"],
        shard_id=0,
        num_shards=2,
        show_progress=False,
    )
    # Shard 1's metadata exists but is corrupt: the run must read as incomplete rather
    # than raising a decode error or, worse, joining without shard 1's annotations.
    (out / "_meta.0001.json").write_text("{ truncated")

    with pytest.raises(AnnotationJoinError, match="incomplete"):
        read_bucket_metadata(out)


def test_metadata_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path, annotations: Path):
    from modalities.dataloader.preprocessing.quality.annotation_join import read_bucket_metadata

    out = tmp_path / "buckets_atomic"
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=out,
        n_buckets=4,
        label_columns=["educational_value"],
        shard_id=0,
        num_shards=1,
        show_progress=False,
    )

    assert not list(out.glob("*.tmp")), "an atomic write must not leave its temporary file behind"
    # A stray .tmp must be ignored rather than parsed as metadata.
    (out / "_meta.0009.json.tmp").write_text("{ half written")
    meta = read_bucket_metadata(out)
    assert meta["n_rows"] == 150


def test_guard_survives_metadata_being_rewritten_concurrently(tmp_path: Path, annotations: Path):
    # Exercises the actual race: one thread rewriting metadata while bucketing tasks keep
    # entering the directory and running the guard.
    import threading

    from modalities.dataloader.preprocessing.quality.annotation_join import _write_metadata

    out = tmp_path / "buckets_concurrent"
    out.mkdir()
    payload = {"n_buckets": 4, "label_columns": ["educational_value"], "n_rows": 1, "shard_id": 7, "num_shards": 4}

    stop = threading.Event()

    def rewrite() -> None:
        while not stop.is_set():
            _write_metadata(out / "_meta.0007.json", payload)

    writer = threading.Thread(target=rewrite, daemon=True)
    writer.start()
    try:
        for _ in range(20):
            # num_shards matches the payload, so the guard must pass rather than raise --
            # and must not blow up on a file being replaced underneath it.
            bucket_annotations(
                shard_paths=sorted(annotations.glob("*.parquet")),
                out_dir=out,
                n_buckets=4,
                label_columns=["educational_value"],
                shard_id=0,
                num_shards=4,
                show_progress=False,
            )
    finally:
        stop.set()
        writer.join(timeout=5)


# ------------------------------------------------- join batching (read amplification)


def _sidecar_with_many_parts(tmp_path: Path, corpus: Path, suffix: str) -> tuple[Path, DatasetEntry]:
    """A sidecar with one part per file, so batching has something to batch."""
    entry = DatasetEntry(
        name="toy",
        jsonl_root=corpus,
        glob="*.jsonl",
        annotation_split="toy",
        key=KeySpec(kind=KeyKind.FIELD, field="id"),
    )
    calibration = TokenCalibration(dataset="toy", tokenizer="w", bytes_per_token=4.0)
    out = tmp_path / f"sidecar_{suffix}"
    SidecarBuilder(entry, calibration, index_root=tmp_path / f"idx_{suffix}").build(out, show_progress=False)
    return out, entry


def test_join_result_is_independent_of_the_batch_size(tmp_path: Path, corpus: Path, annotations: Path):
    buckets = tmp_path / "b"
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=buckets,
        n_buckets=8,
        label_columns=["educational_value"],
        show_progress=False,
    )

    results = {}
    for label, batch in (("one_part_at_a_time", 1), ("all_at_once", 10_000_000)):
        sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, label)
        report = join_annotations(sidecar, buckets, "toy", "toy", max_batch_keys=batch, show_progress=False)
        labels = []
        for part in sorted(sidecar.glob("part-*.parquet")):
            labels.extend(pq.read_table(part).column("educational_value").to_pylist())
        results[label] = (report.n_documents, report.n_matched, labels)

    small, large = results["one_part_at_a_time"], results["all_at_once"]
    assert small[0] == large[0] == 200
    assert small[1] == large[1] == 150
    assert small[2] == large[2], "batching must not change which label each document gets"


def test_join_reads_each_bucket_once_per_batch_not_once_per_part(
    tmp_path: Path, corpus: Path, annotations: Path, monkeypatch
):
    # The regression this guards: reading the bucketed split once per sidecar part meant
    # 454 TB of reads over the real blend, because every part's documents hash across
    # every bucket.
    buckets = tmp_path / "b2"
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=buckets,
        n_buckets=8,
        label_columns=["educational_value"],
        show_progress=False,
    )
    sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, "counted")
    n_parts = len(list(sidecar.glob("part-*.parquet")))
    assert n_parts == 2

    import pyarrow.parquet as pq_module

    reads: list[str] = []
    original = pq_module.read_table

    def counting_read_table(source, *args, **kwargs):
        reads.append(str(source))
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pq_module, "read_table", counting_read_table)
    join_annotations(sidecar, buckets, "toy", "toy", max_batch_keys=10_000_000, show_progress=False)

    bucket_reads = [r for r in reads if "bucket-" in r]
    # One batch covers both parts, so each populated bucket is read once -- not twice.
    assert len(bucket_reads) == len(set(bucket_reads)), f"a bucket was read more than once: {bucket_reads}"
    assert len(bucket_reads) <= 8


def test_join_counts_a_duplicate_key_once_not_once_per_part(tmp_path: Path, corpus: Path):
    # Duplicates used to be counted afresh on every part, inflating the figure by the part
    # count. Only duplicates among keys the join actually wants should be reported.
    rows = {"id": ["doc-0-0", "doc-0-0", "doc-0-1"], "educational_value": ["high", "basic", "high"]}
    ann_dir = tmp_path / "dup_ann"
    ann_dir.mkdir()
    pq.write_table(pa.table(rows), ann_dir / "a.parquet")

    buckets = tmp_path / "dup_buckets"
    bucket_annotations(
        shard_paths=[ann_dir / "a.parquet"],
        out_dir=buckets,
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )
    sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, "dup")

    report = join_annotations(sidecar, buckets, "toy", "toy", max_batch_keys=1, show_progress=False)

    assert report.n_duplicate_keys == 1, f"expected the one duplicate counted once, got {report.n_duplicate_keys}"


# ------------------------------------------------- resume, and cube consistency checks


def _buckets_with_label(tmp_path: Path, corpus: Path, name: str, level: str) -> Path:
    """Buckets giving every document of `corpus` the same educational_value."""
    ids = []
    for shard in sorted(corpus.glob("*.jsonl")):
        with shard.open() as f:
            ids.extend(json.loads(line)["id"] for line in f)
    ann = tmp_path / f"ann_{name}"
    ann.mkdir()
    pq.write_table(pa.table({"id": ids, "educational_value": [level] * len(ids)}), ann / "a.parquet")
    out = tmp_path / f"buckets_{name}"
    bucket_annotations(
        shard_paths=[ann / "a.parquet"],
        out_dir=out,
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )
    return out


def _labels_of(sidecar: Path) -> dict[str, list]:
    return {
        p.name: pq.read_table(p).column("educational_value").to_pylist() for p in sorted(sidecar.glob("part-*.parquet"))
    }


def test_resume_skips_already_labelled_parts_and_keeps_their_values(tmp_path: Path, corpus: Path):
    first = _buckets_with_label(tmp_path, corpus, "first", "high")
    second = _buckets_with_label(tmp_path, corpus, "second", "none")
    sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, "resume")

    join_annotations(sidecar, first, "toy", "toy", show_progress=False)
    before = _labels_of(sidecar)
    assert all(v == "high" for values in before.values() for v in values)

    # Drop the labels from one part, so a resumed run has exactly one part to do.
    parts = sorted(sidecar.glob("part-*.parquet"))
    stripped = parts[0]
    table = pq.read_table(stripped)
    pq.write_table(table.drop_columns(["educational_value"]), stripped)

    report = join_annotations(sidecar, second, "toy", "toy", resume=True, show_progress=False)
    after = _labels_of(sidecar)

    assert all(v == "none" for v in after[stripped.name]), "the unlabelled part must be joined"
    for other in parts[1:]:
        assert after[other.name] == before[other.name], "an already-labelled part must be left alone"
    # The report describes the whole sidecar, not just the part this run redid: a resumed
    # join that counted only its own work reported 0% coverage, which reads as a failure.
    assert report.n_parts_resumed == len(parts) - 1
    assert report.n_documents == sum(len(v) for v in after.values())
    assert report.n_matched == report.n_documents


def test_resume_off_redoes_every_part(tmp_path: Path, corpus: Path):
    first = _buckets_with_label(tmp_path, corpus, "f2", "high")
    second = _buckets_with_label(tmp_path, corpus, "s2", "none")
    sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, "noresume")

    join_annotations(sidecar, first, "toy", "toy", show_progress=False)
    join_annotations(sidecar, second, "toy", "toy", resume=False, show_progress=False)

    after = _labels_of(sidecar)
    assert all(v == "none" for values in after.values() for v in values), "without resume the labels must be replaced"


def test_build_cube_rejects_a_partly_joined_sidecar(tmp_path: Path, corpus: Path):
    from modalities.dataloader.preprocessing.quality.cube import CubeError

    buckets = _buckets_with_label(tmp_path, corpus, "partial", "high")
    sidecar, _ = _sidecar_with_many_parts(tmp_path, corpus, "partial")
    join_annotations(sidecar, buckets, "toy", "toy", show_progress=False)

    # Simulate the interrupted join: one part never got its labels.
    victim = sorted(sidecar.glob("part-*.parquet"))[0]
    pq.write_table(pq.read_table(victim).drop_columns(["educational_value"]), victim)

    with pytest.raises(CubeError, match="sidecar parts are missing") as excinfo:
        build_cube(sidecar, "toy")
    message = str(excinfo.value)
    assert "toy" in message and "educational_value" in message
    assert "--resume" in message, "the error should say how to finish the join"


def test_build_cubes_builds_healthy_datasets_before_raising(tmp_path: Path, corpus: Path):
    from modalities.dataloader.preprocessing.quality import pipeline
    from modalities.dataloader.preprocessing.quality.cube import CubeError

    work = tmp_path / "work"
    healthy_sidecar = pipeline.sidecar_dir(work, "healthy")
    broken_sidecar = pipeline.sidecar_dir(work, "broken")

    buckets = _buckets_with_label(tmp_path, corpus, "cubes", "high")
    for name, target in (("healthy", healthy_sidecar), ("broken", broken_sidecar)):
        built, _ = _sidecar_with_many_parts(tmp_path, corpus, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for p in sorted(built.glob("part-*.parquet")):
            pq.write_table(pq.read_table(p), target / p.name)
        join_annotations(target, buckets, name, "toy", show_progress=False)
    victim = sorted(broken_sidecar.glob("part-*.parquet"))[0]
    pq.write_table(pq.read_table(victim).drop_columns(["educational_value"]), victim)

    registry = CorpusRegistry(
        datasets=[
            DatasetEntry(name="healthy", jsonl_root=corpus, glob="*.jsonl"),
            DatasetEntry(name="broken", jsonl_root=corpus, glob="*.jsonl"),
        ]
    )

    with pytest.raises(CubeError):
        pipeline.build_cubes(registry, work)

    # The healthy dataset's cube must exist despite the other one failing.
    assert pipeline.cube_path(work, "healthy").is_file()
    assert not pipeline.cube_path(work, "broken").is_file()


def test_resumed_join_reports_the_sidecars_real_coverage(
    tmp_path: Path, dataset_entry: DatasetEntry, annotations: Path
):
    """A resumed join must describe the sidecar, not just the work it happened to redo.

    The first version counted only the parts it re-joined, so a run that skipped
    everything wrote a report saying 0 documents and 0% coverage. That reads as a failed
    join, and on the real blend it overwrote genuine coverage figures with zeros.
    """
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=dataset_entry.iter_files(),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=50,
    )
    sidecar_dir = tmp_path / "sidecar"
    SidecarBuilder(dataset_entry, calibration, index_root=tmp_path / "idx").build(sidecar_dir, show_progress=False)
    buckets = tmp_path / "buckets"
    bucket_annotations(
        shard_paths=sorted(annotations.glob("*.parquet")),
        out_dir=buckets,
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )

    first = join_annotations(sidecar_dir, buckets, "toy", "toy", show_progress=False)
    assert first.n_documents == 200
    assert first.n_matched == 150
    assert first.n_parts_resumed == 0

    resumed = join_annotations(sidecar_dir, buckets, "toy", "toy", resume=True, show_progress=False)
    assert resumed.n_parts_resumed == 2
    assert resumed.n_documents == first.n_documents
    assert resumed.n_matched == first.n_matched
    assert resumed.coverage == first.coverage



# --------------------------------------------------------------- quality-aware upsampling


def _curve_selection(target_ratio: float = 1.0, discard: float = 0.0) -> DatasetSelection:
    return DatasetSelection(
        name="toy",
        upsampling=UpsamplingSpec(
            quality_field="educational_value",
            target_ratio=target_ratio,
            max_factor=4.0,
            discard_below_percentile=discard,
        ),
    )


def test_a_curve_over_a_cube_hits_its_target_and_rises_with_quality(built_sidecar: Path, tmp_path: Path):
    cube = build_cube(built_sidecar, "toy", label_dimensions=["educational_value"])
    result = evaluate_on_cube(cube, _curve_selection(target_ratio=1.5), MissingPolicy.KEEP)

    assert result.plan is not None
    assert result.effective_tokens == pytest.approx(result.plan.tokens_available * 1.5, rel=1e-6)
    factors = [b.factor for b in result.plan.buckets]
    assert factors == sorted(factors)
    # The unannotated fifty of two hundred documents cannot be ordered, so they form the
    # bottom bucket rather than being silently dropped or silently kept at full weight.
    assert result.plan.buckets[0].bucket.label == UNANNOTATED_BUCKET


def test_a_curve_and_a_flat_ratio_cannot_both_be_given():
    with pytest.raises(ValueError, match="already determines how much is drawn"):
        DatasetSelection(
            name="toy",
            ratio=2.0,
            upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=1.0),
        )


def test_a_curve_needs_an_ordinal_quality_field():
    with pytest.raises(ValueError, match="needs an ordinal quality_field"):
        DatasetSelection(name="toy", upsampling=UpsamplingSpec(quality_field="score", target_ratio=1.0))


def test_materializing_a_curve_writes_one_index_tree_per_bucket(
    built_sidecar: Path, dataset_entry: DatasetEntry, tmp_path: Path
):
    selection = _curve_selection(target_ratio=1.0, discard=20.0)
    results = materialize_dataset_buckets(
        sidecar_dir=built_sidecar,
        dataset_entry=dataset_entry,
        dataset_selection=selection,
        missing_policy=MissingPolicy.KEEP,
        output_dir=tmp_path / "mix",
        show_progress=False,
    )

    assert results, "a curve must produce at least one bucket"
    # Each row is its own dataset for packing, but still points back at the registry entry.
    assert {r.source_dataset for r in results} == {"toy"}
    assert all(r.name.startswith("toy__") for r in results)
    assert all(r.ratio > 0 for r in results)
    # Factors rise with quality, and the whole point is that they differ.
    assert len({round(r.ratio, 6) for r in results}) > 1

    # No document may appear in two buckets: they are disjoint by construction, and an
    # overlap would silently duplicate documents on top of the intended repetition.
    seen: set[tuple[str, int, int]] = set()
    for result in results:
        for source, index_path in result.index_files.items():
            entries = pickle.loads(Path(index_path).read_bytes())
            for offset, length in entries:
                key = (source, offset, length)
                assert key not in seen, f"document {key} appears in more than one quality bucket"
                seen.add(key)


def test_materializing_a_curve_refuses_a_sidecar_without_the_quality_column(
    tmp_path: Path, dataset_entry: DatasetEntry, annotations: Path
):
    """The curve orders by a joined label, so an unjoined sidecar cannot support one."""
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=dataset_entry.iter_files(),
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_name="whitespace",
        sample_size=50,
    )
    sidecar_dir = tmp_path / "unjoined"
    SidecarBuilder(dataset_entry, calibration, index_root=tmp_path / "idx2").build(
        sidecar_dir, show_progress=False
    )
    with pytest.raises(MaterializationError, match="no column 'educational_value'"):
        materialize_dataset_buckets(
            sidecar_dir=sidecar_dir,
            dataset_entry=dataset_entry,
            dataset_selection=_curve_selection(),
            missing_policy=MissingPolicy.KEEP,
            output_dir=tmp_path / "mix2",
            show_progress=False,
        )


# ------------------------------------------------------------ per-predicate attribution


def test_attribution_finds_the_predicate_that_does_nothing(built_sidecar: Path):
    """A predicate matching everything the others already keep is the interesting case: it
    changes no numbers while making the selection harder to read."""
    cube = build_cube(built_sidecar, "toy", label_dimensions=["educational_value"])
    selection = DatasetSelection(
        name="toy",
        predicates=[
            Predicate(field="educational_value", op=Op.AT_LEAST, value="basic"),
            # Every level is at least "none", so this one cannot remove anything.
            Predicate(field="educational_value", op=Op.AT_LEAST, value="none"),
        ],
    )
    breakdown = predicate_breakdown(cube, selection, MissingPolicy.KEEP)

    binding, redundant = breakdown.contributions
    assert binding.marginal_tokens > 0
    assert redundant.marginal_tokens == 0
    assert "no effect given the others" in breakdown.describe()


def test_attribution_totals_agree_with_the_blend_evaluation(built_sidecar: Path):
    cube = build_cube(built_sidecar, "toy", label_dimensions=["educational_value"])
    selection = DatasetSelection(
        name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")]
    )
    breakdown = predicate_breakdown(cube, selection, MissingPolicy.KEEP)
    result = evaluate_on_cube(cube, selection, MissingPolicy.KEEP)

    assert breakdown.kept_tokens == result.tokens_kept
    assert breakdown.total_tokens == result.tokens_total


def test_the_overlap_diagonal_is_the_predicate_itself(built_sidecar: Path):
    """Not the product with itself: an interpolated predicate has fractional factors, and
    squaring them undercounts. On real data that read 8.19 M where it matched 9.75 M."""
    cube = build_cube(built_sidecar, "toy")
    selection = DatasetSelection(
        name="toy",
        predicates=[
            Predicate(field="score", op=Op.GTE, value=2.5),
            Predicate(field="educational_value", op=Op.AT_LEAST, value="basic"),
        ],
    )
    breakdown = predicate_breakdown(cube, selection, MissingPolicy.KEEP)
    for i, contribution in enumerate(breakdown.contributions):
        assert breakdown.overlap_tokens[i][i] == contribution.matched_tokens
    # And the matrix is symmetric.
    assert breakdown.overlap_tokens[0][1] == breakdown.overlap_tokens[1][0]


def test_attribution_reports_which_predicate_was_interpolated(built_sidecar: Path):
    cube = build_cube(built_sidecar, "toy")
    selection = DatasetSelection(
        name="toy",
        predicates=[
            Predicate(field="score", op=Op.GTE, value=2.5),
            Predicate(field="educational_value", op=Op.AT_LEAST, value="basic"),
        ],
    )
    breakdown = predicate_breakdown(cube, selection, MissingPolicy.KEEP)
    numeric, ordinal = breakdown.contributions
    # A threshold inside a quantile bin is interpolated; an ordinal level never is.
    assert not numeric.exact
    assert ordinal.exact


# --------------------------------------------------------------------------- reruns
#
# Both stages below write into a directory that a later stage globs for work. That makes
# leftovers dangerous rather than merely untidy: a stale index tree is packed, and a stale
# .pbin is trained on. These tests rerun each stage and check the directory afterwards
# describes only the current selection.


@pytest.fixture
def blend_inputs(tmp_path: Path, dataset_entry: DatasetEntry, built_sidecar: Path):
    """A registry and a sidecar root laid out the way `materialize_blend` expects."""
    import shutil

    sidecar_root = tmp_path / "sidecar_root"
    shutil.copytree(built_sidecar, sidecar_root / "toy")
    return CorpusRegistry(datasets=[dataset_entry]), sidecar_root


def _blend_config(**overrides):
    from modalities.dataloader.preprocessing.quality.selection import SelectionConfig

    settings = dict(datasets=[DatasetSelection(name="toy", ratio=2.0)])
    settings.update(overrides)
    return SelectionConfig(**settings)


def _overexposed_config():
    """A selection whose repetition cap the run would blow through.

    Its predicate also selects a different, much smaller document set than
    `_blend_config`. Without that the rejected run would rewrite byte-identical indexes
    and corruption would be undetectable by comparing the directory.
    """
    return _blend_config(
        datasets=[
            DatasetSelection(
                name="toy",
                ratio=2.0,
                predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="high")],
            )
        ],
        target_tokens=1e12,
        max_total_exposure=1.0,
    )


def test_a_rejected_apply_leaves_the_previous_blend_exactly_as_it_was(tmp_path: Path, blend_inputs):
    from modalities.dataloader.preprocessing.quality.materialize import materialize_blend

    registry, sidecar_root = blend_inputs
    output_root = tmp_path / "blend"

    manifest_path = materialize_blend(
        config=_blend_config(),
        registry=registry,
        sidecar_root=sidecar_root,
        output_root=output_root,
        show_progress=False,
    )
    before = {p.relative_to(output_root): p.read_bytes() for p in sorted(output_root.rglob("*")) if p.is_file()}
    assert before, "the first apply must have written something to compare against"

    # The exposure guard fires only after every index has been written, so this is the
    # case where a destination-in-place apply would leave new indexes beside an old
    # manifest: a directory that still looks complete but no longer agrees with itself.
    with pytest.raises(MaterializationError, match="past its declared cap"):
        materialize_blend(
            config=_overexposed_config(),
            registry=registry,
            sidecar_root=sidecar_root,
            output_root=output_root,
            show_progress=False,
        )

    after = {p.relative_to(output_root): p.read_bytes() for p in sorted(output_root.rglob("*")) if p.is_file()}
    assert after == before, "a rejected apply must not touch the blend that is already published"
    assert manifest_path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".blend.")]
    assert leftovers == [], f"staging directories must be cleaned up, found {leftovers}"


def test_a_rejected_first_apply_publishes_nothing_at_all(tmp_path: Path, blend_inputs):
    from modalities.dataloader.preprocessing.quality.materialize import materialize_blend

    registry, sidecar_root = blend_inputs
    output_root = tmp_path / "blend"

    with pytest.raises(MaterializationError):
        materialize_blend(
            config=_overexposed_config(),
            registry=registry,
            sidecar_root=sidecar_root,
            output_root=output_root,
            show_progress=False,
        )

    assert not output_root.exists(), "a failed apply must not leave a half-built blend behind"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".blend.")] == []


def test_a_successful_apply_replaces_the_previous_blend_rather_than_merging_into_it(tmp_path: Path, blend_inputs):
    from modalities.dataloader.preprocessing.quality.materialize import materialize_blend

    registry, sidecar_root = blend_inputs
    output_root = tmp_path / "blend"
    arguments = dict(registry=registry, sidecar_root=sidecar_root, output_root=output_root, show_progress=False)

    materialize_blend(config=_blend_config(), **arguments)
    stale = output_root / "dropped-dataset" / "shard_0.idx"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from an earlier selection")

    materialize_blend(config=_blend_config(datasets=[DatasetSelection(name="toy", ratio=3.0)]), **arguments)

    assert not stale.exists(), "index trees the new selection does not name must not survive the rerun"
    assert (output_root / "toy").is_dir()


def test_the_published_manifest_names_index_files_that_actually_exist(tmp_path: Path, blend_inputs):
    # The indexes are written under a staging directory that is renamed away on
    # publication. If the manifest kept the paths the writers physically used, every path
    # in it would point inside a directory that no longer exists, and every packing config
    # built from it would name a missing index.
    import yaml

    from modalities.dataloader.preprocessing.quality.materialize import materialize_blend

    registry, sidecar_root = blend_inputs
    output_root = tmp_path / "blend"

    manifest_path = materialize_blend(
        config=_blend_config(),
        registry=registry,
        sidecar_root=sidecar_root,
        output_root=output_root,
        show_progress=False,
    )
    manifest = yaml.safe_load(manifest_path.read_text())

    indexes = [Path(index) for dataset in manifest["datasets"] for index in dataset["index_files"].values()]
    assert indexes, "the manifest must name at least one index"
    missing = [str(index) for index in indexes if not index.exists()]
    assert missing == [], f"the manifest names index files that are not on disk: {missing[:3]}"
    for index in indexes:
        assert output_root in index.parents, f"{index} is not under the published blend"


def test_a_curved_blend_also_publishes_usable_index_paths(tmp_path: Path, blend_inputs):
    # Curves take a different code path, writing one index tree per quality bucket, so the
    # rebasing has to hold there too.
    import yaml

    from modalities.dataloader.preprocessing.quality.materialize import materialize_blend

    registry, sidecar_root = blend_inputs
    output_root = tmp_path / "blend"

    manifest_path = materialize_blend(
        config=_blend_config(
            datasets=[
                DatasetSelection(
                    name="toy",
                    upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=1.5),
                )
            ]
        ),
        registry=registry,
        sidecar_root=sidecar_root,
        output_root=output_root,
        show_progress=False,
    )
    manifest = yaml.safe_load(manifest_path.read_text())

    assert len(manifest["datasets"]) > 1, "a curve must split the dataset into per-bucket rows"
    for dataset in manifest["datasets"]:
        for index in dataset["index_files"].values():
            assert Path(index).exists(), f"{dataset['name']} names a missing index {index}"


def _packing_inputs(tmp_path: Path, dataset_entry: DatasetEntry, names: list[str]) -> tuple[Path, Path, Path]:
    """A manifest naming `names`, plus a registry, a template and real index files."""
    import yaml

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "toy",
                        "jsonl_root": str(dataset_entry.jsonl_root),
                        "glob": "*.jsonl",
                        "annotation_split": "toy",
                        "key": {"kind": "field", "field": "id"},
                    }
                ]
            }
        )
    )
    template_path = tmp_path / "template.yaml"
    template_path.write_text(
        yaml.safe_dump({"settings": {"jq_pattern": ".text"}, "tokenizer": {"config": {"name": "whitespace"}}})
    )

    for name in names:
        index_path = tmp_path / f"{name}_0.idx"
        if not index_path.exists():
            index_path.write_bytes(pickle.dumps([(0, 10), (10, 10)]))

    manifest_path = tmp_path / f"manifest_{'_'.join(names)}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": name,
                        "source_dataset": "toy",
                        "index_files": {
                            str(dataset_entry.jsonl_root / "shard_0.jsonl"): str(tmp_path / f"{name}_0.idx")
                        },
                    }
                    for name in names
                ]
            }
        )
    )
    return manifest_path, registry_path, template_path


def _fake_pack(config_path: Path) -> Path:
    """Stands in for the packing stage: writes an output and records its fingerprint.

    Mirrors `pack_many.py`, which writes the marker only after `run()` returns.
    """
    destination = config_path.with_suffix(".pbin")
    destination.write_bytes(b"packed")
    destination.with_name(destination.name + ".fingerprint").write_text(
        config_path.with_suffix(".fingerprint").read_text()
    )
    return destination


def test_rerunning_packing_configs_removes_the_jobs_the_new_manifest_dropped(tmp_path: Path, dataset_entry):
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    wide, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy__high", "toy__low"])
    narrow, _, _ = _packing_inputs(tmp_path, dataset_entry, ["toy__high"])
    arguments = dict(registry_path=registry_path, template_path=template_path, output_dir=output_dir)

    written = quality_pipeline.write_packing_configs(manifest_path=wide, **arguments)
    assert len(written) == 2
    for config_path in written:
        _fake_pack(config_path)

    quality_pipeline.write_packing_configs(manifest_path=narrow, **arguments)

    assert (output_dir / "toy__high" / "shard_0.yaml").exists()
    assert (output_dir / "toy__high" / "shard_0.pbin").exists(), "an unchanged dataset must not be repacked"
    assert not (output_dir / "toy__low").exists(), "the dropped dataset's config and .pbin must both be gone"
    assert sorted(p.name for p in output_dir.rglob("*.pbin")) == ["shard_0.pbin"]


def test_a_changed_selection_discards_the_output_packed_from_the_old_index(tmp_path: Path, dataset_entry):
    # The dangerous case, and the reason a name-based check is not enough: the index is
    # rewritten under the same path, so the .pbin beside it keeps its name while holding
    # the previous selection's documents. Packing skips already-present outputs, so
    # leaving it would train on tokens no current predicate chose.
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy"])
    arguments = dict(manifest_path=manifest_path, registry_path=registry_path, template_path=template_path)

    written = quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)
    destination = _fake_pack(written[0])
    assert destination.exists()

    # A changed predicate keeps the index path and changes its contents.
    (tmp_path / "toy_0.idx").write_bytes(pickle.dumps([(0, 10)]))
    quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)

    assert not destination.exists(), "the output packed from the superseded index must be deleted"
    assert not destination.with_name(destination.name + ".fingerprint").exists()
    assert written[0].exists(), "the config itself is still current and must be rewritten, not removed"


def test_an_unchanged_selection_keeps_its_packed_output(tmp_path: Path, dataset_entry):
    # The other half of the contract: fingerprinting must not force a full repack of a
    # blend that has not changed, which on the real corpus is hours of tokenisation.
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy"])
    arguments = dict(manifest_path=manifest_path, registry_path=registry_path, template_path=template_path)

    written = quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)
    destination = _fake_pack(written[0])
    stamp = destination.stat().st_mtime_ns

    quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)

    assert destination.exists() and destination.stat().st_mtime_ns == stamp


def test_an_output_with_no_fingerprint_record_is_not_trusted(tmp_path: Path, dataset_entry):
    # Outputs packed before fingerprinting existed, and outputs from a pack that died
    # before writing its marker. Neither can be shown to match the current index.
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy"])
    arguments = dict(manifest_path=manifest_path, registry_path=registry_path, template_path=template_path)

    written = quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)
    unmarked = written[0].with_suffix(".pbin")
    unmarked.write_bytes(b"packed by an older run")

    quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)

    assert not unmarked.exists()


def test_packing_configs_refuse_a_manifest_whose_indexes_are_gone(tmp_path: Path, dataset_entry):
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy"])
    (tmp_path / "toy_0.idx").unlink()

    with pytest.raises(MaterializationError, match="not on disk"):
        quality_pipeline.write_packing_configs(
            manifest_path=manifest_path,
            registry_path=registry_path,
            template_path=template_path,
            output_dir=tmp_path / "packcfg",
        )


def test_packing_config_pruning_can_be_turned_off(tmp_path: Path, dataset_entry):
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    wide, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy__high", "toy__low"])
    narrow, _, _ = _packing_inputs(tmp_path, dataset_entry, ["toy__high"])
    arguments = dict(registry_path=registry_path, template_path=template_path, output_dir=output_dir)

    quality_pipeline.write_packing_configs(manifest_path=wide, **arguments)
    quality_pipeline.write_packing_configs(manifest_path=narrow, prune=False, **arguments)

    assert (output_dir / "toy__low" / "shard_0.yaml").exists(), "--no_prune must leave the old jobs in place"


def test_pruning_leaves_files_it_does_not_own_alone(tmp_path: Path, dataset_entry):
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy__high"])
    output_dir.mkdir()
    note = output_dir / "NOTES.md"
    note.write_text("why this blend exists")

    quality_pipeline.write_packing_configs(
        manifest_path=manifest_path,
        registry_path=registry_path,
        template_path=template_path,
        output_dir=output_dir,
    )

    assert note.exists(), "pruning is limited to .yaml, .pbin and .fingerprint, so anything else survives"


def test_adopt_existing_accepts_unfingerprinted_output_but_only_that(tmp_path: Path, dataset_entry):
    # The migration path for the blend already on disk, which was packed before
    # fingerprints existed. It must adopt an unmarked output and still refuse one whose
    # record positively disagrees.
    from modalities.dataloader.preprocessing.quality import pipeline as quality_pipeline

    output_dir = tmp_path / "packcfg"
    manifest_path, registry_path, template_path = _packing_inputs(tmp_path, dataset_entry, ["toy"])
    arguments = dict(manifest_path=manifest_path, registry_path=registry_path, template_path=template_path)

    written = quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)
    legacy = written[0].with_suffix(".pbin")
    legacy.write_bytes(b"packed before fingerprinting")

    quality_pipeline.write_packing_configs(output_dir=output_dir, adopt_existing=True, **arguments)
    assert legacy.exists(), "an unmarked output must be adoptable rather than repacked"

    # Adopted, so a later run without the flag leaves it alone.
    quality_pipeline.write_packing_configs(output_dir=output_dir, **arguments)
    assert legacy.exists()

    # A record that actively disagrees is never adopted, flag or not.
    legacy.with_name(legacy.name + ".fingerprint").write_text("from-another-selection")
    quality_pipeline.write_packing_configs(output_dir=output_dir, adopt_existing=True, **arguments)
    assert not legacy.exists(), "--adopt_existing must not override a fingerprint that disagrees"
