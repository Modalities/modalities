"""Exporting a selection as JSONL, with the sampling materialised in the bytes.

The property under test throughout is that the output is the training set: what the packer
used to leave to `WeightedCombinedDataset` -- repeating a dataset 3x, or drawing 60% of it --
now has to be visible as lines on disk, because a concatenation carries no weights.
"""

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from modalities.dataloader.preprocessing.quality.annotation_join import bucket_annotations, join_annotations

from modalities.dataloader.preprocessing.quality.export import (
    EXPORT_MANIFEST,
    ExportError,
    copies_for,
    export_blend,
    finalize_export,
)
from modalities.dataloader.preprocessing.quality.materialize import materialize_blend
from modalities.dataloader.preprocessing.quality.registry import (
    CorpusRegistry,
    DatasetEntry,
    KeyKind,
    KeySpec,
)
from modalities.dataloader.preprocessing.quality.selection import DatasetSelection, SelectionConfig
from modalities.dataloader.preprocessing.quality.sidecar import SidecarBuilder
from modalities.dataloader.preprocessing.quality.tokens import calibrate_dataset
from modalities.dataloader.preprocessing.quality.upsampling import UpsamplingSpec

LEVELS = ["none", "minimal", "basic", "moderate", "high"]
N_DOCS = 400


class _Whitespace:
    def tokenize(self, text):
        return text.split()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two shards, the last line of the second deliberately lacking a trailing newline."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    for shard in range(2):
        lines = []
        for i in range(N_DOCS // 2):
            doc = shard * (N_DOCS // 2) + i
            lines.append(
                json.dumps(
                    {
                        "id": f"doc-{doc}",
                        "text": " ".join(["word"] * (5 + doc % 17)),
                        "educational_value": LEVELS[doc % len(LEVELS)],
                    }
                )
            )
        body = "\n".join(lines)
        # No trailing newline on the second shard: a real corpus contains both forms, and the
        # index's byte_len excludes the terminator either way.
        (directory / f"shard_{shard}.jsonl").write_text(body + ("\n" if shard == 0 else ""))
    return directory


@pytest.fixture
def entry(corpus: Path) -> DatasetEntry:
    return DatasetEntry(
        name="toy",
        jsonl_root=corpus,
        glob="*.jsonl",
        annotation_split="toy",
        key=KeySpec(kind=KeyKind.FIELD, field="id"),
    )


@pytest.fixture
def blend(tmp_path: Path, entry: DatasetEntry, corpus: Path):
    """Builds a real sidecar, annotations and all, so curves can be exercised too."""
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=entry.iter_files(),
        tokenizer=_Whitespace(),
        tokenizer_name="whitespace",
        sample_size=100,
    )
    sidecar_root = tmp_path / "sidecar"
    SidecarBuilder(entry, calibration, index_root=tmp_path / "idx" / "toy").build(
        sidecar_root / "toy", show_progress=False
    )

    # Quality labels reach the sidecar through the join, not from the source records, so a
    # curve needs the annotation stages to have run.
    rows = {"id": [], "educational_value": []}
    for shard in sorted(corpus.glob("*.jsonl")):
        for line in shard.read_text().splitlines():
            record = json.loads(line)
            rows["id"].append(record["id"])
            rows["educational_value"].append(record["educational_value"])
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    pq.write_table(pa.table(rows), annotations / "shard0.parquet")
    bucket_annotations(
        shard_paths=[annotations / "shard0.parquet"],
        out_dir=tmp_path / "buckets",
        n_buckets=4,
        label_columns=["educational_value"],
        show_progress=False,
    )
    join_annotations(sidecar_root / "toy", tmp_path / "buckets", "toy", "toy", show_progress=False)

    def build(selection: DatasetSelection, name: str = "mix"):
        return materialize_blend(
            config=SelectionConfig(datasets=[selection]),
            registry=CorpusRegistry(datasets=[entry]),
            sidecar_root=sidecar_root,
            output_root=tmp_path / name,
            show_progress=False,
        )

    return build


def _export(tmp_path: Path, entry: DatasetEntry, manifest_path: Path, out: str = "out", **kwargs):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"datasets": [json.loads(CorpusRegistry(datasets=[entry]).model_dump_json())["datasets"][0]]})
    )
    exports = export_blend(
        manifest_path=manifest_path,
        registry_path=registry_path,
        output_root=tmp_path / out,
        show_progress=False,
        **kwargs,
    )
    finalize_export(tmp_path / out)
    return exports, tmp_path / out


def _lines(root: Path, dataset: str = "toy") -> list[str]:
    """Every output line of a dataset, in shard order -- the concatenation."""
    out: list[str] = []
    for shard in sorted((root / dataset).rglob("*.jsonl")):
        out.extend(shard.read_text().splitlines())
    return out


# --------------------------------------------------------------------------- the factor


@pytest.mark.parametrize("factor,expected", [(1.0, 1), (2.0, 2), (3.0, 3), (0.0, 0)])
def test_a_whole_factor_writes_exactly_that_many_copies(factor, expected):
    for offset in range(0, 5000, 137):
        assert copies_for(factor, 42, "/data/x.jsonl", offset) == expected


def test_a_fractional_factor_splits_between_the_two_neighbouring_counts():
    counts = [copies_for(1.2, 42, "/data/x.jsonl", o) for o in range(0, 200_000, 10)]
    assert set(counts) == {1, 2}, "1.2 must mean one copy or two, never three and never none"
    assert sum(counts) / len(counts) == pytest.approx(1.2, abs=0.02)


def test_a_factor_below_one_drops_documents_rather_than_truncating_them():
    counts = [copies_for(0.6, 42, "/data/x.jsonl", o) for o in range(0, 200_000, 10)]
    assert set(counts) == {0, 1}
    assert sum(counts) / len(counts) == pytest.approx(0.6, abs=0.02)


def test_the_choice_is_reproducible_and_seed_dependent():
    a = [copies_for(1.5, 42, "/data/x.jsonl", o) for o in range(2000)]
    assert a == [copies_for(1.5, 42, "/data/x.jsonl", o) for o in range(2000)]
    assert a != [copies_for(1.5, 43, "/data/x.jsonl", o) for o in range(2000)]


def test_the_choice_is_stable_across_processes():
    # A per-process hash seed would redraw which documents were doubled on every run, so a
    # resumed export would disagree with the shards it had already written.
    script = (
        "from modalities.dataloader.preprocessing.quality.export import copies_for;"
        "print([copies_for(1.5, 42, '/data/x.jsonl', o) for o in range(20)])"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert eval(out.stdout.strip()) == [copies_for(1.5, 42, "/data/x.jsonl", o) for o in range(20)]


def test_the_same_document_in_different_files_is_decided_separately():
    a = [copies_for(1.5, 42, "/data/a.jsonl", o) for o in range(500)]
    b = [copies_for(1.5, 42, "/data/b.jsonl", o) for o in range(500)]
    assert a != b


# --------------------------------------------------------------------------- the output


def test_every_document_is_written_once_at_ratio_one(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy"))
    exports, root = _export(tmp_path, entry, manifest)

    lines = _lines(root)
    assert len(lines) == N_DOCS
    assert exports[0].n_lines == N_DOCS
    assert [json.loads(line)["id"] for line in lines] == [f"doc-{i}" for i in range(N_DOCS)]


def test_output_lines_are_byte_identical_to_the_source(tmp_path: Path, entry: DatasetEntry, blend, corpus: Path):
    # Including the last line of the shard that has no trailing newline, which the export
    # must terminate itself without altering the record.
    manifest = blend(DatasetSelection(name="toy"))
    _, root = _export(tmp_path, entry, manifest)

    source = []
    for shard in sorted(corpus.glob("*.jsonl")):
        source.extend(shard.read_text().splitlines())
    assert _lines(root) == source

    for shard in sorted((root / "toy").rglob("*.jsonl")):
        assert shard.read_bytes().endswith(b"\n"), "every shard must be newline-terminated"


def test_an_upsampled_dataset_repeats_every_document(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=3.0))
    exports, root = _export(tmp_path, entry, manifest)

    lines = _lines(root)
    assert len(lines) == 3 * N_DOCS
    ids = [json.loads(line)["id"] for line in lines]
    assert all(ids.count(f"doc-{i}") == 3 for i in range(N_DOCS))
    # Copies are adjacent, as chosen: a document's three copies sit together.
    assert ids[:3] == ["doc-0"] * 3
    assert exports[0].n_documents == N_DOCS and exports[0].n_lines == 3 * N_DOCS


def test_a_downsampled_dataset_writes_fewer_whole_documents(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=0.5))
    _, root = _export(tmp_path, entry, manifest)

    lines = _lines(root)
    assert 0.35 * N_DOCS < len(lines) < 0.65 * N_DOCS
    for line in lines:
        json.loads(line)  # whole records, never a truncated one
    assert len(set(json.loads(line)["id"] for line in lines)) == len(lines), "no duplicates below 1.0"


def test_a_fractional_upsample_writes_singles_and_doubles(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=1.5))
    _, root = _export(tmp_path, entry, manifest)

    ids = [json.loads(line)["id"] for line in _lines(root)]
    counts = {i: ids.count(f"doc-{i}") for i in range(N_DOCS)}
    assert set(counts.values()) == {1, 2}
    assert sum(counts.values()) == pytest.approx(1.5 * N_DOCS, rel=0.1)


def test_a_curve_puts_every_bucket_in_one_directory(tmp_path: Path, entry: DatasetEntry, blend):
    # materialize_dataset_buckets emits one row per quality level, each with its own factor.
    # They describe one input dataset, so they must produce one output directory -- and a
    # source file drawn from by several buckets must yield one shard fed by all of them.
    manifest = blend(
        DatasetSelection(
            name="toy", upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=2.0)
        )
    )
    rows = yaml.safe_load(Path(manifest).read_text())["datasets"]
    assert len(rows) > 1, "the fixture must actually produce several buckets"

    exports, root = _export(tmp_path, entry, manifest)

    assert [d.name for d in exports] == ["toy"]
    assert sorted(p.name for p in (root / "toy").iterdir() if p.is_dir()) == []
    assert len(sorted((root / "toy").rglob("*.jsonl"))) == 2, "one shard per source file, not per bucket"
    # Documents come out in source order rather than grouped by quality bucket, which is what
    # writing bucket by bucket would have produced. The lowest bucket is discarded by the
    # curve, so this checks the order of whatever survived, not of every document.
    doc_numbers = [int(json.loads(line)["id"].removeprefix("doc-")) for line in _lines(root)]
    assert doc_numbers == sorted(doc_numbers), "output must follow source order"
    assert len(set(doc_numbers)) > 1, "the fixture must keep documents from more than one bucket"


def test_a_curves_buckets_each_get_their_own_factor(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(
        DatasetSelection(
            name="toy", upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=2.0)
        )
    )
    rows = {r["name"]: r for r in yaml.safe_load(Path(manifest).read_text())["datasets"]}
    _, root = _export(tmp_path, entry, manifest)

    ids = [json.loads(line)["id"] for line in _lines(root)]
    # The highest-factor bucket must be repeated more than the lowest.
    ranked = sorted(rows.values(), key=lambda r: r["ratio"])
    low, high = ranked[0], ranked[-1]
    assert high["ratio"] > low["ratio"], "the fixture needs a spread of factors to be meaningful"

    def mean_copies(row):
        levels = row["name"].rsplit("__", 1)[-1]
        docs = [i for i in range(N_DOCS) if LEVELS[i % len(LEVELS)] == levels]
        return sum(ids.count(f"doc-{i}") for i in docs) / len(docs) if docs else None

    assert mean_copies(high) > mean_copies(low)


# --------------------------------------------------------------------------- resume


def test_resume_leaves_a_complete_shard_alone(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=2.0))
    _, root = _export(tmp_path, entry, manifest)
    shard = sorted((root / "toy").rglob("*.jsonl"))[0]
    stamp = shard.stat().st_mtime_ns

    exports, _ = _export(tmp_path, entry, manifest)

    assert shard.stat().st_mtime_ns == stamp
    assert all(s.skipped for e in exports for s in e.shards)
    assert exports[0].n_lines == 2 * N_DOCS, "a skipped shard still reports what it holds"


def test_resume_rewrites_a_shard_that_never_finished(tmp_path: Path, entry: DatasetEntry, blend):
    # Existence is not completion. A killed job leaves a plausible-looking partial file, and
    # trusting it would silently shrink the training set.
    manifest = blend(DatasetSelection(name="toy"))
    _, root = _export(tmp_path, entry, manifest)
    shard = sorted((root / "toy").rglob("*.jsonl"))[0]
    shard.write_text(shard.read_text()[: len(shard.read_text()) // 2])

    exports, _ = _export(tmp_path, entry, manifest)

    assert not exports[0].shards[0].skipped
    assert len(_lines(root)) == N_DOCS


def test_no_resume_redoes_everything(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy"))
    _export(tmp_path, entry, manifest)
    exports, _ = _export(tmp_path, entry, manifest, resume=False)
    assert not any(s.skipped for e in exports for s in e.shards)


def test_a_failed_shard_leaves_no_partial_file(tmp_path: Path, entry: DatasetEntry, blend, corpus: Path):
    manifest = blend(DatasetSelection(name="toy"))
    # Truncate a source file so its index names bytes the file cannot supply.
    target = corpus / "shard_0.jsonl"
    target.write_bytes(target.read_bytes()[:200])

    with pytest.raises(ExportError, match="source has changed"):
        _export(tmp_path, entry, manifest)
    assert list((tmp_path / "out").rglob("*.partial")) == []


# --------------------------------------------------------------------------- the manifest


def test_the_manifest_says_the_ratio_is_already_applied(tmp_path: Path, entry: DatasetEntry, blend):
    # The footgun this closes: mix_manifest.yaml still says ratio 3.0, and carrying that into
    # a weighted_combined config after the repetition is in the bytes trains it nine times.
    manifest = blend(DatasetSelection(name="toy", ratio=3.0))
    _, root = _export(tmp_path, entry, manifest)

    exported = yaml.safe_load((root / EXPORT_MANIFEST).read_text())
    assert exported["repeat_factor_applied"] is True
    assert exported["training_ratio"] == 1.0
    assert "do not apply" in exported["note"].lower()

    dataset = exported["datasets"][0]
    assert dataset["ratio"] == 1.0 and dataset["repeat_factor_applied"] is True
    assert dataset["factors_applied"] == {"toy": 3.0}, "what was applied is still recorded"


def test_the_manifest_line_count_matches_the_files(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=2.0))
    _, root = _export(tmp_path, entry, manifest)

    exported = yaml.safe_load((root / EXPORT_MANIFEST).read_text())
    assert exported["n_lines"] == len(_lines(root)) == 2 * N_DOCS
    assert exported["n_bytes"] == sum(p.stat().st_size for p in (root / "toy").rglob("*.jsonl"))


def test_finalizing_without_any_export_is_an_error(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ExportError, match="nothing has been exported"):
        finalize_export(tmp_path / "empty")


# --------------------------------------------------------------------------- sharding
#
# One array task per dataset is fine until one dataset holds far more files than the rest.
# On the real blend dolmino has 40,003 source files against a median near 500, which at a
# second per file is twelve hours against twenty minutes -- past the wall limit.


def test_shards_split_the_files_and_together_cover_everything(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=2.0))

    for shard_id in range(2):
        _export(tmp_path, entry, manifest, shard_id=shard_id, num_shards=2)

    # Two source files, one per task, and between them the whole dataset.
    assert len(sorted((tmp_path / "out" / "toy").rglob("*.jsonl"))) == 2
    exported = yaml.safe_load((tmp_path / "out" / EXPORT_MANIFEST).read_text())
    assert exported["n_lines"] == 2 * N_DOCS
    assert len(_lines(tmp_path / "out")) == 2 * N_DOCS


def test_a_sharded_export_matches_an_unsharded_one(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy", ratio=1.5))

    _export(tmp_path, entry, manifest, out="whole")
    for shard_id in range(3):
        _export(tmp_path, entry, manifest, out="split", shard_id=shard_id, num_shards=3)

    assert _lines(tmp_path / "split") == _lines(tmp_path / "whole")
    whole = yaml.safe_load((tmp_path / "whole" / EXPORT_MANIFEST).read_text())
    split = yaml.safe_load((tmp_path / "split" / EXPORT_MANIFEST).read_text())
    for key in ("n_lines", "n_documents", "n_bytes"):
        assert split[key] == whole[key], key
    assert split["datasets"][0]["n_shards"] == whole["datasets"][0]["n_shards"]


def test_each_task_records_its_own_counts(tmp_path: Path, entry: DatasetEntry, blend):
    # A shared record would have concurrent tasks overwrite each other's counts, which is the
    # race the per-dataset split already avoids at the blend level.
    manifest = blend(DatasetSelection(name="toy"))
    for shard_id in range(2):
        _export(tmp_path, entry, manifest, shard_id=shard_id, num_shards=2)

    records = sorted((tmp_path / "out" / "toy").glob("_export*.yaml"))
    assert len(records) == 2, "one record per task"
    assert sum(yaml.safe_load(p.read_text())["n_lines"] for p in records) == N_DOCS


def test_a_shard_id_outside_the_range_is_refused(tmp_path: Path, entry: DatasetEntry, blend):
    manifest = blend(DatasetSelection(name="toy"))
    with pytest.raises(ValueError, match="not in"):
        _export(tmp_path, entry, manifest, shard_id=3, num_shards=3)
