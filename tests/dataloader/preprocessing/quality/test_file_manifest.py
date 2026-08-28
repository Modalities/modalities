"""Tests that a sidecar refuses to be used against a source tree that moved.

These cover the failure that actually happened: a data transfer re-sharded four corpora
after their sidecars had been built. File ids are positions in a sorted file list, so the
same id then named a different file and every recorded byte offset pointed into the wrong
place. The only check that existed compared file *counts*, which a re-shard can leave
unchanged, so the pipeline carried on and would have produced a blend of garbage byte
ranges.
"""

import json
import random
from pathlib import Path

import pytest

from modalities.dataloader.preprocessing.quality.file_manifest import FileManifest, ManifestError
from modalities.dataloader.preprocessing.quality.materialize import MaterializationError, materialize_dataset
from modalities.dataloader.preprocessing.quality.registry import DatasetEntry, KeyKind, KeySpec
from modalities.dataloader.preprocessing.quality.selection import DatasetSelection, MissingPolicy
from modalities.dataloader.preprocessing.quality.sidecar import SidecarBuilder
from modalities.dataloader.preprocessing.quality.tokens import TokenCalibration
from modalities.dataloader.preprocessing.quality.verify import adopt_manifest, verify_sidecar


@pytest.fixture
def small_corpus(tmp_path: Path) -> Path:
    """Three shards of documents, long enough that offsets are far from zero."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rng = random.Random(5)
    for shard in range(3):
        with (corpus / f"shard_{shard}.jsonl").open("w") as f:
            for i in range(60):
                f.write(
                    json.dumps({"id": f"doc-{shard}-{i}", "text": " ".join(["word"] * rng.randint(20, 200))}) + "\n"
                )
    return corpus


@pytest.fixture
def entry(small_corpus: Path) -> DatasetEntry:
    return DatasetEntry(
        name="toy",
        jsonl_root=small_corpus,
        glob="*.jsonl",
        key=KeySpec(kind=KeyKind.FIELD, field="id"),
    )


@pytest.fixture
def sidecar(tmp_path: Path, entry: DatasetEntry) -> Path:
    directory = tmp_path / "sidecar"
    calibration = TokenCalibration(dataset="toy", tokenizer="t", bytes_per_token=4.0)
    SidecarBuilder(entry, calibration, index_root=tmp_path / "idx").build(directory, show_progress=False)
    return directory


def test_building_a_sidecar_records_the_file_list(sidecar: Path, entry: DatasetEntry):
    manifest = FileManifest.read(sidecar)
    assert manifest.dataset == "toy"
    assert [f.path for f in manifest.files] == ["shard_0.jsonl", "shard_1.jsonl", "shard_2.jsonl"]
    assert all(f.size > 0 for f in manifest.files)
    assert not manifest.drift(entry)


def test_a_renamed_directory_is_detected(sidecar: Path, entry: DatasetEntry, tmp_path: Path):
    # A tree that moved wholesale is fine: paths are relative, so only the file set has
    # to agree. This is what lets a snapshot be taken without invalidating a sidecar.
    moved = tmp_path / "corpus_moved"
    entry.jsonl_root.rename(moved)
    relocated = entry.model_copy(update={"jsonl_root": moved})
    assert not FileManifest.read(sidecar).drift(relocated)


def test_a_reshard_that_preserves_the_file_count_is_detected(sidecar: Path, entry: DatasetEntry):
    # The real failure: file count unchanged, contents different. The old count check
    # passed this and the byte offsets silently pointed past the end of the file.
    files = sorted(entry.jsonl_root.glob("*.jsonl"))
    assert len(files) == 3
    for path in files:
        path.write_text(json.dumps({"id": "replaced", "text": "short"}) + "\n")

    problems = FileManifest.read(sidecar).drift(entry)
    assert len(problems) == 3
    assert all("bytes, was" in p for p in problems)
    with pytest.raises(ManifestError, match="source tree changed"):
        FileManifest.read(sidecar).require_current(entry)


def test_a_removed_file_is_named(sidecar: Path, entry: DatasetEntry):
    (entry.jsonl_root / "shard_1.jsonl").unlink()
    problems = FileManifest.read(sidecar).drift(entry)
    assert any("shard_1.jsonl is gone" in p for p in problems)


def test_an_added_file_does_not_shift_the_ids(sidecar: Path, entry: DatasetEntry):
    # A file sorting *before* the existing ones is the dangerous case: re-globbing would
    # renumber every id. Resolution goes through the recorded paths, so it cannot.
    (entry.jsonl_root / "aaa_new.jsonl").write_text(json.dumps({"id": "new", "text": "hi"}) + "\n")
    manifest = FileManifest.read(sidecar)
    assert [p.name for p in manifest.resolve(entry)] == [
        "shard_0.jsonl",
        "shard_1.jsonl",
        "shard_2.jsonl",
    ]
    assert any("added to the source tree" in p for p in manifest.drift(entry))


def test_materialize_refuses_a_drifted_source_tree(sidecar: Path, entry: DatasetEntry, tmp_path: Path):
    (entry.jsonl_root / "shard_0.jsonl").write_text(json.dumps({"id": "x", "text": "tiny"}) + "\n")
    with pytest.raises(MaterializationError, match="source tree changed"):
        materialize_dataset(
            sidecar_dir=sidecar,
            dataset_entry=entry,
            dataset_selection=DatasetSelection(name="toy", ratio=1.0),
            missing_policy=MissingPolicy.KEEP,
            output_dir=tmp_path / "out",
            show_progress=False,
        )


def test_materialize_explains_a_missing_manifest(sidecar: Path, entry: DatasetEntry, tmp_path: Path):
    FileManifest.path_for(sidecar).unlink()
    with pytest.raises(MaterializationError, match="verify-sidecar --adopt"):
        materialize_dataset(
            sidecar_dir=sidecar,
            dataset_entry=entry,
            dataset_selection=DatasetSelection(name="toy", ratio=1.0),
            missing_policy=MissingPolicy.KEEP,
            output_dir=tmp_path / "out",
            show_progress=False,
        )


def test_verify_passes_on_an_untouched_corpus(sidecar: Path, entry: DatasetEntry):
    report = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8)
    assert report.verdict == "VALID"
    assert report.n_sampled > 0
    assert report.n_matching == report.n_sampled


def test_verify_reports_broken_when_the_bytes_moved(sidecar: Path, entry: DatasetEntry):
    # Prepend a line to every file: same file names, similar sizes, every offset now
    # points at the wrong document. Sizes catch it, and so do the byte probes.
    for path in sorted(entry.jsonl_root.glob("*.jsonl")):
        original = path.read_text()
        path.write_text(json.dumps({"id": "inserted", "text": "x" * 500}) + "\n" + original)

    report = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8)
    assert report.verdict == "DRIFTED"
    assert not report.ok


def test_verify_ignores_offset_zero_rows(sidecar: Path, entry: DatasetEntry):
    # The first document of any JSONL file parses, so a probe at offset 0 succeeds even
    # against a completely different file. Sampling those is how a broken sidecar looked
    # healthy during the incident.
    report = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8)
    assert report.n_sampled > 0
    manifest = FileManifest.read(sidecar)
    for path in manifest.resolve(entry):
        path.write_text(json.dumps({"id": "only", "text": "tiny"}) + "\n")
    after = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8, check_manifest=False)
    assert after.n_matching == 0


def test_adopt_refuses_a_sidecar_that_does_not_verify(sidecar: Path, entry: DatasetEntry):
    FileManifest.path_for(sidecar).unlink()
    for path in sorted(entry.jsonl_root.glob("*.jsonl")):
        path.write_text(json.dumps({"id": "x", "text": "tiny"}) + "\n")
    report = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8)
    with pytest.raises(ManifestError, match="refusing to write a manifest"):
        adopt_manifest(sidecar, entry, report)


def test_adopt_stamps_a_manifest_onto_a_verified_sidecar(sidecar: Path, entry: DatasetEntry):
    FileManifest.path_for(sidecar).unlink()
    report = verify_sidecar(sidecar, entry, n_parts=3, n_rows_per_part=8)
    assert report.ok and not report.has_manifest
    adopt_manifest(sidecar, entry, report)
    assert not FileManifest.read(sidecar).drift(entry)


def test_manifest_is_not_written_into_the_source_tree(sidecar: Path, entry: DatasetEntry):
    # Source corpora are shared and read-only; the pipeline must leave them alone.
    assert not list(entry.jsonl_root.glob("*.idx"))
    assert not list(entry.jsonl_root.glob("_files.json"))
    assert FileManifest.path_for(sidecar).exists()
