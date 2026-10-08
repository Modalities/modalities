"""Tests for the token estimator, which was quietly wrong in two ways at once.

A corpus's bytes-per-token ratio is not one number. On the German FineWiki snapshot it runs
from 3.6 for documents under a kilobyte to 34.6 for the nine documents above 256 KB, and
those nine hold 5.7% of all bytes. Two consequences, both of which bit:

* Measuring one global ratio makes every estimate hostage to whether the sample happened
  to include those documents. Sampling uniformly by document count, it almost never does.
* Taking the sample from the *start* of each file is deterministic, so it looked perfectly
  stable across seeds while being systematically wrong -- 16% on FineWiki, because a
  corpus ordered by article id, fetch time or source is not homogeneous along its length.

So documents are sampled at offsets spread across each file, the document *containing*
each offset is taken (which makes selection proportional to length), and the ratio is
measured per size stratum. These tests pin the properties that makes work, using a
synthetic corpus with the same shape: many small documents and a few enormous ones whose
tokens-per-byte differs sharply.
"""

import json
from pathlib import Path

import pytest

from modalities.dataloader.preprocessing.quality.tokens import (
    SIZE_STRATUM_BOUNDS,
    _sample_documents,
    TokenCalibration,
    calibrate_dataset,
    size_stratum,
)


class _CharTokenizer:
    """One token per character, so token counts are exactly predictable."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class _WordTokenizer:
    """One token per whitespace-separated word."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture
def skewed_corpus(tmp_path: Path) -> tuple[Path, int, int]:
    """A corpus whose bytes live mostly in a few long documents that tokenize differently.

    Short documents are single characters repeated, so with the word tokenizer they are one
    token each -- a high bytes-per-token ratio. Long documents are spaced words, so they
    are many tokens -- a low ratio. The population ratio therefore depends heavily on the
    long documents, which are rare by count and dominant by bytes: the exact shape that
    defeated the original estimator.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    total_bytes = 0
    total_tokens = 0
    tokenizer = _WordTokenizer()
    with (corpus / "shard.jsonl").open("w") as f:
        for i in range(3000):
            text = "x" * 200
            f.write(json.dumps({"id": f"s{i}", "text": text}) + "\n")
            total_bytes += len(text.encode())
            total_tokens += len(tokenizer.tokenize(text))
        for i in range(12):
            text = " ".join(["word"] * 120_000)
            f.write(json.dumps({"id": f"l{i}", "text": text}) + "\n")
            total_bytes += len(text.encode())
            total_tokens += len(tokenizer.tokenize(text))
    return corpus, total_bytes, total_tokens


def test_size_stratum_boundaries_are_inclusive_below():
    assert size_stratum(0) == 0
    assert size_stratum(SIZE_STRATUM_BOUNDS[0] - 1) == 0
    assert size_stratum(SIZE_STRATUM_BOUNDS[0]) == 1
    assert size_stratum(10**12) == len(SIZE_STRATUM_BOUNDS)


def test_the_sample_is_not_the_start_of_the_file(tmp_path: Path):
    """The original sampler read a prefix, which is deterministic and so looked stable
    across seeds while being systematically wrong on any corpus that varies along its
    length -- 16% out on FineWiki."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with (corpus / "shard.jsonl").open("w") as f:
        for i in range(2000):
            f.write(json.dumps({"id": f"first-{i}", "text": "a" * 400}) + "\n")
        for i in range(2000):
            f.write(json.dumps({"id": f"second-{i}", "text": "b" * 400}) + "\n")

    sample = _sample_documents(
        file_paths=[corpus / "shard.jsonl"],
        text_field="text",
        sample_size=200,
        seed=1,
        max_probe_files=32,
        max_lines_per_probe=100_000,
    )
    halves = {record["id"].split("-")[0] for record in sample}
    assert halves == {"first", "second"}, f"sample only reached {halves}; it is not spread over the file"


def test_stratified_estimate_beats_the_global_ratio_on_a_skewed_corpus(
    skewed_corpus: tuple[Path, int, int],
):
    corpus, total_bytes, total_tokens = skewed_corpus
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=[corpus / "shard.jsonl"],
        tokenizer=_WordTokenizer(),
        tokenizer_name="word",
        sample_size=1000,
    )

    sizes = []
    with (corpus / "shard.jsonl").open() as f:
        for line in f:
            sizes.append(len(json.loads(line)["text"].encode()))

    stratified = sum(round(b / calibration.ratio_for(b)) for b in sizes)
    global_only = sum(round(b / calibration.bytes_per_token) for b in sizes)
    stratified_error = abs(stratified / total_tokens - 1)
    global_error = abs(global_only / total_tokens - 1)

    assert stratified_error < 0.05, f"stratified estimate was {stratified_error:.1%} out"
    assert stratified_error < global_error


def test_long_documents_are_actually_reached(skewed_corpus: tuple[Path, int, int]):
    """Sampling by containing document, not the following one, is what finds them.

    Twelve documents out of 3,012 hold most of the bytes. Uniform-by-count sampling finds
    them at a rate of 0.4%; length-proportional sampling finds them in proportion to their
    share of the corpus, which is what the ratio needs.
    """
    corpus, _, _ = skewed_corpus
    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=[corpus / "shard.jsonl"],
        tokenizer=_WordTokenizer(),
        tokenizer_name="word",
        sample_size=1000,
    )
    long_stratum = size_stratum(len(" ".join(["word"] * 120_000).encode()))
    assert calibration.stratum_documents[long_stratum] >= TokenCalibration.MIN_STRATUM_DOCUMENTS


def test_a_sparse_stratum_falls_back_to_the_global_ratio():
    calibration = TokenCalibration(
        dataset="toy",
        tokenizer="t",
        bytes_per_token=4.0,
        stratum_bytes_per_token=[3.0] + [99.0] * len(SIZE_STRATUM_BOUNDS),
        stratum_documents=[500] + [1] * len(SIZE_STRATUM_BOUNDS),
    )
    assert calibration.ratio_for(10) == 3.0
    # Measured from one document, so not trusted as an estimator of its own.
    assert calibration.ratio_for(10**9) == 4.0


def test_a_calibration_without_strata_still_estimates():
    """Calibrations written before strata existed must keep working."""
    calibration = TokenCalibration(dataset="toy", tokenizer="t", bytes_per_token=4.0)
    assert calibration.ratio_for(10) == 4.0
    assert calibration.estimate({}, text_bytes=400) == 101


def test_a_native_token_field_still_wins(tmp_path: Path):
    """Stratification applies to the bytes-per-token path only; a corpus carrying its own
    token count is estimated from that, per document, which needs no stratifying."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with (corpus / "shard.jsonl").open("w") as f:
        for i in range(500):
            text = " ".join(["word"] * (10 + i))
            f.write(json.dumps({"id": str(i), "text": text, "token_count": len(text.split())}) + "\n")

    calibration = calibrate_dataset(
        dataset_name="toy",
        file_paths=[corpus / "shard.jsonl"],
        tokenizer=_WordTokenizer(),
        tokenizer_name="word",
        sample_size=200,
    )
    assert calibration.uses_native_field()
    assert calibration.native_field == "token_count"
    assert calibration.estimate({"token_count": 100}, text_bytes=999_999) == 101
