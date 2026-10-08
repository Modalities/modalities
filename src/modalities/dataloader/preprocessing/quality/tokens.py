"""Per-document token estimates for datasets that have not been tokenized yet.

Selecting documents before tokenizing means the token budget of a selection has to be
predicted rather than measured. Two properties matter:

* The estimate must be **per document**, not per corpus. Quality correlates with
  length, so a filter that keeps the better documents keeps longer ones too. Scaling a
  corpus average by a row-retention rate therefore understates the surviving tokens,
  sometimes badly.
* The estimate must be based on the **text**, not the stored line. Several corpora keep
  several renderings of the same document in one record -- HPLT stores ``text``, ``xml``
  and ``md`` side by side -- so the JSON line can be three times the size of the text
  that will actually be tokenized.

Two estimators are supported, in order of preference:

1. A token count the corpus already carries, rescaled to our tokenizer by a measured
   factor. Most such fields were produced with a different tokenizer, so they are
   proportional to our counts rather than equal to them.
2. The text byte length divided by a measured bytes-per-token ratio.

Both factors are measured per dataset by tokenizing a sample, because they vary a lot
across languages and content types.
"""

from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional

import yaml
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    # Only `calibrate_dataset` needs a tokenizer. Importing it eagerly would make
    # previewing a blend depend on transformers and sentencepiece, which the preview
    # path has no use for -- estimates are applied from stored constants.
    from modalities.tokenization.tokenizer_wrapper import TokenizerWrapper


# Document byte-length boundaries defining the calibration's size strata. Log-spaced,
# because the bytes-per-token ratio varies with document length and document length is
# heavy-tailed: a handful of very long documents dominate a corpus-wide sum ratio, so
# measuring one global constant makes every estimate hostage to whether the sample
# happened to catch them. Six strata are enough to flatten that without needing a large
# sample in each.
SIZE_STRATUM_BOUNDS: tuple[int, ...] = (1_024, 4_096, 16_384, 65_536, 262_144)


def size_stratum(text_bytes: int) -> int:
    """Finds which size stratum a document belongs to.

    Args:
        text_bytes (int): UTF-8 byte length of the document's text.

    Returns:
        int: Index into the stratum list, in ``[0, len(SIZE_STRATUM_BOUNDS)]``.
    """
    return bisect.bisect_right(SIZE_STRATUM_BOUNDS, text_bytes)


class TokenCalibration(BaseModel):
    """Measured constants relating a dataset's records to our tokenizer's counts.

    Attributes:
        dataset (str): Name of the dataset this calibration was measured on.
        tokenizer (str): Identifier of the tokenizer used, recorded so a stale
            calibration cannot be applied to a different tokenizer unnoticed.
        bytes_per_token (float): Mean UTF-8 text bytes per token.
        native_field (Optional[str]): Record field holding the corpus's own token
            count, if it has one.
        native_scale (Optional[float]): Multiplier turning that field into our
            tokenizer's count.
        native_coverage (float): Fraction of sampled documents that carried
            ``native_field``. A field present in only some records is not usable as
            the primary estimator.
        sampled_documents (int): How many documents the calibration was measured on.
        sampled_tokens (int): How many tokens those documents produced.
        eod_tokens_per_document (int): Tokens the packer appends per document, added
            to every estimate so the prediction matches what packing produces.
        stratum_bytes_per_token (list[Optional[float]]): Ratio measured within each size
            stratium of :data:`SIZE_STRATUM_BOUNDS`, or None for a stratum the sample did
            not reach. Conditioning on length is what makes the estimate robust: a
            document's own byte length is known exactly from the sidecar, so it can be
            costed against documents of its own size class instead of a corpus-wide mean
            dominated by the longest documents.
        stratum_documents (list[int]): Sampled documents per stratum, so a ratio measured
            from too few documents can be recognised.
    """

    dataset: str
    tokenizer: str
    bytes_per_token: float = Field(gt=0)
    native_field: Optional[str] = None
    native_scale: Optional[float] = Field(default=None, gt=0)
    native_coverage: float = 0.0
    sampled_documents: int = 0
    sampled_tokens: int = 0
    eod_tokens_per_document: int = 1
    stratum_bytes_per_token: list[Optional[float]] = Field(default_factory=list)
    stratum_documents: list[int] = Field(default_factory=list)

    # A native field present in fewer than this share of sampled documents is ignored,
    # because falling back per record would mix two estimators with different biases.
    MIN_NATIVE_COVERAGE: ClassVar[float] = 0.99

    # A stratum measured from fewer documents than this falls back to the global ratio,
    # so a stratum reached by one outlier does not become an estimator of its own.
    MIN_STRATUM_DOCUMENTS: ClassVar[int] = 20

    def uses_native_field(self) -> bool:
        """Whether the corpus's own token count is the primary estimator.

        Returns:
            bool: True if a native field was found on effectively every sampled
                document and a scale factor was measured for it.
        """
        return (
            self.native_field is not None
            and self.native_scale is not None
            and self.native_coverage >= self.MIN_NATIVE_COVERAGE
        )

    def estimate(self, record: dict[str, Any], text_bytes: int) -> int:
        """Estimates the tokens one document will contribute.

        Args:
            record (dict[str, Any]): The decoded JSONL record.
            text_bytes (int): UTF-8 byte length of the document's text field.

        Returns:
            int: Estimated token count, including the end-of-document token(s) the
                packer appends. Never negative.
        """
        if self.uses_native_field():
            native = record.get(self.native_field)
            if isinstance(native, (int, float)):
                return max(0, round(native * self.native_scale) + self.eod_tokens_per_document)
        return max(0, round(text_bytes / self.ratio_for(text_bytes)) + self.eod_tokens_per_document)

    def ratio_for(self, text_bytes: int) -> float:
        """Picks the bytes-per-token ratio to apply to a document of this length.

        Args:
            text_bytes (int): UTF-8 byte length of the document's text.

        Returns:
            float: The stratum's ratio if one was measured from enough documents,
                otherwise the corpus-wide ratio.
        """
        if not self.stratum_bytes_per_token:
            return self.bytes_per_token
        index = size_stratum(text_bytes)
        if index >= len(self.stratum_bytes_per_token):
            return self.bytes_per_token
        ratio = self.stratum_bytes_per_token[index]
        enough = index < len(self.stratum_documents) and self.stratum_documents[index] >= self.MIN_STRATUM_DOCUMENTS
        return ratio if (ratio and enough) else self.bytes_per_token


class CalibrationSet(BaseModel):
    """Calibrations for every dataset of a blend.

    Attributes:
        tokenizer (str): Identifier of the tokenizer all entries were measured with.
        calibrations (dict[str, TokenCalibration]): Keyed by dataset name.
    """

    tokenizer: str
    calibrations: dict[str, TokenCalibration] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "CalibrationSet":
        """Loads a calibration set from YAML.

        Args:
            path (Path): Path to the calibration file.

        Returns:
            CalibrationSet: The parsed calibrations.
        """
        with Path(path).open() as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: Path) -> None:
        """Writes the calibration set to YAML.

        Args:
            path (Path): Destination path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)

    def get(self, dataset: str) -> TokenCalibration:
        """Looks up one dataset's calibration.

        Args:
            dataset (str): The dataset name.

        Returns:
            TokenCalibration: The calibration for that dataset.

        Raises:
            KeyError: If the dataset has not been calibrated. Estimating without a
                calibration would silently invent a token budget, so this is an error
                rather than a default.
        """
        if dataset not in self.calibrations:
            raise KeyError(
                f"no token calibration for dataset {dataset!r}; "
                f"run 'modalities data quality calibrate' first. Calibrated: {sorted(self.calibrations)}"
            )
        return self.calibrations[dataset]


# Token-count fields observed across the corpora, most specific first. A field is only
# adopted if it is present on effectively every sampled document of a dataset.
NATIVE_TOKEN_FIELDS: tuple[str, ...] = (
    "token_count",
    "num_tokens",
    "total_tokens",
    "len_cl100k_base",
)


def _probe_files(file_paths: list[Path], max_probe_files: int) -> list[Path]:
    # An evenly spaced selection, so the sample spans the whole dataset without the read
    # growing with the file count. Corpora here range from 4 files to 40 003; reading a
    # fixed number of lines from every one of them would mean 26 TB for the largest.
    if len(file_paths) <= max_probe_files:
        return list(file_paths)
    step = len(file_paths) / max_probe_files
    return [file_paths[min(int(i * step), len(file_paths) - 1)] for i in range(max_probe_files)]


def _line_start(handle: Any, pos: int, window: int = 65_536) -> int:
    """Finds the offset of the line containing a byte position.

    Args:
        handle (Any): Binary file handle.
        pos (int): Byte position somewhere inside the wanted line.
        window (int): How many bytes to read backwards at a time looking for the previous
            newline. Documents longer than this are rare, and the loop extends for them.

    Returns:
        int: Offset of the first byte of the line containing ``pos``.
    """
    if pos <= 0:
        return 0
    cursor = pos
    while cursor > 0:
        back = max(0, cursor - window)
        handle.seek(back)
        chunk = handle.read(cursor - back)
        index = chunk.rfind(b"\n")
        if index != -1:
            return back + index + 1
        cursor = back
    return 0


def _sample_one_span(
    handle: Any,
    start: int,
    text_field: str,
    max_lines: int = 64,
) -> Optional[dict[str, Any]]:
    """Reads the document containing a byte offset.

    The *containing* document, not the one after it, and that distinction is the whole
    point. Returning the following document makes selection uniform across documents,
    because which document follows a position is independent of how long that position's
    document is. Returning the containing one makes selection proportional to length,
    which is what a bytes-per-token ratio needs: the ratio is a byte-weighted quantity, so
    the documents that dominate it must be the ones most likely to be sampled.

    On the German FineWiki snapshot the nine documents of 256 KB and above hold 5.7 % of
    all bytes and tokenize at 34.6 bytes per token against 3.6 for the small ones. Uniform
    sampling found two of them in 2,000 draws and the estimate was 19 % out; length-
    proportional sampling finds forty and it is within 0.5 %.

    Args:
        handle (Any): Binary file handle to seek within.
        start (int): Byte offset to sample at.
        text_field (str): Field that must hold a string for the record to be usable.
        max_lines (int): How many lines to try before abandoning this span, so a run of
            records without the text field cannot turn one span into a full-file scan.

    Returns:
        Optional[dict[str, Any]]: The decoded record, or None if the span yielded none.
    """
    handle.seek(_line_start(handle, start))
    for _ in range(max_lines):
        line = handle.readline()
        if not line:
            return None
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record.get(text_field), str):
            return record
    return None


def _measure_slice(text: str, max_bytes: int, rnd: random.Random) -> str:
    """Takes a bounded slice of a document to measure its ratio on.

    Sampling documents in proportion to length is what makes the estimate accurate, but it
    also means the multi-megabyte documents get sampled, and tokenizing those dominates the
    cost: the work scales with the mean of the squared length over the mean length, which
    heavy tails make enormous. Calibrating five snapshot datasets ran past ten minutes.

    A document's own bytes-per-token ratio is far more uniform inside the document than it
    is across documents, so a slice measures it well. The slice is taken from a random
    position rather than the head, for the same reason the file sample is not a prefix: a
    document's opening prose is not representative of a document that is mostly a table.

    Args:
        text (str): The document text.
        max_bytes (int): Largest slice to measure.
        rnd (random.Random): Seeded source for the slice position.

    Returns:
        str: The whole text if it is small enough, otherwise a slice of it.
    """
    if len(text) <= max_bytes // 4:
        # Even at 4 bytes per character this cannot exceed the cap, so skip the encode.
        return text
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    start = rnd.randrange(len(encoded) - max_bytes + 1)
    # A slice of encoded bytes can begin or end mid-character; drop the broken edges.
    return encoded[start : start + max_bytes].decode("utf-8", errors="ignore")


def _sample_documents(
    file_paths: Iterable[Path],
    text_field: str,
    sample_size: int,
    seed: int,
    max_probe_files: int,
    max_lines_per_probe: int,
) -> list[dict[str, Any]]:
    """Samples documents from across a dataset without reading it whole.

    Documents are taken at byte offsets spread evenly over each probe file, one per span,
    rather than from the start of the file. Reading a prefix instead was a real defect: it
    is deterministic, so it looked stable across seeds while being systematically wrong.
    On the German FineWiki snapshot the first 2,000 documents gave 3.531 bytes per token
    where the whole file gives 4.214 -- a 16 % error in every token estimate downstream --
    because a corpus ordered by article id, fetch time, or source is not homogeneous along
    its length.

    Each span contributes the document containing its offset, so a document is sampled in
    proportion to its length and a document spanning several spans is counted by each of
    them. That multiplicity is deliberate -- it is the weighting that makes each stratum's
    measured ratio byte-weighted, which is what predicting a token total from a byte total
    requires. See :func:`_sample_one_span`.

    Args:
        file_paths (Iterable[Path]): The dataset's JSONL files.
        text_field (str): The field holding the document text.
        sample_size (int): Target number of documents.
        seed (int): Seed for placing the offset within each span, so a calibration is
            reproducible while the offsets are not degenerate multiples of the span size.
        max_probe_files (int): How many files to draw from, spread across the dataset.
        max_lines_per_probe (int): Cap on lines read per span before abandoning it.

    Returns:
        list[dict[str, Any]]: The sampled records.
    """
    files = _probe_files(list(file_paths), max_probe_files)
    if not files:
        return []
    per_file = max(1, -(-sample_size // len(files)))
    lines_per_span = max(1, min(64, max_lines_per_probe))
    rnd = random.Random(seed)

    collected: list[dict[str, Any]] = []
    for path in files:
        try:
            size = path.stat().st_size
            if size == 0:
                continue
            with path.open("rb") as f:
                for i in range(per_file):
                    lo = size * i // per_file
                    hi = size * (i + 1) // per_file
                    start = lo if hi <= lo + 1 else lo + rnd.randrange(hi - lo)
                    record = _sample_one_span(f, start, text_field, max_lines=lines_per_span)
                    if record is not None:
                        collected.append(record)
        except OSError:
            continue

    if len(collected) > sample_size:
        collected = random.Random(seed).sample(collected, sample_size)
    return collected


def calibrate_dataset(
    dataset_name: str,
    file_paths: list[Path],
    tokenizer: "TokenizerWrapper",
    tokenizer_name: str,
    text_field: str = "text",
    sample_size: int = 4000,
    seed: int = 42,
    max_probe_files: int = 32,
    max_lines_per_probe: int = 100_000,
    eod_tokens_per_document: int = 1,
    max_measure_bytes: int = 65_536,
) -> TokenCalibration:
    """Measures how a dataset's records relate to our tokenizer's token counts.

    Args:
        dataset_name (str): Name recorded in the calibration.
        file_paths (list[Path]): The dataset's JSONL files.
        tokenizer (TokenizerWrapper): The tokenizer training will use.
        tokenizer_name (str): Identifier recorded alongside the measurement.
        text_field (str): The field holding the document text.
        sample_size (int): How many documents to tokenize. The default leaves margin: the
            stratified estimate held within 0.5 % across seeds at 2,000 documents on
            FineWiki, but one 1,000-document sample was 9.7 % out, and the cost of a larger
            sample is seconds per dataset.
        seed (int): Seed for trimming the sample, so calibration is reproducible.
        max_probe_files (int): How many files to draw the sample from, spread evenly
            across the dataset. This bounds the read: the cost of calibrating is set by
            the sample size, not by how many files the dataset happens to have.
        max_lines_per_probe (int): Safety cap on lines scanned in one probe file, for a
            file whose records mostly lack the text field.
        eod_tokens_per_document (int): Tokens the packer appends per document.
        max_measure_bytes (int): Largest slice of one document to tokenize. Bounds the cost
            of calibrating, which would otherwise be set by the longest documents in the
            corpus. See :func:`_measure_slice`.

    Returns:
        TokenCalibration: The measured calibration.

    Raises:
        ValueError: If no documents could be sampled, which means the files are empty,
            unreadable, or the text field name is wrong.
    """
    documents = _sample_documents(
        file_paths=file_paths,
        text_field=text_field,
        sample_size=sample_size,
        seed=seed,
        max_probe_files=max_probe_files,
        max_lines_per_probe=max_lines_per_probe,
    )
    if not documents:
        raise ValueError(
            f"dataset {dataset_name!r}: no documents with a string {text_field!r} field were found in "
            f"{len(file_paths)} file(s); check the registry's text_field and glob"
        )

    n_strata = len(SIZE_STRATUM_BOUNDS) + 1
    stratum_bytes = [0] * n_strata
    stratum_tokens = [0] * n_strata
    stratum_docs = [0] * n_strata

    # Documents are sampled in proportion to their length, so a plain sum ratio over the
    # whole sample is weighted by the square of length and comes out dominated by the
    # longest documents -- 62 % low on FineWiki. The corpus-wide ratio therefore uses
    # inverse-probability weights, which is the unbiased estimator of sum(bytes) over
    # sum(tokens) under this sampling scheme. Within a stratum the plain ratio is kept: it
    # is byte-weighted, which is what costing a stratum's bytes calls for, and the length
    # spread inside one stratum is small enough for the residual bias not to matter.
    inverse_weighted_tokens = 0.0
    weighted_documents = 0

    total_text_bytes = 0
    total_tokens = 0
    native_totals: dict[str, float] = {field: 0.0 for field in NATIVE_TOKEN_FIELDS}
    native_counts: dict[str, int] = {field: 0 for field in NATIVE_TOKEN_FIELDS}
    native_tokens: dict[str, float] = {field: 0.0 for field in NATIVE_TOKEN_FIELDS}

    measure_rnd = random.Random(seed)
    for record in documents:
        text = record[text_field]
        measured = _measure_slice(text, max_measure_bytes, measure_rnd)
        n_tokens = len(tokenizer.tokenize(measured))
        n_bytes = len(measured.encode("utf-8"))
        # The stratum is the document's real size class, even when the ratio was measured
        # on a slice of it: what is being estimated is the ratio for documents of that size.
        full_bytes = len(text.encode("utf-8")) if measured is not text else n_bytes
        total_text_bytes += n_bytes
        total_tokens += n_tokens
        stratum = size_stratum(full_bytes)
        stratum_bytes[stratum] += n_bytes
        stratum_tokens[stratum] += n_tokens
        stratum_docs[stratum] += 1
        if n_bytes > 0:
            inverse_weighted_tokens += n_tokens / n_bytes
            weighted_documents += 1
        for field in NATIVE_TOKEN_FIELDS:
            value = record.get(field)
            if isinstance(value, (int, float)) and value > 0:
                # The native scale relates a whole document's count to whole-document
                # tokens, so a sliced measurement has to be extrapolated back up.
                scaled_tokens = n_tokens * (full_bytes / n_bytes) if n_bytes else 0
                native_totals[field] += float(value)
                native_counts[field] += 1
                native_tokens[field] += scaled_tokens

    if total_tokens == 0:
        raise ValueError(f"dataset {dataset_name!r}: sampled documents produced zero tokens")

    global_bytes_per_token = (
        weighted_documents / inverse_weighted_tokens if inverse_weighted_tokens > 0 else total_text_bytes / total_tokens
    )

    native_field: Optional[str] = None
    native_scale: Optional[float] = None
    native_coverage = 0.0
    for field in NATIVE_TOKEN_FIELDS:
        coverage = native_counts[field] / len(documents)
        if coverage >= TokenCalibration.MIN_NATIVE_COVERAGE and native_totals[field] > 0:
            native_field = field
            # Scale is measured only over the documents that carry the field, so a
            # partially present field cannot skew it.
            native_scale = native_tokens[field] / native_totals[field]
            native_coverage = coverage
            break

    return TokenCalibration(
        dataset=dataset_name,
        tokenizer=tokenizer_name,
        bytes_per_token=global_bytes_per_token,
        native_field=native_field,
        native_scale=native_scale,
        native_coverage=native_coverage,
        sampled_documents=len(documents),
        sampled_tokens=total_tokens,
        eod_tokens_per_document=eod_tokens_per_document,
        stratum_bytes_per_token=[
            (stratum_bytes[i] / stratum_tokens[i]) if stratum_tokens[i] else None for i in range(n_strata)
        ],
        stratum_documents=stratum_docs,
    )
