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

    # A native field present in fewer than this share of sampled documents is ignored,
    # because falling back per record would mix two estimators with different biases.
    MIN_NATIVE_COVERAGE: ClassVar[float] = 0.99

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
        return max(0, round(text_bytes / self.bytes_per_token) + self.eod_tokens_per_document)


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


def _reservoir_sample_documents(
    file_paths: Iterable[Path],
    text_field: str,
    sample_size: int,
    seed: int,
    max_lines_per_file: int,
) -> list[dict[str, Any]]:
    # Draws documents from across the whole dataset, not just its first file, so the
    # calibration is not biased by whatever happens to sit at the front of the corpus.
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for path in file_paths:
        try:
            with path.open(errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= max_lines_per_file:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record.get(text_field), str):
                        continue
                    seen += 1
                    if len(reservoir) < sample_size:
                        reservoir.append(record)
                    else:
                        j = rng.randrange(seen)
                        if j < sample_size:
                            reservoir[j] = record
        except OSError:
            continue
    return reservoir


def calibrate_dataset(
    dataset_name: str,
    file_paths: list[Path],
    tokenizer: "TokenizerWrapper",
    tokenizer_name: str,
    text_field: str = "text",
    sample_size: int = 2000,
    seed: int = 42,
    max_lines_per_file: int = 20000,
    eod_tokens_per_document: int = 1,
) -> TokenCalibration:
    """Measures how a dataset's records relate to our tokenizer's token counts.

    Args:
        dataset_name (str): Name recorded in the calibration.
        file_paths (list[Path]): The dataset's JSONL files.
        tokenizer (TokenizerWrapper): The tokenizer training will use.
        tokenizer_name (str): Identifier recorded alongside the measurement.
        text_field (str): The field holding the document text.
        sample_size (int): How many documents to tokenize.
        seed (int): Seed for the reservoir sample, so calibration is reproducible.
        max_lines_per_file (int): Cap on lines read per file. Keeps the pass bounded on
            corpora whose individual files hold tens of millions of documents.
        eod_tokens_per_document (int): Tokens the packer appends per document.

    Returns:
        TokenCalibration: The measured calibration.

    Raises:
        ValueError: If no documents could be sampled, which means the files are empty,
            unreadable, or the text field name is wrong.
    """
    documents = _reservoir_sample_documents(
        file_paths=file_paths,
        text_field=text_field,
        sample_size=sample_size,
        seed=seed,
        max_lines_per_file=max_lines_per_file,
    )
    if not documents:
        raise ValueError(
            f"dataset {dataset_name!r}: no documents with a string {text_field!r} field were found in "
            f"{len(file_paths)} file(s); check the registry's text_field and glob"
        )

    total_text_bytes = 0
    total_tokens = 0
    native_totals: dict[str, float] = {field: 0.0 for field in NATIVE_TOKEN_FIELDS}
    native_counts: dict[str, int] = {field: 0 for field in NATIVE_TOKEN_FIELDS}
    native_tokens: dict[str, int] = {field: 0 for field in NATIVE_TOKEN_FIELDS}

    for record in documents:
        text = record[text_field]
        n_tokens = len(tokenizer.tokenize(text))
        total_text_bytes += len(text.encode("utf-8"))
        total_tokens += n_tokens
        for field in NATIVE_TOKEN_FIELDS:
            value = record.get(field)
            if isinstance(value, (int, float)) and value > 0:
                native_totals[field] += float(value)
                native_counts[field] += 1
                native_tokens[field] += n_tokens

    if total_tokens == 0:
        raise ValueError(f"dataset {dataset_name!r}: sampled documents produced zero tokens")

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
        bytes_per_token=total_text_bytes / total_tokens,
        native_field=native_field,
        native_scale=native_scale,
        native_coverage=native_coverage,
        sampled_documents=len(documents),
        sampled_tokens=total_tokens,
        eod_tokens_per_document=eod_tokens_per_document,
    )
