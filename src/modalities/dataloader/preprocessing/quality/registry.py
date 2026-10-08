"""Declares the datasets of a blend and how each one joins to external annotations.

A blend mixes corpora that were built by different people at different times, so the
document identifier is different in almost every one of them. Some carry a plain
``id``, some wrap the same UUID in ``<urn:uuid:...>``, one carries the identifier
under a different name entirely, and two carry no identifier at all and have to be
keyed by a hash of their own text. The registry is where those differences are
written down once, so the rest of the package can treat every dataset the same way.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class KeyKind(str, Enum):
    """How a document's annotation key is obtained.

    Attributes:
        FIELD: The key is a top-level JSON field, used verbatim.
        URN_UUID_FIELD: The key is a JSON field holding a UUID that may or may not be
            wrapped in ``<urn:uuid:...>``. Both forms occur within a single file, on
            both sides of the join, so the wrapper is stripped before comparing.
        SHA256_TEXT: No identifier is stored; the key is the SHA-256 hex digest of the
            document text, taken over the exact UTF-8 bytes with no normalisation.
        SOURCE_POINTER: The identifier is a ``<file>/<line>`` pointer into a separate
            source corpus. The key is resolved by reading that line and hashing its
            text, which is how a translated corpus inherits the annotations of the
            original it was translated from.
    """

    FIELD = "field"
    URN_UUID_FIELD = "urn_uuid_field"
    SHA256_TEXT = "sha256_text"
    SOURCE_POINTER = "source_pointer"


def strip_urn_uuid(value: str) -> str:
    """Reduces a possibly ``<urn:uuid:...>``-wrapped identifier to the bare UUID.

    Args:
        value (str): The identifier as stored, wrapped or bare.

    Returns:
        str: The bare identifier. Values that are not wrapped are returned unchanged.
    """
    if value.startswith("<urn:uuid:") and value.endswith(">"):
        return value[len("<urn:uuid:") : -1]
    return value


def sha256_text(text: str) -> str:
    """Hashes document text the way the annotation corpora key their rows.

    Args:
        text (str): The document text, exactly as stored.

    Returns:
        str: Lowercase hex SHA-256 digest of the UTF-8 encoding of ``text``.

    Note:
        The digest is taken over the unmodified string. Stripping whitespace or
        appending a newline both produce keys that match nothing.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KeySpec(BaseModel):
    """Describes how to derive the annotation key for one dataset.

    Attributes:
        kind (KeyKind): Which derivation to apply.
        field (Optional[str]): The JSON field holding the identifier or pointer.
            Required for every kind except ``SHA256_TEXT``.
        text_field (str): The JSON field holding the document text. Used by
            ``SHA256_TEXT``, and by ``SOURCE_POINTER`` when reading the source corpus.
        source_root (Optional[Path]): Directory holding the source corpus that
            ``SOURCE_POINTER`` resolves into.
        source_line_offset (int): Index of the first line of a source file. The
            pointers we have seen are zero-indexed; an off-by-one here shifts every
            annotation by one document and still looks like a working join.
    """

    kind: KeyKind
    field: Optional[str] = None
    text_field: str = "text"
    source_root: Optional[Path] = None
    source_line_offset: int = 0

    @model_validator(mode="after")
    def _check_required_parts(self) -> "KeySpec":
        if self.kind in (KeyKind.FIELD, KeyKind.URN_UUID_FIELD, KeyKind.SOURCE_POINTER) and not self.field:
            raise ValueError(f"key kind {self.kind.value} requires 'field'")
        if self.kind == KeyKind.SOURCE_POINTER and self.source_root is None:
            raise ValueError("key kind source_pointer requires 'source_root'")
        return self

    def derive(self, record: dict[str, Any]) -> Optional[str]:
        """Derives the annotation key for a single decoded JSON record.

        Args:
            record (dict[str, Any]): One decoded JSONL line.

        Returns:
            Optional[str]: The key, or None if the record lacks the needed field.

        Note:
            ``SOURCE_POINTER`` is not resolved here, because resolving it requires
            reading another corpus. This returns the raw pointer; use
            ``SourcePointerResolver`` to turn a batch of pointers into keys.
        """
        if self.kind == KeyKind.SHA256_TEXT:
            text = record.get(self.text_field)
            return sha256_text(text) if isinstance(text, str) else None

        raw = record.get(self.field)
        if raw is None:
            return None
        if self.kind == KeyKind.URN_UUID_FIELD:
            return strip_urn_uuid(str(raw))
        return str(raw)


class SourcePointerResolver:
    """Turns ``<file>/<line>`` pointers into text hashes of a separate source corpus.

    A translated corpus keeps a pointer back to the line of the original it came from.
    Its own text is in another language, so hashing it matches nothing; the annotation
    belongs to the original. Resolving means reading those specific lines of the
    source corpus.

    Pointers are grouped by source file and each file is read once in a single
    forward pass, because the source files are tens of gigabytes each and seeking per
    pointer would be far slower than streaming.
    """

    def __init__(self, source_root: Path, text_field: str = "text", line_offset: int = 0):
        """
        Args:
            source_root (Path): Directory holding the source corpus files.
            text_field (str): The JSON field holding the source document text.
            line_offset (int): Index of the first line of a source file.
        """
        self._source_root = Path(source_root)
        self._text_field = text_field
        self._line_offset = line_offset

    @staticmethod
    def split_pointer(pointer: str) -> tuple[str, int]:
        """Splits a ``<file>/<line>`` pointer into its parts.

        Args:
            pointer (str): The pointer as stored, e.g. ``part_29.jsonl/5279753``.

        Returns:
            tuple[str, int]: The source file name and the line number.

        Raises:
            ValueError: If the pointer has no ``/`` or a non-integer line number.
        """
        file_name, _, line_str = pointer.rpartition("/")
        if not file_name:
            raise ValueError(f"pointer {pointer!r} is not of the form <file>/<line>")
        try:
            return file_name, int(line_str)
        except ValueError as e:
            raise ValueError(f"pointer {pointer!r} has a non-integer line number") from e

    def resolve(self, pointers: list[str]) -> dict[str, str]:
        """Resolves pointers to annotation keys.

        Args:
            pointers (list[str]): Pointers of the form ``<file>/<line>``.

        Returns:
            dict[str, str]: Maps each resolvable pointer to the SHA-256 digest of the
                source document's text. Pointers whose source file is missing, or
                whose line number is past the end of the file, are absent from the
                result rather than mapped to a wrong key.
        """
        wanted: dict[str, dict[int, str]] = {}
        for pointer in pointers:
            file_name, line_no = self.split_pointer(pointer)
            wanted.setdefault(file_name, {})[line_no - self._line_offset] = pointer

        resolved: dict[str, str] = {}
        for file_name, lines_wanted in wanted.items():
            source_path = self._source_root / file_name
            if not source_path.is_file():
                continue
            last_wanted = max(lines_wanted)
            with source_path.open(errors="replace") as f:
                for i, line in enumerate(f):
                    if i in lines_wanted:
                        try:
                            text = json.loads(line).get(self._text_field)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(text, str):
                            resolved[lines_wanted[i]] = sha256_text(text)
                    if i >= last_wanted:
                        break
        return resolved


class NativeMetric(BaseModel):
    """A quality signal already present in the dataset's own records.

    Attributes:
        name (str): Name used for this metric in selection configs and the cube.
        jq_pattern (str): jq expression evaluated against each record, following the
            same convention as ``PackedDataGenerator``'s ``jq_pattern``.
        aggregation (Optional[str]): How to reduce a list-valued result to one number.
            One of ``first``, ``min``, ``max``, ``mean``. Several corpora store
            per-page score arrays rather than a single document score.
    """

    name: str
    jq_pattern: str
    aggregation: Optional[str] = None


class DatasetEntry(BaseModel):
    """One dataset of the blend.

    Attributes:
        name (str): Identifier used in selection configs and reports.
        jsonl_root (Path): Directory containing the dataset's ``.jsonl`` files.
        glob (str): Pattern matching the dataset's files below ``jsonl_root``.
        annotation_split (Optional[str]): Path of the matching annotation split,
            relative to the annotation root. None means the dataset has no external
            annotations and can only be shaped by its native metrics.
        key (Optional[KeySpec]): How to derive the annotation key. Required whenever
            ``annotation_split`` is set.
        native_metrics (list[NativeMetric]): Quality signals to read out of the
            dataset's own records.
        text_field (str): The JSON field holding the document text.
        enabled (bool): Whether the dataset takes part in the blend. Kept so a dataset
            can be registered, and the reason for excluding it recorded, without
            deleting its entry.
        note (Optional[str]): Free-text remark carried into reports.
    """

    name: str
    jsonl_root: Path
    glob: str = "**/*.jsonl"
    annotation_split: Optional[str] = None
    key: Optional[KeySpec] = None
    native_metrics: list[NativeMetric] = Field(default_factory=list)
    text_field: str = "text"
    enabled: bool = True
    note: Optional[str] = None

    @model_validator(mode="after")
    def _check_key_present_for_annotated(self) -> "DatasetEntry":
        if self.annotation_split and self.key is None:
            raise ValueError(f"dataset {self.name!r} has an annotation_split but no key spec")
        return self

    def iter_files(self) -> list[Path]:
        """Lists the dataset's JSONL files in a stable order.

        Returns:
            list[Path]: Sorted matching files. Sorted so that file ids assigned during
                the sidecar build stay the same across runs.
        """
        return sorted(self.jsonl_root.glob(self.glob))


class CorpusRegistry(BaseModel):
    """The set of datasets a blend is built from.

    Attributes:
        annotation_root (Optional[Path]): Directory holding the annotation parquet
            splits, i.e. the directory whose children are the split paths named by
            ``DatasetEntry.annotation_split``.
        extra_annotation_roots (list[Path]): Further directories searched for splits.
            Annotation shards are often spread over more than one cache.
        datasets (list[DatasetEntry]): The datasets themselves.
    """

    annotation_root: Optional[Path] = None
    extra_annotation_roots: list[Path] = Field(default_factory=list)
    datasets: Annotated[list[DatasetEntry], Field(min_length=1)]

    @model_validator(mode="after")
    def _check_unique_names(self) -> "CorpusRegistry":
        seen = set()
        for dataset in self.datasets:
            if dataset.name in seen:
                raise ValueError(f"duplicate dataset name {dataset.name!r} in registry")
            seen.add(dataset.name)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "CorpusRegistry":
        """Loads a registry from a YAML file.

        Args:
            path (Path): Path to the registry YAML.

        Returns:
            CorpusRegistry: The parsed registry.
        """
        with Path(path).open() as f:
            return cls.model_validate(yaml.safe_load(f))

    def get(self, name: str) -> DatasetEntry:
        """Looks a dataset up by name.

        Args:
            name (str): The dataset name.

        Returns:
            DatasetEntry: The matching entry.

        Raises:
            KeyError: If no dataset of that name is registered.
        """
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"no dataset named {name!r} in registry; known: {[d.name for d in self.datasets]}")

    def enabled_datasets(self) -> list[DatasetEntry]:
        """Lists the datasets taking part in the blend.

        Returns:
            list[DatasetEntry]: Entries whose ``enabled`` flag is set.
        """
        return [d for d in self.datasets if d.enabled]

    def annotation_shards(self, split: str) -> list[Path]:
        """Finds the parquet shards of an annotation split across all roots.

        Args:
            split (str): Split path relative to an annotation root.

        Returns:
            list[Path]: Sorted, de-duplicated shard paths. Empty if the split has not
                been downloaded, which is not an error: a split can be registered
                before its shards are fetched.
        """
        shards: set[Path] = set()
        for root in [self.annotation_root, *self.extra_annotation_roots]:
            if root is None:
                continue
            shards.update(Path(root).glob(f"{split}/*.parquet"))
        return sorted(shards)
