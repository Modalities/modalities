"""Turns a YAML selection into kept-document and kept-token figures.

A selection states, per dataset, which documents to keep and how heavily to sample what
remains. Predicates address two kinds of signal with the same syntax: ordinal
annotation labels such as ``educational_value``, and continuous native metrics such as
``fw_edu_scores``.

Every predicate can be evaluated two ways. Against a :class:`~...cube.Cube` it answers
in microseconds, which is what makes threshold tuning interactive. Against the
per-document sidecar it answers exactly but has to read the table. The two agree except
where a numeric threshold falls inside a cube bin rather than on its edge; the cube
evaluation detects that case and reports the result as approximate instead of pretending
otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field, model_validator

from modalities.dataloader.preprocessing.quality.cube import MISSING, Cube
from modalities.dataloader.preprocessing.quality.upsampling import (
    UNANNOTATED_BUCKET,
    QualityBucket,
    UpsamplingError,
    UpsamplingPlan,
    UpsamplingSpec,
    solve_curve,
)

# Ordinal scales, worst value first. Ordering is what gives `at_least` its meaning, so
# these are stated explicitly rather than inferred: `information_density` in particular
# orders `moderate` below `adequate`, which no alphabetical or intuitive rule reproduces.
ORDINAL_SCALES: dict[str, tuple[str, ...]] = {
    "educational_value": ("none", "minimal", "basic", "moderate", "high"),
    "content_quality": ("unacceptable", "poor", "adequate", "good", "excellent"),
    "information_density": ("empty", "thin", "moderate", "adequate", "dense"),
    "reasoning_indicators": ("none", "minimal", "basic_reasoning", "explanatory", "analytical"),
    "content_integrity": ("severely_degraded", "fragment", "mostly_complete", "complete"),
    "content_safety": ("illegal", "harmful", "nsfw", "mild_concerns", "safe"),
    "pii_presence": ("contains_pii", "no_pii"),
    "commercial_bias": ("pure_marketing", "heavy", "moderate", "minimal", "none"),
    "content_ratio": ("minimal_content", "mostly_navigation", "mixed_content", "mostly_content", "complete_content"),
    "content_length": ("minimal", "brief", "moderate", "substantial"),
}


class Op(str, Enum):
    """Comparison a predicate applies.

    Attributes:
        AT_LEAST: Ordinal label at or above a level.
        AT_MOST: Ordinal label at or below a level.
        IN: Label is one of a set.
        NOT_IN: Label is none of a set.
        GTE: Numeric metric greater than or equal to a value.
        LTE: Numeric metric less than or equal to a value.
        BETWEEN: Numeric metric within an inclusive range.
    """

    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    IN = "in"
    NOT_IN = "not_in"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"


class MissingPolicy(str, Enum):
    """What to do with documents a predicate cannot be evaluated on.

    Attributes:
        KEEP: Treat the predicate as satisfied. Right when annotations are only
            partly downloaded and dropping the unannotated majority would silently
            shrink the dataset.
        DROP: Treat the predicate as failed. Right when the filter is a hard
            requirement and an unannotated document cannot be shown to meet it.
    """

    KEEP = "keep"
    DROP = "drop"


class SelectionError(RuntimeError):
    """Raised when a selection is malformed or cannot be evaluated."""


class Predicate(BaseModel):
    """One condition a document must satisfy.

    Attributes:
        field (str): Annotation label name, or native metric name. Native metrics are
            named without the ``native_`` prefix the sidecar stores them under.
        op (Op): The comparison.
        value (Optional[Any]): Right-hand side for the single-valued comparisons.
        values (Optional[list[Any]]): Right-hand side for ``IN``/``NOT_IN``, or the
            two bounds of ``BETWEEN``.
        missing (Optional[MissingPolicy]): Overrides the selection-wide policy for
            this predicate.
    """

    field: str
    op: Op
    value: Optional[Any] = None
    values: Optional[list[Any]] = None
    missing: Optional[MissingPolicy] = None

    @model_validator(mode="after")
    def _check_operands(self) -> "Predicate":
        if self.op in (Op.AT_LEAST, Op.AT_MOST):
            if self.value is None:
                raise ValueError(f"{self.op.value} on {self.field!r} needs a 'value'")
            scale = ORDINAL_SCALES.get(self.field)
            if scale is None:
                raise ValueError(
                    f"{self.op.value} needs an ordinal field; {self.field!r} has no declared scale. "
                    f"Ordinal fields: {sorted(ORDINAL_SCALES)}"
                )
            if self.value not in scale:
                raise ValueError(f"{self.value!r} is not a level of {self.field!r}; levels are {list(scale)}")
        elif self.op in (Op.IN, Op.NOT_IN):
            if not self.values:
                raise ValueError(f"{self.op.value} on {self.field!r} needs a non-empty 'values'")
        elif self.op == Op.BETWEEN:
            if not self.values or len(self.values) != 2:
                raise ValueError(f"between on {self.field!r} needs 'values: [low, high]'")
        elif self.value is None:
            raise ValueError(f"{self.op.value} on {self.field!r} needs a 'value'")
        return self

    @property
    def is_numeric(self) -> bool:
        """Whether this predicate compares a continuous metric.

        Returns:
            bool: True for ``GTE``/``LTE``/``BETWEEN``.
        """
        return self.op in (Op.GTE, Op.LTE, Op.BETWEEN)

    def allowed_levels(self) -> set[str]:
        """The label values that satisfy this predicate.

        Returns:
            set[str]: Satisfying values for a categorical or ordinal predicate.

        Raises:
            SelectionError: If called on a numeric predicate.
        """
        if self.is_numeric:
            raise SelectionError(f"predicate on {self.field!r} is numeric and has no level set")
        if self.op == Op.AT_LEAST:
            scale = ORDINAL_SCALES[self.field]
            return set(scale[scale.index(self.value) :])
        if self.op == Op.AT_MOST:
            scale = ORDINAL_SCALES[self.field]
            return set(scale[: scale.index(self.value) + 1])
        if self.op == Op.IN:
            return {str(v) for v in self.values}
        scale = ORDINAL_SCALES.get(self.field)
        if scale is None:
            raise SelectionError(
                f"not_in on {self.field!r} needs a declared value set; use 'in' with the values to keep instead"
            )
        return set(scale) - {str(v) for v in self.values}

    def matches_value(self, value: Any, missing_policy: MissingPolicy) -> bool:
        """Evaluates the predicate against one document's value.

        Args:
            value (Any): The document's value for ``field``; None if absent.
            missing_policy (MissingPolicy): Fallback policy for this selection.

        Returns:
            bool: Whether the document satisfies the predicate.
        """
        policy = self.missing or missing_policy
        if value is None or value == MISSING or (isinstance(value, float) and np.isnan(value)):
            return policy == MissingPolicy.KEEP
        if self.is_numeric:
            numeric = float(value)
            if self.op == Op.GTE:
                return numeric >= float(self.value)
            if self.op == Op.LTE:
                return numeric <= float(self.value)
            return float(self.values[0]) <= numeric <= float(self.values[1])
        return str(value) in self.allowed_levels()

    def describe(self) -> str:
        """Renders the predicate for reports.

        Returns:
            str: A compact, readable form of the condition.
        """
        if self.op == Op.BETWEEN:
            return f"{self.field} in [{self.values[0]}, {self.values[1]}]"
        if self.op in (Op.IN, Op.NOT_IN):
            return f"{self.field} {self.op.value} {{{', '.join(str(v) for v in self.values)}}}"
        return f"{self.field} {self.op.value} {self.value}"


class DatasetSelection(BaseModel):
    """The rule applied to one dataset.

    Attributes:
        name (str): Dataset name, matching the corpus registry.
        ratio (float): Up/downsample factor applied after filtering. 1.0 uses the
            surviving documents once; 2.5 draws them two and a half times; 0.3 keeps
            three tenths of them.
        predicates (list[Predicate]): Conditions combined with AND. An empty list
            keeps every document, which is how an unannotated dataset participates.
        missing_annotation (Optional[MissingPolicy]): Overrides the config-wide policy.
        upsampling (Optional[UpsamplingSpec]): Replaces ``ratio`` with a quality-aware
            curve, so the repeat factor rises with quality instead of being one number for
            every surviving document. Mutually exclusive with a non-default ``ratio``.
        enabled (bool): Whether this dataset takes part.
    """

    name: str
    ratio: float = Field(default=1.0, ge=0.0)
    predicates: list[Predicate] = Field(default_factory=list)
    missing_annotation: Optional[MissingPolicy] = None
    upsampling: Optional[UpsamplingSpec] = None
    enabled: bool = True

    @model_validator(mode="after")
    def _check_ratio_or_curve(self) -> "DatasetSelection":
        # Silently ignoring one of them would make a config mean something other than it
        # reads, and the two express the same decision.
        if self.upsampling is not None and self.ratio != 1.0:
            raise ValueError(
                f"dataset {self.name!r} sets both 'ratio: {self.ratio}' and 'upsampling'; the curve "
                f"already determines how much is drawn. Remove the ratio."
            )
        if self.upsampling is not None and self.upsampling.quality_field not in ORDINAL_SCALES:
            # Checked at config load rather than mid-run: a numeric axis would need quantile
            # edges carried from the cube into materialisation, which is not built yet, and
            # discovering that after the preview succeeded would be worse than refusing now.
            raise ValueError(
                f"dataset {self.name!r}: upsampling needs an ordinal quality_field, and "
                f"{self.upsampling.quality_field!r} is not one. Available: {sorted(ORDINAL_SCALES)}"
            )
        return self


class SelectionConfig(BaseModel):
    """A complete blend specification.

    Attributes:
        missing_annotation (MissingPolicy): Default policy for documents that carry no
            annotation.
        target_tokens (Optional[float]): Tokens the training run will consume. It does not
            change any ratio, but it is what makes a ratio mean something: if the blend
            yields fewer effective tokens than this, the loader wraps and every document is
            seen more often than its ratio says.
        seed (int): Decides which documents receive the extra copy of a fractional repeat
            factor when the blend is exported. Changing it redraws that choice, so a blend
            exported twice with different seeds differs in which documents were doubled.
        max_total_exposure (Optional[float]): Refuse to materialise if any dataset -- or any
            quality bucket of one -- would be seen more times than this once wrapping is
            counted. A dataset with an upsampling curve uses its own ``max_factor`` instead,
            since that is already a declared cap.
        datasets (list[DatasetSelection]): Per-dataset rules.
    """

    missing_annotation: MissingPolicy = MissingPolicy.KEEP
    target_tokens: Optional[float] = None
    max_total_exposure: Optional[float] = Field(default=None, gt=0)
    seed: int = 42
    datasets: list[DatasetSelection]

    @model_validator(mode="after")
    def _check_unique(self) -> "SelectionConfig":
        seen = set()
        for dataset in self.datasets:
            if dataset.name in seen:
                raise ValueError(f"dataset {dataset.name!r} appears twice in the selection")
            seen.add(dataset.name)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "SelectionConfig":
        """Loads a selection from YAML.

        Args:
            path (Path): Path to the selection file.

        Returns:
            SelectionConfig: The parsed selection.
        """
        with Path(path).open() as f:
            return cls.model_validate(yaml.safe_load(f))

    def policy_for(self, dataset: DatasetSelection) -> MissingPolicy:
        """Resolves the missing-annotation policy for one dataset.

        Args:
            dataset (DatasetSelection): The dataset rule.

        Returns:
            MissingPolicy: The dataset's own policy if set, else the config default.
        """
        return dataset.missing_annotation or self.missing_annotation

    def enabled_datasets(self) -> list[DatasetSelection]:
        """Lists participating datasets.

        Returns:
            list[DatasetSelection]: Rules whose ``enabled`` flag is set.
        """
        return [d for d in self.datasets if d.enabled]


@dataclass
class DatasetResult:
    """What a selection costs for one dataset.

    Attributes:
        name (str): Dataset name.
        n_documents_total (int): Documents before filtering.
        n_documents_kept (int): Documents surviving the predicates.
        tokens_total (int): Estimated tokens before filtering.
        tokens_kept (int): Estimated tokens surviving the predicates.
        ratio (float): Up/downsample factor applied afterwards.
        exact (bool): Whether the figures are exact. False when a numeric threshold
            fell inside a cube bin, so the count had to be interpolated.
        approximations (list[str]): Predicates that forced interpolation.
        plan (Optional[UpsamplingPlan]): The solved quality curve, when the dataset uses one
            instead of a flat ratio.
    """

    name: str
    n_documents_total: int
    n_documents_kept: int
    tokens_total: int
    tokens_kept: int
    ratio: float
    exact: bool = True
    approximations: list[str] = field(default_factory=list)
    plan: Optional[UpsamplingPlan] = None

    @property
    def effective_tokens(self) -> float:
        """Tokens the blend draws from this dataset.

        Returns:
            float: Kept tokens scaled by the ratio, or what the curve draws.
        """
        if self.plan is not None:
            return self.plan.tokens_drawn
        return self.tokens_kept * self.ratio

    @property
    def ratio_label(self) -> str:
        """How the up/downsampling is described in a report.

        Returns:
            str: The flat ratio, or the curve's range of factors.
        """
        if self.plan is None:
            return f"{self.ratio:.2f}"
        factors = [b.factor for b in self.plan.buckets if b.factor > 0]
        if not factors:
            return "curve"
        return f"{min(factors):.2f}-{max(factors):.2f}x"

    @property
    def row_retention(self) -> float:
        """Share of documents kept.

        Returns:
            float: Kept over total documents; 0.0 for an empty dataset.
        """
        return self.n_documents_kept / self.n_documents_total if self.n_documents_total else 0.0

    @property
    def token_retention(self) -> float:
        """Share of tokens kept.

        Returns:
            float: Kept over total tokens; 0.0 for an empty dataset.

        Note:
            This is normally higher than :attr:`row_retention`, because quality
            correlates with length and quality filters keep the longer documents.
        """
        return self.tokens_kept / self.tokens_total if self.tokens_total else 0.0


@dataclass
class BlendResult:
    """What a selection costs across the whole blend.

    Attributes:
        datasets (list[DatasetResult]): Per-dataset outcomes.
        target_tokens (Optional[float]): Budget the blend aimed at, if any.
    """

    datasets: list[DatasetResult]
    target_tokens: Optional[float] = None

    @property
    def total_effective_tokens(self) -> float:
        """Tokens the blend yields in total.

        Returns:
            float: Sum of the per-dataset effective token counts.
        """
        return sum(d.effective_tokens for d in self.datasets)

    @property
    def exact(self) -> bool:
        """Whether every dataset's figures are exact.

        Returns:
            bool: True if no dataset needed interpolation.
        """
        return all(d.exact for d in self.datasets)

    def share_of(self, dataset: DatasetResult) -> float:
        """Share of the blend one dataset contributes.

        Args:
            dataset (DatasetResult): The dataset outcome.

        Returns:
            float: Its effective tokens over the blend total; 0.0 for an empty blend.
        """
        total = self.total_effective_tokens
        return dataset.effective_tokens / total if total else 0.0


def _bin_fraction_above(binning, threshold: float, bin_index: int) -> float:
    # Fraction of a bin lying at or above a threshold, assuming values are spread
    # evenly inside the bin. Only reached when the threshold splits a bin; on a bin
    # edge the caller takes the whole bin or none of it and stays exact.
    low = binning.lower_bound_of(bin_index)
    high = binning.upper_bound_of(bin_index)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.5
    return float(np.clip((high - threshold) / (high - low), 0.0, 1.0))


def quality_buckets_from_cube(
    cube: Cube, quality_field: str, weight: np.ndarray
) -> list[QualityBucket]:
    """Groups a cube's surviving cells into buckets ordered worst to best quality.

    Unannotated documents cannot be placed on a quality axis, so they form their own bucket
    at the bottom of the order. Putting them there is a choice worth knowing about: it means
    a dataset whose annotation coverage is partial will see its unannotated majority treated
    as lowest quality, and discarded first. The report names the bucket explicitly so this is
    visible rather than implied.

    Args:
        cube (Cube): The dataset's cube.
        quality_field (str): Ordinal label or native metric to order by.
        weight (np.ndarray): Per-cell survival weight from the predicates.

    Returns:
        list[QualityBucket]: Non-empty buckets, worst quality first.

    Raises:
        SelectionError: If the cube was not grouped on the field.
    """
    table = cube.table
    documents = table.column("n_documents").to_numpy(zero_copy_only=False).astype(np.float64) * weight
    tokens = table.column("n_tokens").to_numpy(zero_copy_only=False).astype(np.float64) * weight

    if quality_field not in cube.label_dimensions:
        raise SelectionError(
            f"cube for {cube.dataset!r} was not grouped on {quality_field!r}, so it cannot order "
            f"documents by it. Grouped labels: {cube.label_dimensions}. Rebuild the cube with "
            f"--label_dimension {quality_field}."
        )
    scale = ordered_quality_levels(quality_field)
    rank = {level: i for i, level in enumerate(scale)}
    values = table.column(quality_field).to_pylist()
    keys = np.array(
        [-1 if (v is None or v == MISSING) else rank.get(v, -1) for v in values], dtype=np.int64
    )
    order = sorted({int(k) for k in keys})
    labels = {k: (UNANNOTATED_BUCKET if k < 0 else scale[k]) for k in order}

    buckets: list[QualityBucket] = []
    for key in order:
        mask = keys == key
        n_tokens = int(round(float(tokens[mask].sum())))
        if n_tokens <= 0:
            continue
        buckets.append(
            QualityBucket(
                label=labels[key],
                n_documents=int(round(float(documents[mask].sum()))),
                n_tokens=n_tokens,
                unannotated=key < 0,
            )
        )
    return buckets


def ordered_bucket_labels(quality_field: str) -> list[str]:
    """Bucket labels for a quality field, worst first, unannotated at the bottom.

    Shared by the cube path, the exact sidecar path and materialisation: three places that
    must agree on the axis or a curve would mean different things depending on how it was
    evaluated.

    Args:
        quality_field (str): An ordinal annotation field.

    Returns:
        list[str]: ``UNANNOTATED_BUCKET`` followed by the field's levels in ascending order.
    """
    return [UNANNOTATED_BUCKET, *ordered_quality_levels(quality_field)]


def ordered_quality_levels(quality_field: str) -> tuple[str, ...]:
    """Lists a field's levels from worst to best quality.

    Args:
        quality_field (str): An ordinal annotation field.

    Returns:
        tuple[str, ...]: Its declared levels in ascending order.

    Raises:
        SelectionError: If the field has no declared ordinal scale, so its levels cannot be
            ordered and no curve over them would mean anything.
    """
    scale = ORDINAL_SCALES.get(quality_field)
    if scale is None:
        raise SelectionError(
            f"{quality_field!r} has no declared ordinal scale, so its levels cannot be ordered worst "
            f"to best. Ordinal fields: {sorted(ORDINAL_SCALES)}"
        )
    return scale


def _predicate_factor(
    cube: Cube, predicate: Predicate, policy: MissingPolicy
) -> tuple[np.ndarray, bool]:
    """Computes one predicate's surviving fraction for every cube cell.

    Extracted so a predicate can be costed on its own, which is what attributing retention
    to individual predicates needs, and not only as one term of a product.

    Args:
        cube (Cube): The dataset's cube.
        predicate (Predicate): The condition to apply.
        policy (MissingPolicy): Policy for cells with no annotation.

    Returns:
        tuple[np.ndarray, bool]: Per-cell factor in [0, 1], and whether this predicate was
            answered exactly rather than interpolated inside a bin.

    Raises:
        SelectionError: If the cube was not grouped on the predicate's field.
    """
    table = cube.table
    n_rows = table.num_rows
    exact = True

    if predicate.is_numeric:
        binning = cube.score_binnings.get(predicate.field)
        if binning is None:
            raise SelectionError(
                f"cube for {cube.dataset!r} was not grouped on native metric {predicate.field!r}; "
                f"grouped metrics: {sorted(cube.score_binnings)}. Re-run with --exact to scan the sidecar."
            )
        bins = table.column(f"native_{predicate.field}").to_numpy(zero_copy_only=False).astype(np.int64)
        factor = np.empty(n_rows, dtype=np.float64)
        for i, bin_index in enumerate(bins):
            if bin_index < 0:
                factor[i] = 1.0 if policy == MissingPolicy.KEEP else 0.0
                continue
            low = binning.lower_bound_of(bin_index)
            high = binning.upper_bound_of(bin_index)
            if predicate.op == Op.GTE:
                threshold = float(predicate.value)
                if low >= threshold:
                    factor[i] = 1.0
                elif high <= threshold:
                    factor[i] = 0.0
                else:
                    factor[i] = _bin_fraction_above(binning, threshold, bin_index)
            elif predicate.op == Op.LTE:
                threshold = float(predicate.value)
                if high <= threshold:
                    factor[i] = 1.0
                elif low >= threshold:
                    factor[i] = 0.0
                else:
                    factor[i] = 1.0 - _bin_fraction_above(binning, threshold, bin_index)
            else:
                lower, upper = float(predicate.values[0]), float(predicate.values[1])
                if low >= lower and high <= upper:
                    factor[i] = 1.0
                elif high <= lower or low >= upper:
                    factor[i] = 0.0
                else:
                    factor[i] = max(
                        0.0,
                        _bin_fraction_above(binning, lower, bin_index)
                        - _bin_fraction_above(binning, upper, bin_index),
                    )
            if 0.0 < factor[i] < 1.0:
                exact = False
        return factor, exact

    if predicate.field not in cube.label_dimensions:
        raise SelectionError(
            f"cube for {cube.dataset!r} was not grouped on label {predicate.field!r}; "
            f"grouped labels: {cube.label_dimensions}. Re-run with --exact to scan the sidecar."
        )
    allowed = predicate.allowed_levels()
    values = table.column(predicate.field).to_pylist()
    keep_missing = policy == MissingPolicy.KEEP
    factor = np.array(
        [
            (1.0 if keep_missing else 0.0) if v is None or v == MISSING else (1.0 if v in allowed else 0.0)
            for v in values
        ],
        dtype=np.float64,
    )
    return factor, True


def _cube_weights(
    cube: Cube, dataset: DatasetSelection, missing_policy: MissingPolicy
) -> tuple[np.ndarray, bool, list[str]]:
    """Computes each cube cell's surviving fraction under a dataset's predicates.

    Args:
        cube (Cube): The dataset's cube.
        dataset (DatasetSelection): The rule to apply.
        missing_policy (MissingPolicy): Policy for unannotated documents.

    Returns:
        tuple[np.ndarray, bool, list[str]]: Per-cell weight in [0, 1], whether every
            predicate was answered exactly, and the predicates that had to be interpolated.

    Raises:
        SelectionError: If a predicate names a field the cube was not grouped on.
    """
    weight = np.ones(cube.table.num_rows, dtype=np.float64)
    result_exact = True
    approximations: list[str] = []

    for predicate in dataset.predicates:
        factor, exact = _predicate_factor(cube, predicate, predicate.missing or missing_policy)
        # Per predicate, not cumulative. The previous form tested the running flag, so once
        # any predicate was interpolated every later one was reported as interpolated too.
        if not exact:
            result_exact = False
            if predicate.describe() not in approximations:
                approximations.append(predicate.describe())
        weight *= factor

    return weight, result_exact, approximations


@dataclass
class PredicateContribution:
    """What one predicate does to a dataset, on its own and in company.

    Attributes:
        description (str): The predicate, as written.
        matched_tokens (int): Tokens satisfying this predicate alone, ignoring the others.
        matched_documents (int): Documents satisfying it alone.
        marginal_tokens (int): Extra tokens the dataset would keep if this predicate were
            removed and the others left in place. This is the number that matters when
            tuning: a predicate whose marginal is zero is being fully shadowed by its
            neighbours and is only making the selection harder to read.
        exact (bool): Whether the predicate was answered exactly rather than interpolated.
    """

    description: str
    matched_tokens: int
    matched_documents: int
    marginal_tokens: int
    exact: bool = True


@dataclass
class PredicateBreakdown:
    """Per-predicate attribution for one dataset, plus how the predicates overlap.

    Attributes:
        dataset (str): Dataset name.
        total_tokens (int): Tokens before any predicate.
        kept_tokens (int): Tokens surviving all of them.
        contributions (list[PredicateContribution]): One per predicate, in config order.
        overlap_tokens (list[list[int]]): Symmetric matrix of tokens satisfying both
            predicates i and j; the diagonal is each predicate's own ``matched_tokens``.
            Where either predicate was interpolated, an off-diagonal cell multiplies two
            fractional survival rates and so assumes they are independent within a cube
            cell, which is the same assumption the combined weight already makes.
    """

    dataset: str
    total_tokens: int
    kept_tokens: int
    contributions: list[PredicateContribution]
    overlap_tokens: list[list[int]]

    def describe(self) -> str:
        """Renders the attribution as a small table.

        Returns:
            str: One row per predicate, plus an overlap matrix when there are at least two.
        """
        if not self.contributions:
            return f"  {self.dataset}: no predicates"

        def share(n: int) -> str:
            return f"{n / self.total_tokens:>6.1%}" if self.total_tokens else "     -"

        width = max(len(c.description) for c in self.contributions)
        lines = [
            f"  {self.dataset}: {self.kept_tokens:,} of {self.total_tokens:,} tokens kept",
            f"    {'predicate':<{width}} {'matches':>14} {'share':>7} {'marginal':>14}",
        ]
        for contribution in self.contributions:
            marker = "" if contribution.exact else " ~"
            lines.append(
                f"    {contribution.description:<{width}} {contribution.matched_tokens:>14,} "
                f"{share(contribution.matched_tokens)} {contribution.marginal_tokens:>14,}{marker}"
            )
        redundant = [c.description for c in self.contributions if c.marginal_tokens == 0]
        if redundant and len(self.contributions) > 1:
            lines.append(f"    no effect given the others: {', '.join(redundant)}")

        if len(self.contributions) >= 2:
            lines.append(f"    overlap (tokens matching both), {len(self.contributions)} predicates:")
            labels = [f"P{i + 1}" for i in range(len(self.contributions))]
            lines.append("      " + " ".join(f"{label:>14}" for label in [""] + labels))
            for i, label in enumerate(labels):
                cells = " ".join(f"{self.overlap_tokens[i][j]:>14,}" for j in range(len(labels)))
                lines.append(f"      {label:>14} {cells}")
            for i, contribution in enumerate(self.contributions):
                lines.append(f"      P{i + 1} = {contribution.description}")
        return "\n".join(lines)


def predicate_breakdown(
    cube: Cube, dataset: DatasetSelection, missing_policy: MissingPolicy
) -> PredicateBreakdown:
    """Attributes a dataset's retention to its individual predicates.

    The per-dataset retention a preview reports says nothing about which condition caused
    it. With two or three predicates per dataset, the interesting questions are which one
    binds, which is redundant, and how much they overlap -- and all of them are answerable
    from the cube in milliseconds, because the cell weights are already the whole
    computation.

    Args:
        cube (Cube): The dataset's cube.
        dataset (DatasetSelection): The rule to attribute.
        missing_policy (MissingPolicy): Policy for unannotated documents.

    Returns:
        PredicateBreakdown: Per-predicate matches, marginal effect, and pairwise overlap.

    Raises:
        SelectionError: If a predicate names a field the cube was not grouped on.
    """
    table = cube.table
    documents = table.column("n_documents").to_numpy(zero_copy_only=False).astype(np.float64)
    tokens = table.column("n_tokens").to_numpy(zero_copy_only=False).astype(np.float64)

    factors: list[np.ndarray] = []
    exactness: list[bool] = []
    for predicate in dataset.predicates:
        factor, exact = _predicate_factor(cube, predicate, predicate.missing or missing_policy)
        factors.append(factor)
        exactness.append(exact)

    combined = np.ones(table.num_rows, dtype=np.float64)
    for factor in factors:
        combined *= factor
    kept = float((tokens * combined).sum())

    contributions: list[PredicateContribution] = []
    for i, predicate in enumerate(dataset.predicates):
        # What the dataset would keep with predicate i lifted, everything else in place.
        without = np.ones(table.num_rows, dtype=np.float64)
        for j, factor in enumerate(factors):
            if j != i:
                without *= factor
        contributions.append(
            PredicateContribution(
                description=predicate.describe(),
                matched_tokens=int(round(float((tokens * factors[i]).sum()))),
                matched_documents=int(round(float((documents * factors[i]).sum()))),
                marginal_tokens=int(round(float((tokens * without).sum()) - kept)),
                exact=exactness[i],
            )
        )

    n = len(factors)
    overlap: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        for j in range(n):
            if i == j:
                # Not the product with itself: an interpolated predicate has fractional
                # factors, and squaring them undercounts. On real data this read 8.19 M
                # where the predicate matched 9.75 M.
                row.append(contributions[i].matched_tokens)
            else:
                row.append(int(round(float((tokens * factors[i] * factors[j]).sum()))))
        overlap.append(row)

    return PredicateBreakdown(
        dataset=dataset.name,
        total_tokens=int(round(float(tokens.sum()))),
        kept_tokens=int(round(kept)),
        contributions=contributions,
        overlap_tokens=overlap,
    )


def evaluate_on_cube(cube: Cube, dataset: DatasetSelection, missing_policy: MissingPolicy) -> DatasetResult:
    """Evaluates a dataset's rule against its cube.

    Args:
        cube (Cube): The dataset's cube.
        dataset (DatasetSelection): The rule to apply.
        missing_policy (MissingPolicy): Policy for unannotated documents.

    Returns:
        DatasetResult: Kept documents and tokens, flagged as exact or interpolated, plus the
            solved curve when the dataset uses one.

    Raises:
        SelectionError: If a predicate names a field the cube was not grouped on, or if a
            curve cannot be solved for the dataset.
    """
    weight, result_exact, approximations = _cube_weights(cube, dataset, missing_policy)
    table = cube.table
    documents = table.column("n_documents").to_numpy(zero_copy_only=False).astype(np.float64)
    tokens = table.column("n_tokens").to_numpy(zero_copy_only=False).astype(np.float64)

    plan: Optional[UpsamplingPlan] = None
    if dataset.upsampling is not None:
        buckets = quality_buckets_from_cube(cube, dataset.upsampling.quality_field, weight)
        try:
            plan = solve_curve(buckets, dataset.upsampling)
        except UpsamplingError as e:
            raise SelectionError(f"dataset {dataset.name!r}: {e}") from e

    return DatasetResult(
        name=dataset.name,
        n_documents_total=int(documents.sum()),
        n_documents_kept=plan.documents_kept if plan else int(round(float((documents * weight).sum()))),
        tokens_total=int(tokens.sum()),
        tokens_kept=int(round(float((tokens * weight).sum()))),
        ratio=dataset.ratio,
        exact=result_exact,
        approximations=approximations,
        plan=plan,
    )


def document_mask(table, dataset: DatasetSelection, missing_policy: MissingPolicy) -> np.ndarray:
    """Evaluates a dataset's rule against per-document sidecar rows.

    Args:
        table: A pyarrow table of sidecar rows.
        dataset (DatasetSelection): The rule to apply.
        missing_policy (MissingPolicy): Policy for unannotated documents.

    Returns:
        np.ndarray: Boolean mask, True where the document is kept.

    Raises:
        SelectionError: If a predicate names a column the sidecar does not have.
    """
    n_rows = table.num_rows
    mask = np.ones(n_rows, dtype=bool)
    names = set(table.schema.names)

    for predicate in dataset.predicates:
        column = f"native_{predicate.field}" if predicate.is_numeric else predicate.field
        if column not in names:
            raise SelectionError(
                f"sidecar has no column {column!r} for predicate on {predicate.field!r}; available: {sorted(names)}"
            )
        policy = predicate.missing or missing_policy
        if predicate.is_numeric:
            values = table.column(column).to_numpy(zero_copy_only=False).astype(np.float64)
            missing = np.isnan(values)
            with np.errstate(invalid="ignore"):
                if predicate.op == Op.GTE:
                    ok = values >= float(predicate.value)
                elif predicate.op == Op.LTE:
                    ok = values <= float(predicate.value)
                else:
                    ok = (values >= float(predicate.values[0])) & (values <= float(predicate.values[1]))
            ok = np.where(missing, policy == MissingPolicy.KEEP, ok)
        else:
            allowed = predicate.allowed_levels()
            keep_missing = policy == MissingPolicy.KEEP
            ok = np.array(
                [
                    keep_missing if v is None or v == MISSING else (v in allowed)
                    for v in table.column(column).to_pylist()
                ],
                dtype=bool,
            )
        mask &= ok
    return mask


def evaluate_on_sidecar(sidecar_dir: Path, dataset: DatasetSelection, missing_policy: MissingPolicy) -> DatasetResult:
    """Evaluates a dataset's rule exactly, by scanning its sidecar.

    Args:
        sidecar_dir (Path): Directory of sidecar parts.
        dataset (DatasetSelection): The rule to apply.
        missing_policy (MissingPolicy): Policy for unannotated documents.

    Returns:
        DatasetResult: Kept documents and tokens, always exact.

    Raises:
        SelectionError: If the directory holds no sidecar parts.
    """
    parts = sorted(Path(sidecar_dir).glob("part-*.parquet"))
    if not parts:
        raise SelectionError(f"no sidecar parts found in {sidecar_dir}")

    spec = dataset.upsampling
    # A curved dataset must be costed as a curve here too. Returning the flat ratio would
    # report it at 1.0x -- the value the config validator forces when a curve is present --
    # so an --exact preview of a curved selection would silently understate its tokens,
    # blend share and exposure.
    bucket_labels = ordered_bucket_labels(spec.quality_field) if spec else []
    bucket_docs: dict[str, int] = dict.fromkeys(bucket_labels, 0)
    bucket_tokens: dict[str, int] = dict.fromkeys(bucket_labels, 0)

    n_total = n_kept = tokens_total = tokens_kept = 0
    for part in parts:
        parquet_file = pq.ParquetFile(part)
        if spec and spec.quality_field not in parquet_file.schema_arrow.names:
            raise SelectionError(
                f"dataset {dataset.name!r}: sidecar has no column {spec.quality_field!r} to order "
                f"quality by; join the annotations before costing a curve"
            )
        for group_idx in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_idx)
            tokens = table.column("est_tokens").to_numpy(zero_copy_only=False).astype(np.int64)
            mask = document_mask(table, dataset, missing_policy)
            n_total += table.num_rows
            n_kept += int(mask.sum())
            tokens_total += int(tokens.sum())
            tokens_kept += int(tokens[mask].sum())

            if spec:
                levels = table.column(spec.quality_field).to_pylist()
                for level, keep, token_count in zip(levels, mask, tokens):
                    if not keep:
                        continue
                    label = level if level in bucket_docs else UNANNOTATED_BUCKET
                    bucket_docs[label] += 1
                    bucket_tokens[label] += int(token_count)

    plan: Optional[UpsamplingPlan] = None
    if spec:
        buckets = [
            QualityBucket(
                label=label,
                n_documents=bucket_docs[label],
                n_tokens=bucket_tokens[label],
                unannotated=label == UNANNOTATED_BUCKET,
            )
            for label in bucket_labels
            if bucket_tokens[label] > 0
        ]
        try:
            plan = solve_curve(buckets, spec)
        except UpsamplingError as e:
            raise SelectionError(f"dataset {dataset.name!r}: {e}") from e
        n_kept = plan.documents_kept

    return DatasetResult(
        name=dataset.name,
        n_documents_total=n_total,
        n_documents_kept=n_kept,
        tokens_total=tokens_total,
        tokens_kept=tokens_kept,
        ratio=dataset.ratio,
        exact=True,
        plan=plan,
    )


def evaluate_blend(
    config: SelectionConfig,
    cubes: dict[str, Cube],
    sidecar_dirs: Optional[dict[str, Path]] = None,
    force_exact: bool = False,
    allow_sidecar_fallback: bool = False,
) -> BlendResult:
    """Evaluates a whole selection.

    Args:
        config (SelectionConfig): The blend specification.
        cubes (dict[str, Cube]): Cube per dataset name.
        sidecar_dirs (Optional[dict[str, Path]]): Sidecar directory per dataset, used when
            a cube cannot answer a predicate or when exactness is demanded.
        force_exact (bool): Scan sidecars for every dataset instead of using cubes.
        allow_sidecar_fallback (bool): Permit scanning a sidecar for datasets whose cube
            cannot answer a predicate. Off by default, because that scan is thousands of
            times more expensive than a cube lookup and a `preview` is meant to return in
            seconds: a selection thresholding one ungrouped field turned a preview of this
            blend into a read over 1.7 billion documents.

    Returns:
        BlendResult: Per-dataset and total figures.

    Raises:
        SelectionError: If a dataset can be evaluated neither from a cube nor from a
            sidecar, or if a predicate needs a sidecar scan that was not permitted. Every
            such dataset is reported together, so one run tells you everything to fix.
    """
    results: list[DatasetResult] = []
    unanswerable: list[str] = []

    for dataset in config.enabled_datasets():
        policy = config.policy_for(dataset)
        sidecar_dir = (sidecar_dirs or {}).get(dataset.name)

        if force_exact:
            if sidecar_dir is None:
                raise SelectionError(f"exact evaluation of {dataset.name!r} needs a sidecar directory")
            results.append(evaluate_on_sidecar(sidecar_dir, dataset, policy))
            continue

        cube = cubes.get(dataset.name)
        if cube is None:
            if sidecar_dir is None:
                raise SelectionError(f"no cube and no sidecar for dataset {dataset.name!r}")
            if not allow_sidecar_fallback:
                unanswerable.append(f"  {dataset.name}: no cube was built; run 'quality build-cube'")
                continue
            results.append(evaluate_on_sidecar(sidecar_dir, dataset, policy))
            continue

        try:
            results.append(evaluate_on_cube(cube, dataset, policy))
        except SelectionError as e:
            if sidecar_dir is None:
                raise
            if not allow_sidecar_fallback:
                unanswerable.append(
                    f"  {dataset.name}: {e} "
                    f"(answering it from the sidecar means reading {cube.n_documents:,} documents)"
                )
                continue
            results.append(evaluate_on_sidecar(sidecar_dir, dataset, policy))

    if unanswerable:
        raise SelectionError(
            "these predicates cannot be answered from the cubes:\n"
            + "\n".join(unanswerable)
            + "\n\nEither drop or replace the offending predicate, rebuild the cube with that field as a "
            "dimension (build-cube --label_dimension ...), or accept the cost with --allow-fallback. "
            "A sidecar scan is exact but reads every document, so it takes minutes to hours rather than seconds."
        )

    return BlendResult(datasets=results, target_tokens=config.target_tokens)


@dataclass
class ExposureRow:
    """How often one dataset, or one quality bucket, is actually seen.

    Attributes:
        label (str): Dataset name, or ``dataset / bucket``.
        factor (float): The repeat factor the selection asked for.
        passes (float): Times the run traverses the whole blend.
        cap (Optional[float]): Declared limit, if any.
    """

    label: str
    factor: float
    passes: float
    cap: Optional[float] = None

    @property
    def exposure(self) -> float:
        """Times each document here is actually seen during the run.

        Returns:
            float: The requested factor multiplied by the number of passes.
        """
        return self.factor * self.passes

    @property
    def exceeded(self) -> bool:
        """Whether the declared cap is broken once wrapping is counted.

        Returns:
            bool: True if a cap exists and the exposure is above it.
        """
        return self.cap is not None and self.exposure > self.cap * (1.0 + 1e-9)


@dataclass
class ExposureReport:
    """What the run really repeats, as opposed to what the ratios say.

    A ratio is relative to one pass over the blend. If the run consumes more tokens than the
    blend yields, the loader comes round again and every factor is multiplied by the number
    of passes -- so a bucket set to 7x under a curve becomes 14x on a second pass. That is
    well past the point where repetition stops paying, and nothing in the pipeline noticed
    it before: ``target_tokens`` was only ever printed.

    Attributes:
        training_tokens (Optional[float]): Tokens the run consumes, if declared.
        effective_tokens (float): Tokens one pass over the blend yields.
        rows (list[ExposureRow]): One per dataset, or per bucket for curved datasets.
    """

    training_tokens: Optional[float]
    effective_tokens: float
    rows: list[ExposureRow]

    @property
    def passes(self) -> float:
        """Times the run traverses the blend.

        Returns:
            float: Training tokens over effective tokens; 1.0 when nothing was declared.
        """
        if not self.training_tokens or self.effective_tokens <= 0:
            return 1.0
        return self.training_tokens / self.effective_tokens

    @property
    def exceeded(self) -> list[ExposureRow]:
        """Rows whose exposure breaks their declared cap.

        Returns:
            list[ExposureRow]: Offending rows, worst first.
        """
        return sorted(
            (row for row in self.rows if row.exceeded), key=lambda r: r.exposure, reverse=True
        )

    def describe(self) -> str:
        """Renders the exposure accounting.

        Returns:
            str: A summary line, then the rows that repeat most, then any cap breaches.
        """
        lines = []
        if self.training_tokens:
            verb = "wraps" if self.passes > 1.0 else "uses"
            lines.append(
                f"  run consumes {self.training_tokens:,.0f} tokens from {self.effective_tokens:,.0f} "
                f"effective -- {verb} {self.passes:.2f} passes over the blend"
            )
        else:
            lines.append(
                "  no target_tokens declared, so repetition beyond one pass cannot be checked; "
                "set it to the tokens the run will consume"
            )
        top = sorted(self.rows, key=lambda r: r.exposure, reverse=True)[:6]
        if top:
            width = max(len(row.label) for row in top)
            lines.append(f"    {'':<{width}} {'factor':>8} {'exposure':>9} {'cap':>7}")
            for row in top:
                cap = f"{row.cap:g}" if row.cap is not None else "-"
                flag = "  OVER" if row.exceeded else ""
                lines.append(
                    f"    {row.label:<{width}} {row.factor:>8.2f} {row.exposure:>9.2f} {cap:>7}{flag}"
                )
        for row in self.exceeded:
            lines.append(
                f"    {row.label} is seen {row.exposure:.2f}x against a declared cap of {row.cap:g}x"
            )
        return "\n".join(lines)


def exposure_report(
    entries: list[tuple[str, float, Optional[float]]],
    effective_tokens: float,
    training_tokens: Optional[float],
) -> ExposureReport:
    """Builds the exposure accounting for a blend.

    Args:
        entries (list[tuple[str, float, Optional[float]]]): One ``(label, factor, cap)`` per
            dataset, or per quality bucket where a curve splits one.
        effective_tokens (float): Tokens one pass over the blend yields.
        training_tokens (Optional[float]): Tokens the run consumes.

    Returns:
        ExposureReport: The rows and the derived number of passes.
    """
    report = ExposureReport(
        training_tokens=training_tokens, effective_tokens=effective_tokens, rows=[]
    )
    passes = report.passes
    report.rows = [
        ExposureRow(label=label, factor=factor, passes=passes, cap=cap) for label, factor, cap in entries
    ]
    return report


def exposure_entries_from_blend(
    result: "BlendResult", config: SelectionConfig
) -> list[tuple[str, float, Optional[float]]]:
    """Collects exposure entries from an evaluated blend.

    Args:
        result (BlendResult): The evaluated blend.
        config (SelectionConfig): The selection, supplying caps.

    Returns:
        list[tuple[str, float, Optional[float]]]: One entry per dataset, or per bucket for a
            dataset with a curve.
    """
    by_name = {dataset.name: dataset for dataset in config.datasets}
    entries: list[tuple[str, float, Optional[float]]] = []
    for dataset_result in result.datasets:
        selection = by_name.get(dataset_result.name)
        spec = selection.upsampling if selection else None
        if dataset_result.plan is not None:
            for bucket in dataset_result.plan.buckets:
                if bucket.factor > 0:
                    entries.append(
                        (
                            f"{dataset_result.name} / {bucket.bucket.label}",
                            bucket.factor,
                            spec.max_factor if spec else None,
                        )
                    )
        else:
            entries.append((dataset_result.name, dataset_result.ratio, config.max_total_exposure))
    return entries


def format_blend_report(
    result: BlendResult,
    datasets_in_order: Optional[Iterable[str]] = None,
    config: Optional[SelectionConfig] = None,
) -> str:
    """Renders a blend result as a fixed-width table.

    Args:
        result (BlendResult): The evaluated blend.
        datasets_in_order (Optional[Iterable[str]]): Preferred row order by name.
        config (Optional[SelectionConfig]): The selection. Supplies the repetition caps, so
            the report can say how often documents are really seen; omitted, that block is
            left out.

    Returns:
        str: A table with per-dataset retention, ratio, effective tokens and share,
            followed by the blend total and any accuracy caveats.
    """

    def humanise(n: float) -> str:
        for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
            if abs(n) >= size:
                return f"{n / size:.2f}{unit}"
        return f"{n:.0f}"

    rows = list(result.datasets)
    if datasets_in_order is not None:
        order = {name: i for i, name in enumerate(datasets_in_order)}
        rows.sort(key=lambda d: order.get(d.name, len(order)))

    width = max([len(d.name) for d in rows] + [len("dataset")])
    header = (
        f"{'dataset':<{width}}  {'docs kept':>11}  {'row%':>6}  {'tokens kept':>12}  "
        f"{'tok%':>6}  {'ratio':>11}  {'effective':>12}  {'share':>6}"
    )
    lines = [header, "-" * len(header)]
    for d in rows:
        marker = "" if d.exact else " ~"
        lines.append(
            f"{d.name:<{width}}  {humanise(d.n_documents_kept):>11}  {d.row_retention:>5.1%}  "
            f"{humanise(d.tokens_kept):>12}  {d.token_retention:>5.1%}  {d.ratio_label:>11}  "
            f"{humanise(d.effective_tokens):>12}{marker:<2}  {result.share_of(d):>5.1%}"
        )
    lines.append("-" * len(header))
    total = result.total_effective_tokens
    lines.append(
        f"{'TOTAL':<{width}}  {'':>11}  {'':>6}  {'':>12}  {'':>6}  {'':>11}  {humanise(total):>12}  {1.0:>5.1%}"
    )

    exposure = exposure_report(
        exposure_entries_from_blend(result, config) if config is not None else [],
        effective_tokens=total,
        training_tokens=result.target_tokens,
    )
    if config is not None:
        lines.append("\nrepetition, once wrapping is counted")
        lines.append(exposure.describe())

    curved = [d for d in rows if d.plan is not None]
    if curved:
        lines.append("\nquality-aware upsampling curves")
        for d in curved:
            lines.append(f"\n  {d.name}")
            lines.append(d.plan.describe())

    if result.target_tokens:
        gap = total - result.target_tokens
        verb = "over" if gap >= 0 else "under"
        lines.append(
            f"\ntarget {humanise(result.target_tokens)} tokens -- {humanise(abs(gap))} {verb} "
            f"({abs(gap) / result.target_tokens:.1%})"
        )
    if not result.exact:
        lines.append("\n~ interpolated: a threshold fell inside a cube bin rather than on an edge.")
        for d in rows:
            for approximation in d.approximations:
                lines.append(f"    {d.name}: {approximation}")
        lines.append("  Re-run with --exact to scan the per-document sidecar instead.")
    return "\n".join(lines)
