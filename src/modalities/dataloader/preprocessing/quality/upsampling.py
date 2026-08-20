"""Turns one scalar up/downsample ratio per dataset into a quality-aware upsampling curve.

A single ratio treats every surviving document alike: a dataset filtered to "educational
value at least basic" and set to 1.2 draws the barely-basic documents exactly as often as
the excellent ones. A curve instead makes the repeat factor rise with quality, so the token
budget is spent where the quality signal says it is worth spending.

The idea and the functional form are from Dolma 3 / Olmo 3 (arXiv:2512.13961, §3.4.4 and
appendix A.2.4), which reports it beating flat quality filtering at every matched repetition
factor -- for instance 0.740 against 0.843-0.870 bits-per-byte on their Math suite. They
discard the bottom 40% of web text by quality and repeat the top 5% seven times.

**The parameterisation.** Quality is placed on a [0, 1] axis, ordered worst to best, where a
bucket's width is its share of the dataset's tokens. The repeat factor is

    f(x) = 0                     for x < a
    f(x) = C * (x - a)**p        for x >= a

subject to three constraints, following the paper: the integral equals the target token
yield, no bucket averages more than ``max_factor``, and the curve is monotone.

The paper's family carries an extra ``exp(lam * (x - a))`` factor. We fix ``lam = 0``, which
is a deliberate simplification: with two shape parameters the constraints define a curve of
feasible solutions rather than a point, so something further has to pick one. Dropping the
exponential makes the solution unique *and* every integral analytic, so no numerical
quadrature is involved. What is left is one degree of freedom, ``p``, and we spend it by
pushing the top bucket to exactly ``max_factor`` -- the steepest admissible curve, which is
also what the paper's own figure shows.

**Why the cumulative form.** Writing ``q = p + 1`` and

    g(t) = ((t - a) / (1 - a))**q   for t >= a, else 0

makes ``g`` the share of the token budget drawn from below quality ``t``, with ``g(1) = 1``.
The scale ``C`` cancels, so a bucket spanning ``[u, v]`` receives ``Z * (g(v) - g(u))``
tokens and is repeated ``R * (g(v) - g(u)) / (v - u)`` times, where ``R = Z / X``. Every
quantity below is that expression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, model_validator

# Ceiling on the exponent, and it earns its keep in the downsampling regime. Pushing the
# curve as steep as the cap allows is right when the cap binds -- it is what reproduces the
# published example, bottom 40% discarded and top 5% at 7x. But when the target is a small
# multiple of the pool, as it is whenever a blend draws far fewer tokens than it has, no
# exponent violates the cap and "steepest" is unbounded: the budget collapses onto the
# highest bucket and the curve degenerates into the hard top-k filtering it exists to beat.
# A modest ceiling keeps it a curve. Below it, nothing changes.
MAX_EXPONENT = 8.0

# Label for documents that carry no annotation and so cannot be placed on a quality axis.
UNANNOTATED_BUCKET = "<unannotated>"


class UpsamplingError(RuntimeError):
    """Raised when no curve can satisfy the requested constraints."""


class UpsamplingSpec(BaseModel):
    """How to build a quality-aware curve for one dataset.

    Attributes:
        quality_field (str): The field that orders documents from worst to best. Either an
            ordinal annotation field, whose declared levels become the buckets, or a native
            numeric metric, whose cube score bins become the buckets.
        target_tokens (Optional[float]): Tokens to draw from this dataset in total. Exactly
            one of this and ``target_ratio`` must be given.
        target_ratio (Optional[float]): Tokens to draw as a multiple of the dataset's kept
            tokens -- the average repeat factor over the whole dataset, discarded part
            included. 1.0 means "as many tokens as the dataset has".
        max_factor (float): No bucket may be repeated more than this on average. 7.0 is the
            value Dolma 3 arrived at empirically; beyond a handful of repeats the returns
            fall off sharply.
        discard_below_percentile (float): Share of tokens, from the bottom of the quality
            order, to drop entirely. This is on top of the dataset's predicates, which have
            already been applied: the percentile is of what survived them.
        exponent (Optional[float]): Fixes the curve's shape instead of solving for it. For
            experiments; 1.0 is flat over the kept range, larger is steeper.
    """

    quality_field: str
    target_tokens: Optional[float] = Field(default=None, gt=0)
    target_ratio: Optional[float] = Field(default=None, gt=0)
    max_factor: float = Field(default=7.0, gt=0)
    discard_below_percentile: float = Field(default=0.0, ge=0.0, lt=100.0)
    exponent: Optional[float] = Field(default=None, ge=1.0)

    @model_validator(mode="after")
    def _check_one_target(self) -> "UpsamplingSpec":
        if (self.target_tokens is None) == (self.target_ratio is None):
            raise ValueError(
                f"upsampling on {self.quality_field!r} needs exactly one of 'target_tokens' or "
                f"'target_ratio'"
            )
        return self

    @property
    def discard_fraction(self) -> float:
        """The discard threshold as a fraction rather than a percentage.

        Returns:
            float: Value in [0, 1).
        """
        return self.discard_below_percentile / 100.0


@dataclass(frozen=True)
class QualityBucket:
    """One quality level of a dataset, after its predicates have been applied.

    Attributes:
        label (str): The level, or ``UNANNOTATED_BUCKET``.
        n_documents (int): Documents in this bucket.
        n_tokens (int): Estimated tokens in this bucket.
        unannotated (bool): Whether this is the bucket of documents with no label, which
            cannot be placed on the quality axis and so sit at the bottom of it.
    """

    label: str
    n_documents: int
    n_tokens: int
    unannotated: bool = False


@dataclass(frozen=True)
class BucketPlan:
    """What the curve prescribes for one bucket.

    Attributes:
        bucket (QualityBucket): The bucket itself.
        lower (float): Start of its interval on the quality axis.
        upper (float): End of its interval on the quality axis.
        factor (float): Repeat factor. 0.0 means discarded.
        tokens_drawn (float): Tokens the blend takes from this bucket.
    """

    bucket: QualityBucket
    lower: float
    upper: float
    factor: float
    tokens_drawn: float


@dataclass(frozen=True)
class UpsamplingPlan:
    """A solved curve together with its per-bucket consequences.

    Attributes:
        field (str): The quality field the axis was built from.
        exponent (float): The solved ``q = p + 1``. 1.0 is flat over the kept range, and
            :data:`MAX_EXPONENT` means the cap never bound, so the shape came from the
            ceiling rather than from the constraints.
        discard_fraction (float): Share of tokens dropped from the bottom.
        max_factor (float): The cap that was honoured.
        target_ratio (float): Tokens drawn over tokens available.
        buckets (list[BucketPlan]): Ordered worst to best quality.
        saturated (bool): Whether the top bucket reached ``max_factor``. False means the
            target was reachable without needing the steepest curve.
    """

    field: str
    exponent: float
    discard_fraction: float
    max_factor: float
    target_ratio: float
    buckets: list[BucketPlan]
    saturated: bool

    @property
    def tokens_available(self) -> int:
        """Tokens in the dataset after its predicates, before the curve.

        Returns:
            int: Sum over all buckets.
        """
        return sum(plan.bucket.n_tokens for plan in self.buckets)

    @property
    def tokens_drawn(self) -> float:
        """Tokens the blend takes from this dataset.

        Returns:
            float: Sum over buckets of tokens drawn.
        """
        return sum(plan.tokens_drawn for plan in self.buckets)

    @property
    def documents_kept(self) -> int:
        """Distinct documents that survive the curve's discard threshold.

        Returns:
            int: Documents in buckets with a non-zero factor. Repetition does not
                multiply this: the same documents are drawn more than once.
        """
        return sum(plan.bucket.n_documents for plan in self.buckets if plan.factor > 0)

    def describe(self) -> str:
        """Renders the curve as a small table.

        Returns:
            str: One line per bucket, worst quality first.
        """
        lines = [
            f"  curve on {self.field}: exponent {self.exponent:.2f}, "
            f"discard bottom {self.discard_fraction:.0%}, cap {self.max_factor:g}x"
            + ("" if self.saturated else "  (cap not reached)"),
            f"    {'bucket':<22} {'tokens':>14} {'share':>7} {'factor':>8} {'drawn':>14}",
        ]
        for plan in self.buckets:
            share = plan.upper - plan.lower
            factor = "discarded" if plan.factor == 0 else f"{plan.factor:.2f}x"
            lines.append(
                f"    {plan.bucket.label:<22} {plan.bucket.n_tokens:>14,} {share:>6.1%} "
                f"{factor:>8} {plan.tokens_drawn:>14,.0f}"
            )
        return "\n".join(lines)


def _cumulative(exponent: float, discard: float, t: float) -> float:
    """Share of the token budget drawn from quality below ``t``.

    Args:
        exponent (float): ``q = p + 1``, at least 1.
        discard (float): The discard threshold ``a``.
        t (float): Point on the quality axis.

    Returns:
        float: Value in [0, 1].
    """
    if t <= discard:
        return 0.0
    if discard >= 1.0:
        return 0.0
    return min(1.0, ((t - discard) / (1.0 - discard)) ** exponent)


def _edges(buckets: list[QualityBucket]) -> list[tuple[float, float]]:
    """Places buckets on the [0, 1] quality axis, weighted by tokens.

    Args:
        buckets (list[QualityBucket]): Ordered worst to best quality.

    Returns:
        list[tuple[float, float]]: One (lower, upper) per bucket.

    Raises:
        UpsamplingError: If the buckets hold no tokens, so no axis can be built.
    """
    total = sum(b.n_tokens for b in buckets)
    if total <= 0:
        raise UpsamplingError("cannot build a quality axis: the surviving buckets hold no tokens")
    edges: list[tuple[float, float]] = []
    cursor = 0.0
    for bucket in buckets:
        width = bucket.n_tokens / total
        edges.append((cursor, min(1.0, cursor + width)))
        cursor += width
    # Absorb rounding so the last bucket ends exactly at 1.0.
    if edges:
        edges[-1] = (edges[-1][0], 1.0)
    return edges


def _top_factor(exponent: float, discard: float, target_ratio: float, edges: list[tuple[float, float]]) -> float:
    """Repeat factor of the highest-quality bucket for a given exponent.

    Args:
        exponent (float): ``q = p + 1``.
        discard (float): The discard threshold.
        target_ratio (float): ``Z / X``.
        edges (list[tuple[float, float]]): Bucket intervals.

    Returns:
        float: The top bucket's average repeat factor.
    """
    lower, upper = edges[-1]
    width = upper - lower
    if width <= 0:
        return float("inf")
    drawn = _cumulative(exponent, discard, upper) - _cumulative(exponent, discard, lower)
    return target_ratio * drawn / width


def solve_curve(
    buckets: list[QualityBucket],
    spec: UpsamplingSpec,
) -> UpsamplingPlan:
    """Finds the steepest curve satisfying the spec, and its per-bucket factors.

    The exponent is chosen so the top bucket sits exactly at ``max_factor``. That is the
    steepest admissible curve, which is the point of the exercise: a flatter one would spend
    budget on weaker documents while leaving the cap unused. If even a flat curve over the
    kept range exceeds the cap the request is infeasible, and if the cap cannot be reached
    however steep the curve, the exponent is capped instead.

    Args:
        buckets (list[QualityBucket]): Buckets ordered worst to best quality, holding the
            documents that survived the dataset's predicates.
        spec (UpsamplingSpec): The constraints.

    Returns:
        UpsamplingPlan: The solved curve and what it draws from each bucket.

    Raises:
        UpsamplingError: If no curve can meet the constraints, with the numbers needed to
            see which constraint to relax.
    """
    if not buckets:
        raise UpsamplingError(f"no quality buckets for field {spec.quality_field!r}")

    edges = _edges(buckets)
    available = sum(b.n_tokens for b in buckets)
    discard = spec.discard_fraction
    kept_share = 1.0 - discard

    if spec.target_ratio is not None:
        target_ratio = spec.target_ratio
    else:
        target_ratio = spec.target_tokens / available

    # A flat curve over the kept range repeats everything target_ratio / kept_share times,
    # which is the least any admissible curve can ask of its top bucket.
    flat_factor = target_ratio / kept_share
    if flat_factor > spec.max_factor + 1e-9:
        raise UpsamplingError(
            f"{spec.quality_field!r}: cannot draw {target_ratio:.2f}x this dataset's "
            f"{available:,} tokens while discarding the bottom {discard:.0%} and repeating "
            f"nothing more than {spec.max_factor:g}x -- even repeating every kept document "
            f"equally needs {flat_factor:.2f}x. Discard less, raise max_factor, or lower the "
            f"target."
        )

    if spec.exponent is not None:
        exponent = spec.exponent
        saturated = False
    else:
        # _top_factor rises monotonically with the exponent, from flat_factor towards
        # target_ratio / top_width, so bisect for the cap.
        low, high = 1.0, MAX_EXPONENT
        if _top_factor(high, discard, target_ratio, edges) <= spec.max_factor:
            # The cap does not bind anywhere in the admissible range, so there is no
            # constraint left to pin the shape; take the steepest curve the ceiling allows.
            exponent, saturated = high, False
        else:
            for _ in range(200):
                mid = (low + high) / 2.0
                if _top_factor(mid, discard, target_ratio, edges) > spec.max_factor:
                    high = mid
                else:
                    low = mid
            exponent, saturated = (low + high) / 2.0, True

    plans: list[BucketPlan] = []
    for bucket, (lower, upper) in zip(buckets, edges):
        drawn_share = _cumulative(exponent, discard, upper) - _cumulative(exponent, discard, lower)
        tokens_drawn = target_ratio * drawn_share * available
        factor = (tokens_drawn / bucket.n_tokens) if bucket.n_tokens > 0 else 0.0
        # A bucket entirely below the discard threshold draws nothing; float noise can leave
        # a vanishing factor, which would materialise an index for no reason.
        if factor < 1e-9:
            factor, tokens_drawn = 0.0, 0.0
        plans.append(
            BucketPlan(
                bucket=bucket, lower=lower, upper=upper, factor=factor, tokens_drawn=tokens_drawn
            )
        )

    worst = max((p.factor for p in plans), default=0.0)
    if worst > spec.max_factor * (1.0 + 1e-6):
        raise UpsamplingError(
            f"{spec.quality_field!r}: solved curve repeats a bucket {worst:.2f}x, above the "
            f"{spec.max_factor:g}x cap. This happens when one bucket is much narrower than the "
            f"others; widen the buckets or raise max_factor."
        )

    return UpsamplingPlan(
        field=spec.quality_field,
        exponent=exponent,
        discard_fraction=discard,
        max_factor=spec.max_factor,
        target_ratio=target_ratio,
        buckets=plans,
        saturated=saturated,
    )
