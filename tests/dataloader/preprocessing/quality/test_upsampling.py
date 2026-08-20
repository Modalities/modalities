"""Tests for quality-aware upsampling curves.

The behaviour worth pinning is not "it produces numbers" but the four properties that make
the curve mean anything: the token target is hit, the repeat cap is honoured, the factors
rise with quality, and the bottom of the distribution is actually dropped. Everything here
is synthetic and runs in milliseconds -- no corpus, no cluster.
"""

import pytest

from modalities.dataloader.preprocessing.quality.upsampling import (
    UNANNOTATED_BUCKET,
    QualityBucket,
    UpsamplingError,
    UpsamplingSpec,
    solve_curve,
)


def vigintiles(tokens_each: int = 1_000_000) -> list[QualityBucket]:
    """Twenty equal-token buckets, as Dolma 3 partitions web text."""
    return [
        QualityBucket(label=f"p{5 * i}-{5 * (i + 1)}", n_documents=1000, n_tokens=tokens_each)
        for i in range(20)
    ]


def test_the_token_target_is_hit():
    buckets = vigintiles()
    available = sum(b.n_tokens for b in buckets)
    plan = solve_curve(buckets, UpsamplingSpec(quality_field="q", target_tokens=available * 1.5))
    assert plan.tokens_drawn == pytest.approx(available * 1.5, rel=1e-6)


def test_target_ratio_and_target_tokens_agree():
    buckets = vigintiles()
    available = sum(b.n_tokens for b in buckets)
    by_tokens = solve_curve(buckets, UpsamplingSpec(quality_field="q", target_tokens=available * 2))
    by_ratio = solve_curve(buckets, UpsamplingSpec(quality_field="q", target_ratio=2.0))
    assert by_tokens.exponent == pytest.approx(by_ratio.exponent)
    assert by_tokens.tokens_drawn == pytest.approx(by_ratio.tokens_drawn)


def test_the_cap_is_honoured_and_reached():
    plan = solve_curve(
        vigintiles(),
        UpsamplingSpec(quality_field="q", target_ratio=1.0, max_factor=7.0, discard_below_percentile=40.0),
    )
    factors = [b.factor for b in plan.buckets]
    assert max(factors) == pytest.approx(7.0, rel=1e-4)
    assert plan.saturated


def test_factors_rise_with_quality():
    plan = solve_curve(vigintiles(), UpsamplingSpec(quality_field="q", target_ratio=1.0, max_factor=7.0))
    factors = [b.factor for b in plan.buckets]
    assert factors == sorted(factors), "a curve whose factors are not monotone is not quality-aware"


def test_the_discarded_share_is_actually_discarded():
    plan = solve_curve(
        vigintiles(),
        UpsamplingSpec(quality_field="q", target_ratio=1.0, max_factor=7.0, discard_below_percentile=40.0),
    )
    # Eight of twenty equal-token buckets sit below the 40th percentile.
    assert [b.factor for b in plan.buckets[:8]] == [0.0] * 8
    assert plan.documents_kept == sum(b.bucket.n_documents for b in plan.buckets[8:] if b.factor > 0)


def test_an_impossible_target_is_refused_with_the_numbers_to_fix_it():
    # Asking for 5x the data while discarding half of it and capping repeats at 2x cannot work:
    # the kept half would have to be repeated 10x.
    with pytest.raises(UpsamplingError, match="even repeating every kept document equally"):
        solve_curve(
            vigintiles(),
            UpsamplingSpec(
                quality_field="q", target_ratio=5.0, max_factor=2.0, discard_below_percentile=50.0
            ),
        )


def test_a_reachable_target_does_not_saturate_the_cap():
    # Drawing only a third of the data needs no repetition at all, so the cap stays unused.
    plan = solve_curve(
        vigintiles(), UpsamplingSpec(quality_field="q", target_ratio=0.33, max_factor=7.0)
    )
    assert max(b.factor for b in plan.buckets) < 7.0


def test_a_flat_exponent_reproduces_flat_filtering():
    plan = solve_curve(
        vigintiles(),
        UpsamplingSpec(
            quality_field="q", target_ratio=0.6, discard_below_percentile=40.0, exponent=1.0
        ),
    )
    kept = [b.factor for b in plan.buckets if b.factor > 0]
    assert len(kept) == 12
    # Flat over the kept range: every surviving bucket repeated the same amount.
    assert max(kept) == pytest.approx(min(kept), rel=1e-6)
    assert max(kept) == pytest.approx(0.6 / 0.6, rel=1e-6)


def test_unequal_bucket_widths_are_weighted_by_tokens():
    # Ordinal levels are not equal-sized the way vigintiles are; a level holding most of the
    # tokens must occupy most of the axis, or the curve prices it wrongly.
    buckets = [
        QualityBucket(label="none", n_documents=10, n_tokens=100),
        QualityBucket(label="basic", n_documents=900, n_tokens=9_000),
        QualityBucket(label="excellent", n_documents=90, n_tokens=900),
    ]
    plan = solve_curve(buckets, UpsamplingSpec(quality_field="q", target_ratio=1.0, max_factor=7.0))
    widths = [b.upper - b.lower for b in plan.buckets]
    assert widths[1] == pytest.approx(0.9, abs=1e-9)
    assert plan.tokens_drawn == pytest.approx(10_000, rel=1e-6)


def test_unannotated_documents_sit_at_the_bottom():
    buckets = [
        QualityBucket(label=UNANNOTATED_BUCKET, n_documents=50, n_tokens=500, unannotated=True),
        QualityBucket(label="basic", n_documents=50, n_tokens=500),
        QualityBucket(label="excellent", n_documents=50, n_tokens=500),
    ]
    plan = solve_curve(buckets, UpsamplingSpec(quality_field="q", target_ratio=1.0, max_factor=7.0))
    assert plan.buckets[0].bucket.label == UNANNOTATED_BUCKET
    assert plan.buckets[0].factor <= plan.buckets[-1].factor


def test_empty_buckets_are_refused():
    with pytest.raises(UpsamplingError, match="no quality buckets"):
        solve_curve([], UpsamplingSpec(quality_field="q", target_ratio=1.0))
    with pytest.raises(UpsamplingError, match="hold no tokens"):
        solve_curve(
            [QualityBucket(label="a", n_documents=0, n_tokens=0)],
            UpsamplingSpec(quality_field="q", target_ratio=1.0),
        )


def test_a_spec_needs_exactly_one_target():
    with pytest.raises(ValueError, match="exactly one of"):
        UpsamplingSpec(quality_field="q")
    with pytest.raises(ValueError, match="exactly one of"):
        UpsamplingSpec(quality_field="q", target_tokens=1, target_ratio=1.0)
