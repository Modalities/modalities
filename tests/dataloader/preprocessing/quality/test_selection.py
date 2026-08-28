from pathlib import Path

import pyarrow as pa
import pytest

from modalities.dataloader.preprocessing.quality.cube import MISSING, Cube, ScoreBinning
from modalities.dataloader.preprocessing.quality.selection import (
    ORDINAL_SCALES,
    DatasetSelection,
    MissingPolicy,
    Op,
    Predicate,
    SelectionConfig,
    SelectionError,
    document_mask,
    evaluate_on_cube,
    format_blend_report,
)


def test_at_least_uses_the_declared_order_not_the_alphabet():
    predicate = Predicate(field="information_density", op=Op.AT_LEAST, value="adequate")

    # `moderate` sorts before `adequate` alphabetically but ranks below it on the scale,
    # so an alphabetical implementation would wrongly include it.
    assert predicate.allowed_levels() == {"adequate", "dense"}


def test_at_most_includes_everything_up_to_the_level():
    predicate = Predicate(field="content_integrity", op=Op.AT_MOST, value="fragment")

    assert predicate.allowed_levels() == {"severely_degraded", "fragment"}


def test_ordinal_predicate_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="is not a level of"):
        Predicate(field="educational_value", op=Op.AT_LEAST, value="excellent")


def test_ordinal_predicate_rejects_a_field_without_a_scale():
    with pytest.raises(ValueError, match="no declared scale"):
        Predicate(field="fw_edu", op=Op.AT_LEAST, value="high")


def test_every_declared_scale_has_unique_levels():
    for field, scale in ORDINAL_SCALES.items():
        assert len(set(scale)) == len(scale), f"{field} has duplicate levels"


@pytest.mark.parametrize("policy,expected", [(MissingPolicy.KEEP, True), (MissingPolicy.DROP, False)])
def test_missing_value_follows_the_policy(policy, expected):
    predicate = Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")

    assert predicate.matches_value(None, policy) is expected
    assert predicate.matches_value(MISSING, policy) is expected


def test_per_predicate_policy_overrides_the_selection_policy():
    predicate = Predicate(field="educational_value", op=Op.AT_LEAST, value="basic", missing=MissingPolicy.DROP)

    assert predicate.matches_value(None, MissingPolicy.KEEP) is False


def _cube_with_labels() -> Cube:
    table = pa.table(
        {
            "educational_value": ["high", "basic", "none", MISSING],
            "n_documents": [10, 20, 30, 40],
            "n_tokens": [1000, 1500, 900, 2000],
        }
    )
    return Cube(
        dataset="toy",
        label_dimensions=["educational_value"],
        score_binnings={},
        table=table,
        n_documents=100,
        n_tokens=5400,
    )


def test_cube_evaluation_sums_the_matching_cells():
    result = evaluate_on_cube(
        _cube_with_labels(),
        DatasetSelection(name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")]),
        MissingPolicy.DROP,
    )

    assert result.n_documents_kept == 30
    assert result.tokens_kept == 2500
    assert result.exact


def test_cube_evaluation_keeps_unannotated_cells_when_told_to():
    result = evaluate_on_cube(
        _cube_with_labels(),
        DatasetSelection(name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")]),
        MissingPolicy.KEEP,
    )

    assert result.n_documents_kept == 70
    assert result.tokens_kept == 4500


def test_cube_evaluation_rejects_an_ungrouped_field():
    with pytest.raises(SelectionError, match="not grouped on"):
        evaluate_on_cube(
            _cube_with_labels(),
            DatasetSelection(name="toy", predicates=[Predicate(field="content_quality", op=Op.AT_LEAST, value="good")]),
            MissingPolicy.KEEP,
        )


def _cube_with_scores() -> Cube:
    # Two bins with edges at 0, 2 and 4.
    table = pa.table({"native_score": [0, 1], "n_documents": [100, 100], "n_tokens": [1000, 3000]})
    return Cube(
        dataset="toy",
        label_dimensions=[],
        score_binnings={"score": ScoreBinning(column="native_score", edges=(0.0, 2.0, 4.0))},
        table=table,
        n_documents=200,
        n_tokens=4000,
    )


def test_numeric_threshold_on_a_bin_edge_is_exact():
    result = evaluate_on_cube(
        _cube_with_scores(),
        DatasetSelection(name="toy", predicates=[Predicate(field="score", op=Op.GTE, value=2.0)]),
        MissingPolicy.KEEP,
    )

    assert result.exact
    assert result.n_documents_kept == 100
    assert result.tokens_kept == 3000


def test_numeric_threshold_inside_a_bin_is_reported_as_interpolated():
    result = evaluate_on_cube(
        _cube_with_scores(),
        DatasetSelection(name="toy", predicates=[Predicate(field="score", op=Op.GTE, value=3.0)]),
        MissingPolicy.KEEP,
    )

    assert not result.exact, "a threshold splitting a bin cannot be answered exactly from a cube"
    assert result.approximations == ["score gte 3.0"]
    # Half of the upper bin, none of the lower one.
    assert result.n_documents_kept == 50


def test_document_mask_matches_cube_counts():
    table = pa.table(
        {
            "educational_value": ["high", "basic", "none", None],
            "native_score": [3.0, 1.0, 0.5, float("nan")],
            "est_tokens": [100, 200, 300, 400],
        }
    )
    selection = DatasetSelection(
        name="toy",
        predicates=[
            Predicate(field="educational_value", op=Op.AT_LEAST, value="basic"),
            Predicate(field="score", op=Op.GTE, value=1.0),
        ],
    )

    mask = document_mask(table, selection, MissingPolicy.DROP)

    assert list(mask) == [True, True, False, False]


def test_document_mask_rejects_a_missing_column():
    table = pa.table({"est_tokens": [1, 2]})

    with pytest.raises(SelectionError, match="no column"):
        document_mask(
            table,
            DatasetSelection(name="toy", predicates=[Predicate(field="nope", op=Op.GTE, value=1)]),
            MissingPolicy.KEEP,
        )


def test_token_retention_can_exceed_row_retention():
    # Quality correlates with length, so a quality filter keeps a larger share of the
    # tokens than of the documents. Scaling a corpus average by row retention would
    # understate the surviving budget, which is why both are reported.
    result = evaluate_on_cube(
        _cube_with_labels(),
        DatasetSelection(name="toy", predicates=[Predicate(field="educational_value", op=Op.AT_LEAST, value="basic")]),
        MissingPolicy.DROP,
    )

    assert result.row_retention == pytest.approx(0.3)
    assert result.token_retention > result.row_retention


def test_duplicate_dataset_in_a_selection_is_rejected():
    with pytest.raises(ValueError, match="appears twice"):
        SelectionConfig(datasets=[DatasetSelection(name="toy"), DatasetSelection(name="toy")])


def test_report_marks_interpolated_rows_and_shows_the_target_gap():
    from modalities.dataloader.preprocessing.quality.selection import BlendResult, DatasetResult

    result = BlendResult(
        datasets=[
            DatasetResult("exact_one", 100, 50, 1000, 600, 2.0),
            DatasetResult("fuzzy_one", 100, 50, 1000, 400, 1.0, exact=False, approximations=["score gte 3.0"]),
        ],
        target_tokens=2000,
    )

    report = format_blend_report(result)

    assert "~" in report
    assert "score gte 3.0" in report
    assert "under" in report


def _cube_without(field: str) -> Cube:
    """A cube grouped on educational_value only, so any other field is unanswerable."""
    table = pa.table({"educational_value": ["high", "basic"], "n_documents": [10, 20], "n_tokens": [100, 200]})
    return Cube(
        dataset="toy",
        label_dimensions=["educational_value"],
        score_binnings={},
        table=table,
        n_documents=30,
        n_tokens=300,
    )


def test_blend_refuses_a_silent_sidecar_scan_and_names_every_offender(tmp_path: Path):
    # A predicate the cube cannot answer used to fall back to reading every document
    # without saying so, which turned a "seconds" preview into a 1.7-billion-document
    # scan. It must be reported instead.
    from modalities.dataloader.preprocessing.quality.selection import evaluate_blend

    config = SelectionConfig(
        datasets=[
            DatasetSelection(
                name="toy",
                predicates=[Predicate(field="commercial_bias", op=Op.AT_LEAST, value="minimal")],
            ),
            DatasetSelection(
                name="other",
                predicates=[Predicate(field="content_quality", op=Op.AT_LEAST, value="good")],
            ),
        ]
    )
    cubes = {"toy": _cube_without("commercial_bias"), "other": _cube_without("content_quality")}
    sidecars = {"toy": tmp_path, "other": tmp_path}

    with pytest.raises(SelectionError) as excinfo:
        evaluate_blend(config, cubes, sidecar_dirs=sidecars)

    message = str(excinfo.value)
    assert "commercial_bias" in message and "content_quality" in message, "both offenders must be reported at once"
    assert "30" in message, "the message should say how many documents a fallback would read"
    assert "--allow-fallback" in message


def test_blend_reports_a_dataset_with_no_cube_at_all(tmp_path: Path):
    from modalities.dataloader.preprocessing.quality.selection import evaluate_blend

    config = SelectionConfig(datasets=[DatasetSelection(name="toy")])

    with pytest.raises(SelectionError, match="no cube was built"):
        evaluate_blend(config, {}, sidecar_dirs={"toy": tmp_path})


# --------------------------------------------------------------------------- curved selections
#
# A curved dataset has `ratio` pinned to 1.0 by the config validator, because its factors
# come from the solved curve rather than from a single number. That makes any code path
# which reports `ratio` instead of solving the curve look successful while costing the
# dataset at a flat 1.0x. The exact sidecar path did exactly this, so these tests check
# the two paths agree rather than only that each returns something.


def _curved(**overrides) -> DatasetSelection:
    """A dataset upsampled along educational_value."""
    from modalities.dataloader.preprocessing.quality.upsampling import UpsamplingSpec

    return DatasetSelection(
        name="toy",
        upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=2.0, **overrides),
    )


_CURVE_ROWS = [
    # (educational_value, n_documents, tokens per document)
    ("minimal", 40, 100),
    ("basic", 30, 100),
    ("moderate", 20, 100),
    ("high", 10, 100),
]


def _curve_cube() -> Cube:
    """A cube over `_CURVE_ROWS`, grouped on educational_value alone."""
    return Cube(
        dataset="toy",
        label_dimensions=["educational_value"],
        score_binnings={},
        table=pa.table(
            {
                "educational_value": [level for level, _, _ in _CURVE_ROWS],
                "n_documents": [n for _, n, _ in _CURVE_ROWS],
                "n_tokens": [n * t for _, n, t in _CURVE_ROWS],
            }
        ),
        n_documents=sum(n for _, n, _ in _CURVE_ROWS),
        n_tokens=sum(n * t for _, n, t in _CURVE_ROWS),
    )


def _curve_sidecar(tmp_path: Path, extra_column: dict | None = None) -> Path:
    """The same documents as `_curve_cube`, written out one row per document."""
    import pyarrow.parquet as pq

    levels: list[str] = []
    tokens: list[int] = []
    for level, n_documents, per_document in _CURVE_ROWS:
        levels.extend([level] * n_documents)
        tokens.extend([per_document] * n_documents)

    columns = {
        "file_id": [0] * len(levels),
        "line_no": list(range(len(levels))),
        "byte_offset": [0] * len(levels),
        "byte_len": [per for per in tokens],
        "text_bytes": [per for per in tokens],
        "est_tokens": tokens,
        "educational_value": levels,
    }
    if extra_column:
        columns.update({name: values * len(levels) for name, values in extra_column.items()})

    sidecar_dir = tmp_path / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    # Several parts, because the exact path accumulates buckets across parts and row
    # groups; a single-part fixture would not exercise that.
    total = len(levels)
    for part, start in enumerate(range(0, total, 37)):
        chunk = {name: values[start : start + 37] for name, values in columns.items()}
        pq.write_table(pa.table(chunk), sidecar_dir / f"part-{part:05d}.parquet", row_group_size=11)
    return sidecar_dir


def test_the_exact_path_solves_the_curve_instead_of_reporting_a_flat_ratio(tmp_path: Path):
    from modalities.dataloader.preprocessing.quality.selection import evaluate_on_sidecar

    dataset = _curved()
    assert dataset.ratio == 1.0, "the validator pins a curved dataset's flat ratio to 1.0"

    result = evaluate_on_sidecar(_curve_sidecar(tmp_path), dataset, MissingPolicy.KEEP)

    assert result.plan is not None, "an exact evaluation of a curved dataset must carry its solved curve"
    assert result.effective_tokens > result.tokens_kept, "a 2.0x target must draw more than it holds"
    assert result.ratio_label != "1.00", "reporting the pinned flat ratio hides the curve entirely"


def test_a_curve_costs_the_same_whether_it_is_evaluated_exactly_or_from_the_cube(tmp_path: Path):
    # The regression that motivates this: the cube path solved the curve and the exact
    # path did not, so `--exact` quietly reported a different, much smaller blend.
    from modalities.dataloader.preprocessing.quality.selection import evaluate_on_sidecar

    dataset = _curved()

    from_cube = evaluate_on_cube(_curve_cube(), dataset, MissingPolicy.KEEP)
    from_sidecar = evaluate_on_sidecar(_curve_sidecar(tmp_path), dataset, MissingPolicy.KEEP)

    assert from_sidecar.tokens_total == from_cube.tokens_total
    assert from_sidecar.n_documents_kept == from_cube.n_documents_kept
    assert from_sidecar.effective_tokens == pytest.approx(from_cube.effective_tokens, rel=1e-9)
    assert [b.factor for b in from_sidecar.plan.buckets] == pytest.approx(
        [b.factor for b in from_cube.plan.buckets], rel=1e-9
    )


def test_a_curve_that_discards_its_worst_bucket_drops_those_documents_exactly(tmp_path: Path):
    from modalities.dataloader.preprocessing.quality.selection import evaluate_on_sidecar

    # The 40 `minimal` documents hold 4,000 of the 10,000 tokens, so they are exactly the
    # bottom 40% of the quality axis and a 40% discard should remove that bucket whole.
    dataset = _curved(discard_below_percentile=40.0)
    result = evaluate_on_sidecar(_curve_sidecar(tmp_path), dataset, MissingPolicy.KEEP)

    assert result.plan.discard_fraction > 0.0
    assert result.n_documents_kept == 60, "documents kept must reflect the curve's discard, not the predicates alone"
    assert result.n_documents_kept < result.n_documents_total


def test_the_exact_path_names_the_missing_quality_column_rather_than_failing_obscurely(tmp_path: Path):
    import pyarrow.parquet as pq

    from modalities.dataloader.preprocessing.quality.selection import evaluate_on_sidecar

    sidecar_dir = tmp_path / "unjoined"
    sidecar_dir.mkdir()
    pq.write_table(pa.table({"est_tokens": [10, 20]}), sidecar_dir / "part-00000.parquet")

    with pytest.raises(SelectionError, match="educational_value"):
        evaluate_on_sidecar(sidecar_dir, _curved(), MissingPolicy.KEEP)


def test_the_fallback_path_keeps_the_curve_when_a_predicate_forces_a_sidecar_scan(tmp_path: Path):
    # Deliberately unmocked. The previous version of this test stubbed out
    # `evaluate_on_sidecar` and only covered a flat selection, which is precisely why a
    # curve being flattened on that path went unnoticed.
    from modalities.dataloader.preprocessing.quality.selection import evaluate_blend
    from modalities.dataloader.preprocessing.quality.upsampling import UpsamplingSpec

    dataset = DatasetSelection(
        name="toy",
        # The cube below is not grouped on commercial_bias, so this predicate is
        # unanswerable from it and the blend must fall back to the sidecar.
        predicates=[Predicate(field="commercial_bias", op=Op.AT_LEAST, value="minimal")],
        upsampling=UpsamplingSpec(quality_field="educational_value", target_ratio=2.0),
    )
    sidecar_dir = _curve_sidecar(tmp_path, extra_column={"commercial_bias": ["minimal"]})

    report = evaluate_blend(
        SelectionConfig(datasets=[dataset]),
        {"toy": _curve_cube()},
        sidecar_dirs={"toy": sidecar_dir},
        allow_sidecar_fallback=True,
    )

    result = report.datasets[0]
    assert result.exact, "a sidecar scan is exact by construction"
    assert result.plan is not None, "falling back to the sidecar must not discard the curve"
    assert result.effective_tokens == pytest.approx(2.0 * result.tokens_kept, rel=0.05)
