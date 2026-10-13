import pandas as pd
import pytest

from demandpilot.evaluation.split import chronological_split


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2011-09-01", "2011-12-09", freq="D")

    return pd.DataFrame(
        {
            "sku": ["A"] * len(dates),
            "date": dates,
            "demand": 1.0,
            "lag_28": 1.0,
            "rolling_mean_28": 1.0,
            "rolling_std_28": 0.0,
            "rolling_min_28": 1.0,
            "rolling_max_28": 1.0,
        }
    )


def test_split_is_strictly_chronological() -> None:
    split, _ = chronological_split(_frame())

    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()


def test_validation_contains_28_calendar_days() -> None:
    split, _ = chronological_split(_frame())

    assert split.validation["date"].nunique() == 28


def test_test_contains_28_calendar_days() -> None:
    split, _ = chronological_split(_frame())

    assert split.test["date"].nunique() == 28


def test_incomplete_history_is_excluded() -> None:
    frame = _frame()
    frame.loc[0, "lag_28"] = float("nan")

    _, report = chronological_split(frame)

    assert report.excluded_incomplete_history_rows == 1


def test_input_is_not_mutated() -> None:
    frame = _frame()
    original = frame.copy(deep=True)

    chronological_split(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_split_is_input_order_independent() -> None:
    frame = _frame()

    split_a, _ = chronological_split(frame)
    split_b, _ = chronological_split(
        frame.sample(frac=1.0, random_state=42)
    )

    pd.testing.assert_frame_equal(split_a.train, split_b.train)
    pd.testing.assert_frame_equal(split_a.validation, split_b.validation)
    pd.testing.assert_frame_equal(split_a.test, split_b.test)


def test_overlapping_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError):
        chronological_split(
            _frame(),
            train_end="2011-10-20",
            validation_start="2011-10-15",
        )


def test_missing_required_feature_is_rejected() -> None:
    frame = _frame().drop(columns=["lag_28"])

    with pytest.raises(ValueError):
        chronological_split(frame)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame())