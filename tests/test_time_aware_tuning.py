import pandas as pd
import pytest

from demandpilot.models.tuning import _tuning_window


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2011-01-01", "2011-10-14", freq="D")
    return pd.DataFrame({"date": dates, "value": range(len(dates))})


def test_tuning_window_is_strictly_chronological() -> None:
    fit, validation = _tuning_window(_frame(), validation_days=7, training_days=180)
    assert fit["date"].max() < validation["date"].min()
    assert len(validation["date"].unique()) == 7
    assert len(fit["date"].unique()) == 180


def test_tuning_window_ends_at_training_partition_boundary() -> None:
    _, validation = _tuning_window(_frame())
    assert validation["date"].max() == pd.Timestamp("2011-10-14")


def test_tuning_window_does_not_reach_locked_validation() -> None:
    _, validation = _tuning_window(_frame())
    assert validation["date"].max() < pd.Timestamp("2011-10-15")


def test_tuning_window_rejects_missing_date() -> None:
    with pytest.raises(ValueError, match="contain date"):
        _tuning_window(pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize(("validation_days", "training_days"), [(0, 180), (7, 0)])
def test_tuning_window_requires_positive_lengths(validation_days, training_days) -> None:
    with pytest.raises(ValueError, match="positive"):
        _tuning_window(
            _frame(),
            validation_days=validation_days,
            training_days=training_days,
        )
