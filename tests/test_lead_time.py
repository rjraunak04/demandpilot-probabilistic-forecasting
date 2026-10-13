import pandas as pd
import pytest

from demandpilot.evaluation.lead_time import (
    build_lead_time_examples,
    snapshot_examples,
    training_examples_at_origin,
)
from demandpilot.models.boosting import MODEL_FEATURES


def _features() -> pd.DataFrame:
    rows = []
    for sku in ("A", "B"):
        for i, date in enumerate(pd.date_range("2011-01-01", periods=40, freq="D")):
            row = {"sku": sku, "date": date, "demand": float(i + (0 if sku == "A" else 100))}
            row.update({feature: float(i + 1) for feature in MODEL_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_lead_time_target_is_current_plus_next_six_days() -> None:
    examples = build_lead_time_examples(_features())
    first = examples.loc[(examples["sku"] == "A") & (examples["date"] == "2011-01-01")].iloc[0]
    assert first["lead_time_demand"] == sum(range(7))
    assert first["target_end"] == pd.Timestamp("2011-01-07")


def test_incomplete_future_window_is_excluded() -> None:
    examples = build_lead_time_examples(_features())
    assert examples.groupby("sku").size().eq(34).all()


def test_gap_invalidates_window_crossing_gap() -> None:
    data = _features()
    data = data.loc[~((data["sku"] == "A") & (data["date"] == "2011-01-05"))]
    examples = build_lead_time_examples(data)
    invalid_starts = pd.date_range("2011-01-01", "2011-01-04", freq="D")
    invalid = examples.loc[
        (examples["sku"] == "A") & examples["date"].isin(invalid_starts)
    ]
    assert invalid.empty


def test_skus_do_not_bleed_into_each_other() -> None:
    examples = build_lead_time_examples(_features())
    a = examples.loc[(examples["sku"] == "A") & (examples["date"] == "2011-01-01")].iloc[0]
    b = examples.loc[(examples["sku"] == "B") & (examples["date"] == "2011-01-01")].iloc[0]
    assert b["lead_time_demand"] - a["lead_time_demand"] == 700.0


def test_training_labels_finish_before_origin() -> None:
    examples = build_lead_time_examples(_features())
    train = training_examples_at_origin(examples, "2011-01-20")
    assert (train["target_end"] < pd.Timestamp("2011-01-20")).all()


def test_snapshot_is_exact_origin() -> None:
    examples = build_lead_time_examples(_features())
    snap = snapshot_examples(examples, "2011-01-20")
    assert snap["date"].eq(pd.Timestamp("2011-01-20")).all()
    assert set(snap["sku"]) == {"A", "B"}


def test_invalid_horizon_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_lead_time_examples(_features(), horizon_days=0)


def test_training_excludes_incomplete_model_history() -> None:
    data = _features()
    data.loc[data.index[0], "lag_28"] = float("nan")
    examples = build_lead_time_examples(data)
    train = training_examples_at_origin(examples, "2011-01-20")
    assert train[list(MODEL_FEATURES)].notna().all().all()


def test_snapshot_excludes_incomplete_model_history() -> None:
    data = _features()
    mask = (data["sku"] == "A") & (data["date"] == pd.Timestamp("2011-01-20"))
    data.loc[mask, "rolling_mean_28"] = float("nan")
    examples = build_lead_time_examples(data)
    snapshot = snapshot_examples(examples, "2011-01-20")
    assert set(snapshot["sku"]) == {"B"}
    assert snapshot[list(MODEL_FEATURES)].notna().all().all()
