from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SplitReport:
    input_rows: int
    eligible_rows: int
    excluded_incomplete_history_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "input_rows": self.input_rows,
            "eligible_rows": self.eligible_rows,
            "excluded_incomplete_history_rows": self.excluded_incomplete_history_rows,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def chronological_split(
    features: pd.DataFrame,
    *,
    train_end: str = "2011-10-14",
    validation_start: str = "2011-10-15",
    validation_end: str = "2011-11-11",
    test_start: str = "2011-11-12",
    test_end: str = "2011-12-09",
    required_history_columns: tuple[str, ...] = (
        "lag_28",
        "rolling_mean_28",
        "rolling_std_28",
        "rolling_min_28",
        "rolling_max_28",
    ),
) -> tuple[ChronologicalSplit, SplitReport]:
    if features.empty:
        raise ValueError("Feature dataset must not be empty.")

    required = {"sku", "date", "demand", *required_history_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values(["date", "sku"]).reset_index(drop=True)

    boundaries = {
        "train_end": pd.Timestamp(train_end),
        "validation_start": pd.Timestamp(validation_start),
        "validation_end": pd.Timestamp(validation_end),
        "test_start": pd.Timestamp(test_start),
        "test_end": pd.Timestamp(test_end),
    }

    if not (
        boundaries["train_end"] < boundaries["validation_start"]
        <= boundaries["validation_end"] < boundaries["test_start"]
        <= boundaries["test_end"]
    ):
        raise ValueError("Chronological split boundaries overlap or are out of order.")

    if boundaries["validation_start"] != boundaries["train_end"] + timedelta(days=1):
        raise ValueError("Train and validation periods must be contiguous.")

    if boundaries["test_start"] != boundaries["validation_end"] + timedelta(days=1):
        raise ValueError("Validation and test periods must be contiguous.")

    history_complete = data[list(required_history_columns)].notna().all(axis=1)
    eligible = data.loc[history_complete].copy()

    train = eligible.loc[eligible["date"] <= boundaries["train_end"]].copy()
    validation = eligible.loc[
        eligible["date"].between(
            boundaries["validation_start"],
            boundaries["validation_end"],
        )
    ].copy()
    test = eligible.loc[
        eligible["date"].between(
            boundaries["test_start"],
            boundaries["test_end"],
        )
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    if not (
        train["date"].max()
        < validation["date"].min()
        <= validation["date"].max()
        < test["date"].min()
    ):
        raise ValueError("Temporal leakage detected between partitions.")

    split = ChronologicalSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )

    report = SplitReport(
        input_rows=len(data),
        eligible_rows=len(eligible),
        excluded_incomplete_history_rows=len(data) - len(eligible),
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
        train_start=str(train["date"].min().date()),
        train_end=str(train["date"].max().date()),
        validation_start=str(validation["date"].min().date()),
        validation_end=str(validation["date"].max().date()),
        test_start=str(test["date"].min().date()),
        test_end=str(test["date"].max().date()),
    )

    return split, report