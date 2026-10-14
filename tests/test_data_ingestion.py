"""Tests for DemandPilot retail ingestion utilities."""

from pathlib import Path

import pandas as pd
import pytest

from demandpilot.data.ingest import (
    summarize_dataset,
    validate_raw_file,
)


def test_validate_raw_file_rejects_missing_file(tmp_path: Path) -> None:
    """Missing raw workbooks should fail loudly."""
    missing_file = tmp_path / "missing.xlsx"

    with pytest.raises(
        FileNotFoundError,
        match="Raw retail dataset not found",
    ):
        validate_raw_file(missing_file)


def test_validate_raw_file_rejects_wrong_extension(
    tmp_path: Path,
) -> None:
    """Raw ingestion currently requires an Excel workbook."""
    csv_file = tmp_path / "retail.csv"
    csv_file.write_text("example", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Expected an \.xlsx workbook",
    ):
        validate_raw_file(csv_file)


def test_validate_raw_file_accepts_xlsx(tmp_path: Path) -> None:
    """Existing Excel workbooks should pass file-level validation."""
    workbook = tmp_path / "retail.xlsx"
    workbook.touch()

    validate_raw_file(workbook)


def test_summarize_dataset_returns_expected_metadata() -> None:
    """Ingestion summary should preserve basic dataset metadata."""
    data = pd.DataFrame(
        {
            "InvoiceDate": [
                "2009-12-01 08:00:00",
                "2009-12-02 09:30:00",
            ],
            "SourceSheet": [
                "Year 2009-2010",
                "Year 2010-2011",
            ],
            "Quantity": [1, 2],
        }
    )

    summary = summarize_dataset(data)

    assert summary["rows"] == 2
    assert summary["columns"] == 3
    assert summary["source_sheets"] == [
        "Year 2009-2010",
        "Year 2010-2011",
    ]
    assert summary["date_min"] == "2009-12-01T08:00:00"
    assert summary["date_max"] == "2009-12-02T09:30:00"