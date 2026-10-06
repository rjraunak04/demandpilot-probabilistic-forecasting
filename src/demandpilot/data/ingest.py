"""Data ingestion utilities for the Online Retail II dataset."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the DemandPilot data configuration."""
    config_path = Path(config_path)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Data configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Data configuration must contain a YAML mapping.")

    return config


def resolve_raw_file(config: dict[str, Any]) -> Path:
    """Resolve the configured raw retail workbook path."""
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    raw_file = config["dataset"]["raw_file"]
    return raw_dir / raw_file


def validate_raw_file(raw_file: Path) -> None:
    """Fail loudly when the configured raw workbook is unavailable."""
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw retail dataset not found: {raw_file}")

    if raw_file.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx workbook, received: {raw_file.name}")


def load_sheet(raw_file: Path, sheet_name: str) -> pd.DataFrame:
    """Load one worksheet from the Online Retail II workbook."""
    return pd.read_excel(raw_file, sheet_name=sheet_name, engine="openpyxl")


def load_retail_data(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Load and combine all configured Online Retail II worksheets."""
    config = load_config(config_path)
    raw_file = resolve_raw_file(config)
    validate_raw_file(raw_file)

    sheet_names = config["ingestion"]["sheet_names"]

    if not sheet_names:
        raise ValueError("At least one worksheet must be configured.")

    frames: list[pd.DataFrame] = []

    for sheet_name in sheet_names:
        frame = load_sheet(raw_file, sheet_name)
        frame["SourceSheet"] = sheet_name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def summarize_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight ingestion metadata without transforming the data."""
    invoice_date = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "column_names": data.columns.tolist(),
        "date_min": (
            invoice_date.min().isoformat()
            if invoice_date.notna().any()
            else None
        ),
        "date_max": (
            invoice_date.max().isoformat()
            if invoice_date.notna().any()
            else None
        ),
        "source_sheets": sorted(data["SourceSheet"].dropna().unique().tolist()),
    }


def main() -> None:
    """Run a command-line ingestion smoke check."""
    data = load_retail_data()
    summary = summarize_dataset(data)

    print("DemandPilot retail ingestion successful.")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Date range: {summary['date_min']} -> {summary['date_max']}")
    print(f"Source sheets: {summary['source_sheets']}")


if __name__ == "__main__":
    main()