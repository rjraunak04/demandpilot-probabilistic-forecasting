"""Numeric PSI monitoring reports and optional Evidently HTML export."""
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from demandpilot.monitoring.drift import drift_status, population_stability_index


def summarize_drift(reference: pd.DataFrame, current: pd.DataFrame, columns=None, bins=10):
    """Return a validated, JSON-serializable feature drift report."""
    if reference.empty or current.empty:
        raise ValueError("reference and current must be non-empty")
    if columns is None:
        columns = sorted(
            set(reference.select_dtypes("number"))
            & set(current.select_dtypes("number"))
        )
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("select at least one unique numeric column")
    missing = set(columns) - (set(reference) & set(current))
    if missing:
        raise ValueError(f"missing comparison columns: {sorted(missing)}")
    features = []
    for name in columns:
        ref = pd.to_numeric(reference[name], errors="coerce").dropna()
        cur = pd.to_numeric(current[name], errors="coerce").dropna()
        if ref.empty or cur.empty:
            raise ValueError(f"no valid numeric observations for {name}")
        psi = population_stability_index(ref, cur, bins=bins)
        features.append({"feature": name, "psi": round(psi, 6), "status": drift_status(psi),
                         "reference_count": len(ref), "current_count": len(cur)})
    counts = {s: sum(f["status"] == s for f in features) for s in ("stable", "watch", "drift")}
    overall = "drift" if counts["drift"] else "watch" if counts["watch"] else "stable"
    return {"schema_version": "1.0", "created_at_utc": datetime.now(UTC).isoformat(),
            "reference_rows": len(reference), "current_rows": len(current),
            "overall_status": overall, "counts": counts, "features": features,
            "method": "reference-quantile PSI", "thresholds": {"watch": 0.1, "drift": 0.25}}


def save_summary(summary, path):
    """Persist JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + chr(10), encoding="utf-8"
    )
    return path


def save_evidently_html(reference, current, path):
    """Render optional Evidently 0.7+ HTML report."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: pip install -e '.[monitoring]'") from exc
    if reference.empty or current.empty:
        raise ValueError("reference and current must be non-empty")
    shared = sorted(set(reference) & set(current))
    if not shared:
        raise ValueError("no shared columns")
    snapshot = Report([DataDriftPreset()]).run(
        reference_data=reference[shared], current_data=current[shared]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(path))
    return path
