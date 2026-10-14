"""Lightweight feature drift checks for production monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate PSI using reference-distribution quantile bins."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        raise ValueError("reference and current must contain numeric values")

    edges = ref.quantile(np.linspace(0, 1, bins + 1)).drop_duplicates().to_numpy()
    if len(edges) < 2:
        return 0.0

    edges[0] = float("-inf")
    edges[-1] = float("inf")
    ref_p = (
        pd.cut(ref, edges, include_lowest=True)
        .value_counts(normalize=True, sort=False)
        .clip(lower=1e-6)
    )
    cur_p = (
        pd.cut(cur, edges, include_lowest=True)
        .value_counts(normalize=True, sort=False)
        .clip(lower=1e-6)
    )
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


def drift_status(psi: float) -> str:
    """Map PSI to an operational monitoring status."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drift"
