"""Translate demand uncertainty into auditable replenishment recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InventoryPolicy:
    service_quantile: str = "p90"
    review_period_days: int = 7
    minimum_order_quantity: float = 0.0

    def __post_init__(self) -> None:
        if self.service_quantile not in {"p50", "p90"}:
            raise ValueError("service_quantile must be p50 or p90.")
        if self.review_period_days <= 0:
            raise ValueError("review_period_days must be positive.")
        if self.minimum_order_quantity < 0:
            raise ValueError("minimum_order_quantity must be non-negative.")


def inventory_recommendations(
    forecasts: pd.DataFrame,
    *,
    on_hand: pd.Series | np.ndarray | None = None,
    policy: InventoryPolicy | None = None,
) -> pd.DataFrame:
    """Create inventory recommendations from P10/P50/P90 lead-time demand.

    If on-hand inventory is unavailable, recommendation quantities are scenario
    requirements rather than executable purchase orders. This distinction
    prevents fabricated stock levels in the portfolio project.
    """
    policy = policy or InventoryPolicy()
    required = {"p10", "p50", "p90"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")

    result = forecasts.copy()
    q = result[["p10", "p50", "p90"]].to_numpy(dtype=float)
    if not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Forecast quantiles must be finite and non-negative.")
    if not ((result["p10"] <= result["p50"]) & (result["p50"] <= result["p90"])).all():
        raise ValueError("Forecast quantiles must satisfy p10 <= p50 <= p90.")

    result["uncertainty_width"] = result["p90"] - result["p10"]
    result["safety_stock_proxy"] = result["p90"] - result["p50"]
    result["target_stock"] = result[policy.service_quantile]

    if on_hand is None:
        result["on_hand"] = np.nan
        result["stockout_risk"] = np.nan
        result["reorder_quantity"] = np.nan
        result["decision_status"] = "inventory_required"
        return result

    stock = np.asarray(on_hand, dtype=float)
    if stock.shape != (len(result),) or not np.isfinite(stock).all() or (stock < 0).any():
        raise ValueError("on_hand must be finite, non-negative, and row-aligned.")
    result["on_hand"] = stock
    result["stockout_risk"] = np.where(
        stock < result["p50"],
        "high",
        np.where(stock < result["p90"], "medium", "low"),
    )
    reorder = np.maximum(result["target_stock"].to_numpy() - stock, 0.0)
    if policy.minimum_order_quantity > 0:
        reorder = np.where(
            (reorder > 0) & (reorder < policy.minimum_order_quantity),
            policy.minimum_order_quantity,
            reorder,
        )
    result["reorder_quantity"] = reorder
    result["decision_status"] = "actionable"
    return result
