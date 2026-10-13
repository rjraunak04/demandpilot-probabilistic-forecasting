import numpy as np
import pandas as pd
import pytest

from demandpilot.inventory.policy import InventoryPolicy, inventory_recommendations


def _forecast() -> pd.DataFrame:
    return pd.DataFrame({"p10": [5.0, 10.0], "p50": [10.0, 20.0], "p90": [15.0, 30.0]})


def test_without_inventory_does_not_invent_reorder_quantity() -> None:
    result = inventory_recommendations(_forecast())
    assert result["reorder_quantity"].isna().all()
    assert result["stockout_risk"].isna().all()
    assert result["decision_status"].eq("inventory_required").all()
    assert np.allclose(result["safety_stock_proxy"], [5.0, 10.0])


def test_with_inventory_creates_actionable_decisions() -> None:
    result = inventory_recommendations(_forecast(), on_hand=[8.0, 35.0])
    assert result["stockout_risk"].tolist() == ["high", "low"]
    assert np.allclose(result["reorder_quantity"], [7.0, 0.0])
    assert result["decision_status"].eq("actionable").all()


def test_medium_risk_is_between_p50_and_p90() -> None:
    result = inventory_recommendations(_forecast().iloc[[0]], on_hand=[12.0])
    assert result.iloc[0]["stockout_risk"] == "medium"


def test_minimum_order_quantity_is_respected() -> None:
    policy = InventoryPolicy(minimum_order_quantity=10.0)
    result = inventory_recommendations(_forecast().iloc[[0]], on_hand=[14.0], policy=policy)
    assert result.iloc[0]["reorder_quantity"] == 10.0


def test_crossed_quantiles_are_rejected() -> None:
    invalid = pd.DataFrame({"p10": [10.0], "p50": [5.0], "p90": [12.0]})
    with pytest.raises(ValueError, match="p10 <= p50 <= p90"):
        inventory_recommendations(invalid)
