import pandas as pd
import pytest

from demandpilot.models.selection import TUNED_PARAMS, select_champion, tuned_estimator


def test_tuned_parameter_contract_contains_both_models() -> None:
    assert set(TUNED_PARAMS) == {"lightgbm", "xgboost"}


def test_tuned_estimator_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        tuned_estimator("random_forest")


def test_select_champion_uses_wape() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["lightgbm", "xgboost"],
            "wape": [0.80, 0.85],
            "mae": [5.0, 4.0],
        }
    )
    selection = select_champion(metrics, baseline_wape=1.20)
    assert selection.champion == "lightgbm"
    assert selection.primary_metric == "wape"
    assert selection.validation_wape == 0.80


def test_select_champion_reports_baseline_improvement() -> None:
    metrics = pd.DataFrame({"model": ["lightgbm"], "wape": [0.9]})
    selection = select_champion(metrics, baseline_wape=1.2)
    assert selection.improvement_over_baseline_pct == pytest.approx(25.0)


def test_select_champion_requires_metrics() -> None:
    with pytest.raises(ValueError, match="model and wape"):
        select_champion(pd.DataFrame(), baseline_wape=1.2)
