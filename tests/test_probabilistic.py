import numpy as np
import pytest

from demandpilot.models.probabilistic import QuantileForecast, pinball_loss, quantile_metrics


def test_pinball_loss_is_zero_for_perfect_forecast() -> None:
    assert pinball_loss([1, 2, 3], [1, 2, 3], 0.5) == 0.0


def test_pinball_rejects_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="strictly"):
        pinball_loss([1], [1], 1.0)


def test_quantile_metrics_report_coverage_and_width() -> None:
    forecast = QuantileForecast(
        p10=np.array([1.0, 1.0]),
        p50=np.array([2.0, 2.0]),
        p90=np.array([3.0, 3.0]),
    )
    result = quantile_metrics(np.array([2.0, 4.0]), forecast)
    assert result["coverage_80"] == 0.5
    assert result["mean_interval_width"] == 2.0
