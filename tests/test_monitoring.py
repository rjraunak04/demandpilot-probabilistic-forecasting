import pandas as pd

from demandpilot.monitoring.drift import drift_status, population_stability_index


def test_identical_distribution_is_stable():
    series = pd.Series(range(100))
    assert population_stability_index(series, series) < 0.1
    assert drift_status(0.05) == "stable"


def test_drift_status_thresholds():
    assert drift_status(0.15) == "watch"
    assert drift_status(0.30) == "drift"


def test_population_stability_index_rejects_invalid_bins():
    series = pd.Series(range(10))
    try:
        population_stability_index(series, series, bins=1)
    except ValueError as exc:
        assert str(exc) == "bins must be at least 2"
    else:
        raise AssertionError("Expected ValueError")
