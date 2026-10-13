import pandas as pd
from demandpilot.monitoring.drift import drift_status, population_stability_index

def test_identical_distribution_is_stable():
    series=pd.Series(range(100))
    assert population_stability_index(series,series) < 0.1
    assert drift_status(0.05)=="stable"

def test_drift_status_thresholds():
    assert drift_status(0.15)=="watch"
    assert drift_status(0.30)=="drift"
