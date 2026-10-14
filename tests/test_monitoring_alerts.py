import pytest

from demandpilot.monitoring.alerts import evaluate_drift_alert


def summary(*statuses):
    return {
        "features": [
            {"feature": f"feature_{index}", "status": status}
            for index, status in enumerate(statuses)
        ]
    }


@pytest.mark.parametrize(
    ("statuses", "severity", "action"),
    [
        (("stable", "stable"), "ok", "continue_monitoring"),
        (("stable", "watch"), "warning", "monitor_next_window"),
        (("watch", "drift"), "critical", "investigate_before_retraining"),
    ],
)
def test_alert_policy(statuses, severity, action):
    alert = evaluate_drift_alert(summary(*statuses))
    assert alert.severity == severity
    assert alert.action == action


def test_alert_lists_affected_features():
    alert = evaluate_drift_alert(summary("drift", "watch", "stable"))
    assert alert.as_dict()["drift_features"] == ["feature_0"]
    assert alert.as_dict()["watch_features"] == ["feature_1"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"features": []}, {"features": [{"feature": "x", "status": "unknown"}]}],
)
def test_alert_rejects_invalid_summary(payload):
    with pytest.raises(ValueError):
        evaluate_drift_alert(payload)
