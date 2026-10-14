"""Operational alert policy for drift monitoring."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DriftAlert:
    """Machine-readable operational decision derived from a drift summary."""

    severity: str
    action: str
    message: str
    drift_features: tuple[str, ...]
    watch_features: tuple[str, ...]

    def as_dict(self):
        """Return a JSON-serializable representation."""
        payload = asdict(self)
        payload["drift_features"] = list(self.drift_features)
        payload["watch_features"] = list(self.watch_features)
        return payload


def evaluate_drift_alert(summary: dict) -> DriftAlert:
    """Map PSI report statuses to an explicit operator action."""
    features = summary.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("drift summary must contain at least one feature")

    drift = tuple(
        str(item["feature"]) for item in features if item.get("status") == "drift"
    )
    watch = tuple(
        str(item["feature"]) for item in features if item.get("status") == "watch"
    )
    unknown = [
        item.get("status")
        for item in features
        if item.get("status") not in {"stable", "watch", "drift"}
    ]
    if unknown:
        raise ValueError(f"unsupported drift status: {unknown[0]}")

    if drift:
        return DriftAlert(
            severity="critical",
            action="investigate_before_retraining",
            message=f"Material drift detected in {len(drift)} feature(s).",
            drift_features=drift,
            watch_features=watch,
        )
    if watch:
        return DriftAlert(
            severity="warning",
            action="monitor_next_window",
            message=f"Drift watch threshold reached in {len(watch)} feature(s).",
            drift_features=(),
            watch_features=watch,
        )
    return DriftAlert(
        severity="ok",
        action="continue_monitoring",
        message="No material numeric feature drift detected.",
        drift_features=(),
        watch_features=(),
    )
