"""Exit-code based drift gate for scheduled monitoring jobs."""
import argparse
import json
from pathlib import Path

from demandpilot.monitoring.alerts import evaluate_drift_alert


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON report produced by generate_drift_report.py")
    parser.add_argument(
        "--fail-on",
        choices=("never", "drift", "watch"),
        default="drift",
        help="Exit non-zero at or above this monitoring severity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = json.loads(args.report.read_text(encoding="utf-8"))
    alert = evaluate_drift_alert(summary)
    print(json.dumps(alert.as_dict(), indent=2))

    should_fail = args.fail_on == "watch" and alert.severity in {"warning", "critical"}
    should_fail = should_fail or (args.fail_on == "drift" and alert.severity == "critical")
    raise SystemExit(2 if should_fail else 0)


if __name__ == "__main__":
    main()
