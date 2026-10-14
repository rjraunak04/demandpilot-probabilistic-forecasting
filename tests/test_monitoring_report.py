import json

import pandas as pd
import pytest

from demandpilot.monitoring.report import save_summary, summarize_drift


def test_identical_is_stable():
    frame = pd.DataFrame({"units": range(100), "sku": ["A"] * 100})
    report = summarize_drift(frame, frame)
    assert report["overall_status"] == "stable"
    assert report["counts"]["stable"] == 1


def test_shifted_distribution_is_drift():
    ref = pd.DataFrame({"units": range(100)})
    cur = pd.DataFrame({"units": range(1000, 1100)})
    assert summarize_drift(ref, cur)["overall_status"] == "drift"


@pytest.mark.parametrize("ref,cur,columns", [
    (pd.DataFrame(), pd.DataFrame({"a": [1]}), None),
    (pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}), None),
    (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}), ["missing"]),
    (pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["y"]}), ["a"]),
    (pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}), ["a", "a"]),
])
def test_invalid_inputs(ref, cur, columns):
    with pytest.raises(ValueError):
        summarize_drift(ref, cur, columns=columns)


def test_json_roundtrip(tmp_path):
    frame = pd.DataFrame({"units": range(100)})
    output = save_summary(summarize_drift(frame, frame), tmp_path / "monitor" / "drift.json")
    assert json.loads(output.read_text())["overall_status"] == "stable"
