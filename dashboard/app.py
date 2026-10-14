"""DemandPilot forecasting and monitoring dashboard."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demandpilot.monitoring.report import summarize_drift  # noqa: E402

st.set_page_config(page_title="DemandPilot", layout="wide")
st.title("DemandPilot — Demand & Inventory Intelligence")
st.caption("Probabilistic 7-day SKU demand forecasts and monitoring")


def load_export(file):
    return pd.read_parquet(file) if file.name.endswith(".parquet") else pd.read_csv(file)


forecast_tab, drift_tab = st.tabs(["Forecast explorer", "Drift monitoring"])
with forecast_tab:
    uploaded = st.file_uploader("Load forecast export", type=["csv", "parquet"])
    if uploaded is None:
        st.info("Upload a forecast export with sku, p10, p50 and p90 columns.")
    else:
        frame = load_export(uploaded)
        missing = {"sku", "p10", "p50", "p90"} - set(frame)
        if missing:
            st.error(f"Missing columns: {sorted(missing)}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("SKUs", f"{frame['sku'].nunique():,}")
            c2.metric("Median P50", f"{frame['p50'].median():,.1f}")
            c3.metric("Median uncertainty", f"{(frame['p90'] - frame['p10']).median():,.1f}")
            sku = st.selectbox("SKU", sorted(frame["sku"].astype(str).unique()))
            st.dataframe(frame[frame["sku"].astype(str) == sku], use_container_width=True)

with drift_tab:
    st.subheader("Numeric feature drift")
    st.caption("PSI: <0.10 stable, 0.10–<0.25 watch, >=0.25 drift.")
    reference_file = st.file_uploader("Reference snapshot", type=["csv", "parquet"], key="ref")
    current_file = st.file_uploader("Current snapshot", type=["csv", "parquet"], key="cur")
    if reference_file is not None and current_file is not None:
        try:
            reference = load_export(reference_file)
            current = load_export(current_file)
            common = sorted(set(reference.select_dtypes("number")) & set(current.select_dtypes("number")))
            selected = st.multiselect("Compare features", common, default=common)
            if selected:
                report = summarize_drift(reference, current, columns=selected)
                st.metric("Drift status", report["overall_status"].upper())
                st.dataframe(pd.DataFrame(report["features"]), use_container_width=True)
                st.download_button("Download JSON report", json.dumps(report, indent=2),
                                   file_name="demandpilot_drift.json", mime="application/json")
            else:
                st.warning("Select at least one numeric feature.")
        except (ValueError, TypeError, OSError) as exc:
            st.error(f"Invalid monitoring inputs: {exc}")
