"""DemandPilot Streamlit operations dashboard."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DemandPilot",layout="wide")
st.title("DemandPilot — Demand & Inventory Intelligence")
st.caption("Probabilistic 7-day SKU demand forecasts and inventory decision support")

uploaded=st.file_uploader("Load forecast export",type=["csv","parquet"])
if uploaded is None:
    st.info("Upload a forecast export with sku, p10, p50 and p90 columns.")
else:
    frame=pd.read_parquet(uploaded) if uploaded.name.endswith(".parquet") else pd.read_csv(uploaded)
    required={"sku","p10","p50","p90"}
    missing=required-set(frame.columns)
    if missing: st.error(f"Missing columns: {sorted(missing)}")
    else:
        c1,c2,c3=st.columns(3)
        c1.metric("SKUs",f"{frame['sku'].nunique():,}")
        c2.metric("Median P50",f"{frame['p50'].median():,.1f}")
        c3.metric("Median uncertainty",f"{(frame['p90']-frame['p10']).median():,.1f}")
        st.subheader("Forecast explorer")
        sku=st.selectbox("SKU",sorted(frame["sku"].astype(str).unique()))
        st.dataframe(frame[frame["sku"].astype(str)==sku],use_container_width=True)
