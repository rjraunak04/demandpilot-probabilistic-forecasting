"""FastAPI serving application."""

from datetime import date

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError

from demandpilot.api.schemas import ForecastCreate, ForecastResponse, InventoryRequest
from demandpilot.db.models import ForecastRecord
from demandpilot.db.repository import ForecastRepository
from demandpilot.db.session import get_session
from demandpilot.inventory.policy import InventoryPolicy, inventory_recommendations

app = FastAPI(title="DemandPilot API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(session=Depends(get_session)):
    try:
        session.connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ready"}


@app.post(
    "/forecasts",
    response_model=ForecastResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_forecast(payload: ForecastCreate, session=Depends(get_session)):
    try:
        return ForecastRepository(session).add(
            ForecastRecord(**payload.model_dump())
        )
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="forecast already exists",
        ) from exc


@app.get("/forecasts/{sku}", response_model=ForecastResponse)
def get_forecast(
    sku: str,
    forecast_date: date = Query(...),
    session=Depends(get_session),
):
    record = ForecastRepository(session).get(sku, forecast_date)
    if record is None:
        raise HTTPException(status_code=404, detail="forecast not found")
    return record


@app.post("/inventory/recommendation")
def inventory(payload: InventoryRequest):
    frame = pd.DataFrame(
        [payload.model_dump(include={"p10", "p50", "p90"})]
    )
    policy = InventoryPolicy(
        minimum_order_quantity=payload.minimum_order_quantity
    )
    row = inventory_recommendations(
        frame,
        on_hand=[payload.on_hand],
        policy=policy,
    ).iloc[0]
    return {
        "target_stock": float(row["target_stock"]),
        "safety_stock_proxy": float(row["safety_stock_proxy"]),
        "uncertainty_width": float(row["uncertainty_width"]),
        "stockout_risk": str(row["stockout_risk"]),
        "reorder_quantity": float(row["reorder_quantity"]),
    }
