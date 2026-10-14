"""Pydantic API contracts."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ForecastCreate(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    forecast_date: date
    horizon_days: int = Field(default=7, ge=1, le=28)
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    model_version: str = Field(
        default="day04-quantile-lgbm",
        min_length=1,
        max_length=64,
    )

    @model_validator(mode="after")
    def ordered_quantiles(self):
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("Forecasts must satisfy p10 <= p50 <= p90.")
        return self


class ForecastResponse(ForecastCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int


class InventoryRequest(BaseModel):
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    on_hand: float = Field(ge=0)
    minimum_order_quantity: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def ordered_quantiles(self):
        if not self.p10 <= self.p50 <= self.p90:
            raise ValueError("Forecasts must satisfy p10 <= p50 <= p90.")
        return self
