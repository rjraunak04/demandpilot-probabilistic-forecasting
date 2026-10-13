"""Forecast persistence schema."""
from datetime import date
from sqlalchemy import Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
class Base(DeclarativeBase): pass
class ForecastRecord(Base):
    __tablename__="forecasts"
    __table_args__=(UniqueConstraint("sku","forecast_date"),)
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    sku: Mapped[str]=mapped_column(String(64),index=True)
    forecast_date: Mapped[date]=mapped_column(Date,index=True)
    horizon_days: Mapped[int]=mapped_column(Integer,default=7)
    p10: Mapped[float]=mapped_column(Float)
    p50: Mapped[float]=mapped_column(Float)
    p90: Mapped[float]=mapped_column(Float)
    model_version: Mapped[str]=mapped_column(String(64),default="day04-quantile-lgbm")
