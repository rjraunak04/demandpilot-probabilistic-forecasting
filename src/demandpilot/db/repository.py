"""Forecast repository."""

from sqlalchemy import select

from demandpilot.db.models import ForecastRecord


class ForecastRepository:
    def __init__(self, session):
        self.session = session

    def add(self, record):
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get(self, sku, forecast_date):
        statement = select(ForecastRecord).where(
            ForecastRecord.sku == sku,
            ForecastRecord.forecast_date == forecast_date,
        )
        return self.session.scalar(statement)
