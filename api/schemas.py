from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CarRequest(BaseModel):
    """Сырые признаки автомобиля, которые ожидает preprocessing pipeline."""

    model_config = ConfigDict(extra="forbid")

    manufacturer: str
    model: str | None = None
    year: int = Field(..., ge=2000, le=2035)
    mileage: float = Field(..., ge=0, le=800_000)
    engine: str | None = None
    transmission: str | None = None
    drivetrain: str | None = None
    fuel_type: str | None = None
    mpg: str | None = None
    exterior_color: str | None = None
    interior_color: str | None = None
    accidents_or_damage: float | None = Field(default=None, ge=0, le=1)
    one_owner: float | None = Field(default=None, ge=0, le=1)
    personal_use_only: float | None = Field(default=None, ge=0, le=1)
    seller_name: str | None = None
    seller_rating: float | None = Field(default=None, ge=0, le=5)
    driver_rating: float | None = Field(default=None, ge=0, le=5)
    driver_reviews_num: float | None = Field(default=None, ge=0)
    price_drop: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    model_version: str
