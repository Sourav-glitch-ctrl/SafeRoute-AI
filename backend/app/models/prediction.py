from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ---------------------------------------------------------
    # Input features
    # ---------------------------------------------------------

    distance_mi: Mapped[float] = mapped_column(Float)
    year: Mapped[int] = mapped_column(Integer)

    start_lng: Mapped[float] = mapped_column(Float)
    start_lat: Mapped[float] = mapped_column(Float)

    pressure_in: Mapped[float] = mapped_column(Float)
    temperature_f: Mapped[float] = mapped_column(Float)

    month: Mapped[int] = mapped_column(Integer)
    humidity_percent: Mapped[float] = mapped_column(Float)

    hour: Mapped[int] = mapped_column(Integer)
    wind_speed_mph: Mapped[float] = mapped_column(Float)

    quarter: Mapped[int] = mapped_column(Integer)
    day_of_week: Mapped[int] = mapped_column(Integer)

    traffic_signal: Mapped[int] = mapped_column(Integer)
    weather_category: Mapped[str] = mapped_column(String(100))
    visibility_mi: Mapped[float] = mapped_column(Float)

    near_road_infrastructure: Mapped[int] = mapped_column(Integer)
    crossing: Mapped[int] = mapped_column(Integer)
    junction: Mapped[int] = mapped_column(Integer)

    is_weekend: Mapped[int] = mapped_column(Integer)
    is_night: Mapped[int] = mapped_column(Integer)
    is_rush_hour: Mapped[int] = mapped_column(Integer)

    precipitation_in: Mapped[float] = mapped_column(Float)

    morning_rush_hour: Mapped[int] = mapped_column(Integer)
    evening_rush_hour: Mapped[int] = mapped_column(Integer)

    stop: Mapped[int] = mapped_column(Integer)
    has_precipitation: Mapped[int] = mapped_column(Integer)
    low_visibility: Mapped[int] = mapped_column(Integer)
    railway: Mapped[int] = mapped_column(Integer)

    # ---------------------------------------------------------
    # Prediction result
    # ---------------------------------------------------------

    severity: Mapped[int] = mapped_column(Integer)

    # Probability for each severity class
    probability_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_3: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability_4: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )