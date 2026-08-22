from typing import List
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Distance_mi: float
    Year: int
    Start_Lng: float
    Start_Lat: float
    Pressure_in: float
    Temperature_F: float
    Month: int
    Humidity_percent: float
    Hour: int
    Wind_Speed_mph: float
    Quarter: int
    DayOfWeek: int

    Traffic_Signal: int
    Weather_Category: str
    Visibility_mi: float

    NearRoadInfrastructure: int
    Crossing: int
    Junction: int
    IsWeekend: int
    IsNight: int
    IsRushHour: int

    Precipitation_in: float
    MorningRushHour: int
    EveningRushHour: int
    Stop: int
    HasPrecipitation: int
    LowVisibility: int
    Railway: int


class PredictionResponse(BaseModel):
    severity: int
    probabilities: dict[int, float]


class RoutePoint(BaseModel):
    lat: float
    lng: float


class RoutePredictionRequest(BaseModel):
    points: List[RoutePoint]
    distance_mi: float


class RoutePredictionPoint(BaseModel):
    lat: float
    lng: float
    severity: int
    probability: float


class RoutePredictionResponse(BaseModel):
    overall_severity: int
    points: List[RoutePredictionPoint]