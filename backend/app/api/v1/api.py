from fastapi import APIRouter

from app.api.v1.endpoints import health
from app.api.v1.endpoints import prediction


api_router = APIRouter()


api_router.include_router(
    health.router,
    tags=["Health"],
)


api_router.include_router(
    prediction.router,
    prefix="/prediction",
    tags=["Prediction"],
)