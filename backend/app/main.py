from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import AnomalyRequest, ForecastRequest, InsightRequest, RecommendationRequest
from app.services.anomaly_service import AnomalyService
from app.services.forecast_service import ForecastService
from app.services.insight_service import InsightService
from app.services.recommendation_service import RecommendationService
from app.utils.data_loader import ensure_artifacts


settings = get_settings()
services: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_artifacts(settings)
    forecast_service = ForecastService(settings)
    anomaly_service = AnomalyService(settings)
    recommendation_service = RecommendationService(settings)
    services["forecast"] = forecast_service
    services["anomaly"] = anomaly_service
    services["recommendation"] = recommendation_service
    services["insight"] = InsightService(settings, forecast_service, anomaly_service, recommendation_service)
    yield


app = FastAPI(
    title="RetailPulse AI",
    description="Forecasting, recommendation, anomaly detection, and AI insight API for retail analytics.",
    version=settings.api_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service(name: str) -> Any:
    if name not in services:
        raise HTTPException(status_code=503, detail="Service is still initializing")
    return services[name]


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "RetailPulse AI API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    model_version = "initializing"
    if "forecast" in services:
        model_version = services["forecast"].ctx.metrics.get("model_version", "unknown")
    return {"status": "healthy", "service": settings.app_name, "model_version": model_version}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return _service("forecast").metadata()


@app.post("/forecast")
def forecast(payload: ForecastRequest) -> dict[str, Any]:
    try:
        return _service("forecast").forecast(payload.store_id, payload.product_id, payload.horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/recommend")
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    try:
        return _service("recommendation").recommend(payload.product_id, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/anomalies")
def anomalies(payload: AnomalyRequest) -> dict[str, Any]:
    try:
        return _service("anomaly").detect(payload.store_id, payload.product_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/insights")
async def insights(payload: InsightRequest) -> dict[str, Any]:
    try:
        return await _service("insight").generate(payload.store_id, payload.product_id, payload.horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _service("forecast").metrics()
