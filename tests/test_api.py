from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


def test_health_and_metadata() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        metadata = client.get("/metadata")
        assert metadata.status_code == 200
        payload = metadata.json()
        assert len(payload["stores"]) == 5
        assert len(payload["products"]) == 50


def test_forecast_recommend_anomalies_and_insights() -> None:
    with TestClient(app) as client:
        metadata = client.get("/metadata").json()
        store_id = metadata["stores"][0]
        product_id = metadata["products"][0]["product_id"]

        forecast = client.post("/forecast", json={"store_id": store_id, "product_id": product_id, "horizon_days": 30})
        assert forecast.status_code == 200
        forecast_payload = forecast.json()
        assert len(forecast_payload["forecast"]) == 30
        assert forecast_payload["summary"]["recommended_reorder_quantity"] >= 0

        recommend = client.post("/recommend", json={"product_id": product_id, "top_k": 5})
        assert recommend.status_code == 200
        assert "similar_products" in recommend.json()

        anomalies = client.post("/anomalies", json={"store_id": store_id, "product_id": product_id})
        assert anomalies.status_code == 200
        assert "anomalies" in anomalies.json()

        insight = client.post("/insights", json={"store_id": store_id, "product_id": product_id, "horizon_days": 30})
        assert insight.status_code == 200
        assert insight.json()["summary"]
