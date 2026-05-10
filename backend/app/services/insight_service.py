from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings
from app.services.anomaly_service import AnomalyService
from app.services.forecast_service import ForecastService
from app.services.recommendation_service import RecommendationService


class InsightService:
    """Creates business-readable analyst summaries using LLMs or deterministic rules."""

    def __init__(
        self,
        settings: Settings,
        forecast_service: ForecastService,
        anomaly_service: AnomalyService,
        recommendation_service: RecommendationService,
    ):
        self.settings = settings
        self.forecast_service = forecast_service
        self.anomaly_service = anomaly_service
        self.recommendation_service = recommendation_service

    async def _openai_summary(self, prompt: str) -> str | None:
        if not self.settings.openai_api_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You are a concise retail analytics consultant."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 220,
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    async def _gemini_summary(self, prompt: str) -> str | None:
        if not self.settings.gemini_api_key:
            return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.settings.gemini_api_key}"
            async with httpx.AsyncClient(timeout=12) as client:
                response = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            return None

    @staticmethod
    def _fallback_summary(forecast: dict[str, Any], anomalies: dict[str, Any], recommendations: dict[str, Any]) -> dict[str, Any]:
        product = forecast["product"]
        summary = forecast["summary"]
        anomaly_count = len(anomalies["anomalies"])
        recent_anomaly = anomalies["anomalies"][-1] if anomaly_count else None
        lead = (
            f"Demand for {product['product_id']} ({product['product_name']}) is forecast at "
            f"{summary['forecasted_30_day_demand']} units over the next 30 days, averaging "
            f"{summary['average_daily_demand']} units per day."
        )
        risk = (
            f"Stockout risk is {summary['stockout_risk'].lower()} because current inventory covers about "
            f"{summary['inventory_days_of_cover']} forecasted days."
        )
        anomaly_text = (
            f"The latest notable anomaly was a {recent_anomaly['severity'].lower()} {recent_anomaly['anomaly_type'].lower()} "
            f"on {recent_anomaly['date']} likely driven by {recent_anomaly['reason']}."
            if recent_anomaly
            else "No major recent anomalies require immediate action."
        )
        bundle = recommendations["recommended_bundles"][0]["bundle_name"] if recommendations["recommended_bundles"] else "a same-category companion SKU"
        action = (
            f"Recommended action: reorder {summary['recommended_reorder_quantity']} units, monitor promotion timing, "
            f"and test a bundle with {bundle}."
        )
        return {
            "summary": " ".join([lead, risk, anomaly_text, action]),
            "risks": [summary["stockout_risk"], f"{anomaly_count} anomalies detected in recent history"],
            "recommended_actions": [
                f"Reorder {summary['recommended_reorder_quantity']} units",
                "Review upcoming promotions and weekend staffing",
                f"Promote bundle: {bundle}",
            ],
            "provider": "rule_based",
        }

    async def generate(self, store_id: str, product_id: str, horizon_days: int) -> dict[str, Any]:
        forecast = self.forecast_service.forecast(store_id, product_id, horizon_days)
        anomalies = self.anomaly_service.detect(store_id, product_id)
        recommendations = self.recommendation_service.recommend(product_id, top_k=3)
        prompt = (
            "Create a concise retail analyst summary using these model outputs. "
            f"Forecast summary: {forecast['summary']}. "
            f"Product: {forecast['product']}. "
            f"Recent anomalies: {anomalies['anomalies'][-5:]}. "
            f"Recommendations: {recommendations['recommended_bundles'][:2]}. "
            "Include demand outlook, stockout risk, anomaly explanation, and recommended action."
        )
        llm_text = await self._openai_summary(prompt) or await self._gemini_summary(prompt)
        if llm_text:
            return {
                "summary": llm_text,
                "risks": [forecast["summary"]["stockout_risk"], f"{len(anomalies['anomalies'])} anomalies detected"],
                "recommended_actions": [f"Reorder {forecast['summary']['recommended_reorder_quantity']} units", "Monitor anomalies", "Review bundles"],
                "provider": "llm",
            }
        return self._fallback_summary(forecast, anomalies, recommendations)
