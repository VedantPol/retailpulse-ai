from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.training_proxy import FEATURE_COLUMNS
from app.utils.data_loader import load_json, load_sales_data
from app.utils.model_loader import load_joblib


@dataclass
class ForecastContext:
    sales: pd.DataFrame
    model_bundle: dict[str, Any]
    metrics: dict[str, Any]
    feature_importance: list[dict[str, Any]]
    category_errors: list[dict[str, Any]]


class ForecastService:
    """Generates recursive SKU-level forecasts and inventory recommendations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ctx = ForecastContext(
            sales=load_sales_data(settings),
            model_bundle=load_joblib(settings.models_dir / "forecast_model.joblib"),
            metrics=load_json(settings.metrics_dir / "model_metrics.json", {}),
            feature_importance=load_json(settings.metrics_dir / "feature_importance.json", []),
            category_errors=load_json(settings.metrics_dir / "category_errors.json", []),
        )
        self.model = self.ctx.model_bundle["model"]
        self.feature_columns = self.ctx.model_bundle.get("feature_columns", FEATURE_COLUMNS)
        self.mappings = self.ctx.model_bundle.get("category_mappings", {})

    def metadata(self) -> dict[str, Any]:
        products = (
            self.ctx.sales[["product_id", "product_name", "category"]]
            .drop_duplicates()
            .sort_values(["category", "product_id"])
            .to_dict(orient="records")
        )
        return {
            "stores": sorted(self.ctx.sales["store_id"].unique().tolist()),
            "categories": sorted(self.ctx.sales["category"].unique().tolist()),
            "products": products,
            "model_version": self.ctx.metrics.get("model_version", "unknown"),
            "training_date": self.ctx.metrics.get("training_date", "unknown"),
        }

    def _encode(self, field: str, value: str) -> int:
        return int(self.mappings.get(field, {}).get(value, -1))

    def _latest_group(self, store_id: str, product_id: str) -> pd.DataFrame:
        group = self.ctx.sales[(self.ctx.sales["store_id"] == store_id) & (self.ctx.sales["product_id"] == product_id)].copy()
        if group.empty:
            raise ValueError(f"No sales history found for store={store_id}, product={product_id}")
        return group.sort_values("date").reset_index(drop=True)

    def _future_row(self, group: pd.DataFrame, history: list[float], target_date: pd.Timestamp) -> dict[str, float]:
        latest = group.iloc[-1]
        lag = lambda n: history[-n] if len(history) >= n else float(np.mean(history))
        rolling_7 = np.array(history[-7:] if len(history) >= 7 else history, dtype=float)
        rolling_14 = np.array(history[-14:] if len(history) >= 14 else history, dtype=float)
        dow = int(target_date.dayofweek)
        month = int(target_date.month)
        weekend = int(dow >= 5)
        category = str(latest["category"])
        future_promo = int(weekend and category in {"Grocery", "Beverages", "Apparel"} and target_date.day % 3 == 0)
        discount = 0.12 if future_promo else 0.0
        price = max(0.99, float(latest["price"]) * (1 - discount))
        inventory_projection = max(0.0, float(latest["inventory_level"]) - float(np.sum(history[-7:]) if history else 0) * 0.1)
        return {
            "lag_1": lag(1),
            "lag_7": lag(7),
            "lag_14": lag(14),
            "lag_28": lag(28),
            "rolling_mean_7": float(rolling_7.mean()),
            "rolling_mean_14": float(rolling_14.mean()),
            "rolling_std_7": float(rolling_7.std(ddof=0)),
            "day_of_week": dow,
            "month": month,
            "weekend_flag": weekend,
            "promotion_flag": future_promo,
            "holiday_flag": int((target_date.month, target_date.day) in {(1, 1), (7, 4), (11, 25), (12, 24), (12, 25)}),
            "price": price,
            "discount": discount,
            "store_id_encoded": self._encode("store_id", str(latest["store_id"])),
            "product_id_encoded": self._encode("product_id", str(latest["product_id"])),
            "category_encoded": self._encode("category", category),
            "inventory_level": inventory_projection,
        }

    @staticmethod
    def _risk_label(days_of_cover: float) -> str:
        if days_of_cover < 12:
            return "High"
        if days_of_cover < 24:
            return "Medium"
        return "Low"

    def forecast(self, store_id: str, product_id: str, horizon_days: int = 30) -> dict[str, Any]:
        start = time.perf_counter()
        group = self._latest_group(store_id, product_id)
        history = group["units_sold"].astype(float).tolist()
        latest_date = pd.to_datetime(group["date"].max())
        residual_std = float(self.ctx.metrics.get("residual_std", self.ctx.model_bundle.get("residual_std", 3.0)))

        forecast_rows: list[dict[str, Any]] = []
        for step in range(1, horizon_days + 1):
            target_date = latest_date + timedelta(days=step)
            row = self._future_row(group, history, target_date)
            x = pd.DataFrame([row], columns=self.feature_columns)
            prediction = max(0.0, float(self.model.predict(x)[0]))
            widening = 1 + step / max(horizon_days, 1) * 0.35
            lower = max(0.0, prediction - 1.35 * residual_std * widening)
            upper = prediction + 1.35 * residual_std * widening
            history.append(prediction)
            forecast_rows.append(
                {
                    "date": target_date.date().isoformat(),
                    "predicted_units": round(prediction, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                    "promotion_flag": row["promotion_flag"],
                    "holiday_flag": row["holiday_flag"],
                }
            )

        forecast_total = float(sum(item["predicted_units"] for item in forecast_rows))
        avg_daily = forecast_total / horizon_days
        current_inventory = int(group.iloc[-1]["inventory_level"])
        days_of_cover = current_inventory / max(avg_daily, 0.1)
        stockout_risk = self._risk_label(days_of_cover)
        reorder_quantity = max(0, int(round(forecast_total * 1.15 - current_inventory)))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        historical = group.tail(180)[["date", "units_sold", "inventory_level", "price", "promotion_flag", "holiday_flag"]].copy()
        historical["date"] = pd.to_datetime(historical["date"]).dt.date.astype(str)
        product_info = group.iloc[-1][["product_id", "product_name", "category", "store_id"]].to_dict()
        return {
            "product": product_info,
            "historical": historical.to_dict(orient="records"),
            "forecast": forecast_rows,
            "summary": {
                "forecasted_30_day_demand": round(forecast_total, 1),
                "average_daily_demand": round(avg_daily, 1),
                "stockout_risk": stockout_risk,
                "current_inventory": current_inventory,
                "recommended_reorder_quantity": reorder_quantity,
                "inventory_days_of_cover": round(days_of_cover, 1),
                "prediction_latency_ms": latency_ms,
            },
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "metrics": self.ctx.metrics,
            "feature_importance": self.ctx.feature_importance,
            "category_errors": self.ctx.category_errors,
            "api_health": "healthy",
        }

