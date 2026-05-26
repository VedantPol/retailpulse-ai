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
        self.forecast_blend_weight = float(self.ctx.model_bundle.get("forecast_blend_weight", self.ctx.metrics.get("forecast_blend_weight", 0.35)))

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

    @staticmethod
    @staticmethod
    def _winsorized(values: pd.Series | np.ndarray, quantile: float = 0.92) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if len(array) == 0:
            return array
        return np.clip(array, 0, np.quantile(array, quantile))

    @classmethod
    def _profile_prediction(cls, group: pd.DataFrame, target_date: pd.Timestamp, horizon_step: int) -> float:
        recent = group.tail(180)
        units = recent["units_sold"].astype(float)
        winsorized_units = cls._winsorized(units, 0.92)
        recent_30 = cls._winsorized(recent.tail(30)["units_sold"], 0.90)
        recent_90 = cls._winsorized(recent.tail(90)["units_sold"], 0.92)
        level = 0.55 * float(np.mean(recent_30)) + 0.30 * float(np.mean(recent_90)) + 0.15 * float(np.mean(winsorized_units))

        previous_window = recent.iloc[-90:-30]["units_sold"] if len(recent) >= 90 else recent["units_sold"]
        previous = cls._winsorized(previous_window, 0.92)
        trend = (float(np.mean(recent_30)) - float(np.mean(previous))) / 60 if len(previous) else 0.0
        trend = float(np.clip(trend, -0.035 * max(level, 1.0), 0.035 * max(level, 1.0)))

        dow = int(target_date.dayofweek)
        overall = max(float(np.mean(winsorized_units)), 0.1)
        same_dow = cls._winsorized(recent[recent["day_of_week"] == dow].tail(10)["units_sold"], 0.90)
        dow_mean = float(np.mean(same_dow)) if len(same_dow) >= 3 else overall
        weekday_factor = float(np.clip(1 + 0.45 * (dow_mean / overall - 1), 0.88, 1.14))

        recent_promo_rate = float(0.65 * recent.tail(60)["promotion_flag"].mean() + 0.35 * recent["promotion_flag"].mean())
        non_promo = recent[recent["promotion_flag"] == 0]["units_sold"].astype(float)
        promo = recent[recent["promotion_flag"] == 1]["units_sold"].astype(float)
        non_promo_mean = float(np.mean(cls._winsorized(non_promo, 0.92))) if len(non_promo) else overall
        promo_mean = float(np.mean(cls._winsorized(promo, 0.85))) if len(promo) >= 3 else non_promo_mean
        promo_lift = float(np.clip(promo_mean / max(non_promo_mean, 0.1) - 1, 0, 0.7))
        expected_promo_multiplier = 1 + min(recent_promo_rate, 0.25) * promo_lift

        return max(0.1, (level + trend * horizon_step) * weekday_factor * expected_promo_multiplier)

    @staticmethod
    def _local_uncertainty(group: pd.DataFrame, fallback_std: float) -> float:
        recent = group.tail(120)["units_sold"].astype(float)
        if recent.empty:
            return fallback_std
        recent_std = float(recent.std(ddof=0))
        q25, q75 = recent.quantile([0.25, 0.75])
        iqr_std = float((q75 - q25) / 1.349) if q75 > q25 else recent_std
        local_std = max(0.75, min(fallback_std, max(iqr_std, recent_std * 0.85)))
        return float(local_std)

    def _future_row(self, group: pd.DataFrame, history: list[float], target_date: pd.Timestamp) -> dict[str, float]:
        latest = group.iloc[-1]
        recent = group.tail(90)
        lag = lambda n: history[-n] if len(history) >= n else float(np.mean(history))
        rolling_7 = np.array(history[-7:] if len(history) >= 7 else history, dtype=float)
        rolling_14 = np.array(history[-14:] if len(history) >= 14 else history, dtype=float)
        dow = int(target_date.dayofweek)
        month = int(target_date.month)
        weekend = int(dow >= 5)
        category = str(latest["category"])
        future_promo = 0
        discount = 0.0
        base_price = float(recent["price"].median())
        price = max(0.99, base_price)
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
        uncertainty_std = self._local_uncertainty(group, residual_std)

        forecast_rows: list[dict[str, Any]] = []
        for step in range(1, horizon_days + 1):
            target_date = latest_date + timedelta(days=step)
            row = self._future_row(group, history, target_date)
            x = pd.DataFrame([row], columns=self.feature_columns)
            model_prediction = max(0.0, float(self.model.predict(x)[0]))
            profile_prediction = self._profile_prediction(group, target_date, step)
            prediction = self.forecast_blend_weight * model_prediction + (1 - self.forecast_blend_weight) * profile_prediction
            widening = 1 + step / max(horizon_days, 1) * 0.25
            lower = max(0.0, prediction - 1.15 * uncertainty_std * widening)
            upper = prediction + 1.15 * uncertainty_std * widening
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
                "forecast_horizon_days": horizon_days,
                "forecasted_demand": round(forecast_total, 1),
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

