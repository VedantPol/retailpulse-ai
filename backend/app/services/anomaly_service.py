from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.utils.data_loader import load_sales_data


class AnomalyService:
    """Detects abnormal sales spikes and drops using rolling z-scores."""

    def __init__(self, settings: Settings):
        self.sales = load_sales_data(settings)

    @staticmethod
    def _severity(z_score: float) -> str:
        z = abs(z_score)
        if z >= 3.8:
            return "High"
        if z >= 2.8:
            return "Medium"
        return "Low"

    @staticmethod
    def _reason(row: pd.Series, anomaly_type: str) -> str:
        if int(row.get("promotion_flag", 0)) and anomaly_type == "Spike":
            return "promotion effect"
        if int(row.get("day_of_week", 0)) >= 5 and anomaly_type == "Spike":
            return "weekend spike"
        if float(row.get("inventory_level", 0)) < max(8, float(row.get("expected_sales", 0)) * 1.2) and anomaly_type == "Drop":
            return "stockout drop"
        if float(row.get("discount", 0)) >= 0.15:
            return "price change"
        return "unusual demand"

    def detect(self, store_id: str, product_id: str) -> dict[str, Any]:
        group = self.sales[(self.sales["store_id"] == store_id) & (self.sales["product_id"] == product_id)].copy()
        if group.empty:
            raise ValueError(f"No sales history found for store={store_id}, product={product_id}")
        group = group.sort_values("date").reset_index(drop=True)
        shifted = group["units_sold"].shift(1)
        group["expected_sales"] = shifted.rolling(21, min_periods=7).mean()
        group["rolling_std"] = shifted.rolling(21, min_periods=7).std().replace(0, np.nan)
        group["z_score"] = (group["units_sold"] - group["expected_sales"]) / group["rolling_std"]
        anomalies = group[group["z_score"].abs() >= 2.4].copy()
        anomalies["anomaly_type"] = np.where(anomalies["units_sold"] >= anomalies["expected_sales"], "Spike", "Drop")
        anomalies["severity"] = anomalies["z_score"].apply(self._severity)
        anomalies["reason"] = anomalies.apply(lambda row: self._reason(row, str(row["anomaly_type"])), axis=1)
        anomalies = anomalies.tail(50)
        records = []
        for _, row in anomalies.iterrows():
            records.append(
                {
                    "date": pd.to_datetime(row["date"]).date().isoformat(),
                    "actual_sales": int(row["units_sold"]),
                    "expected_sales": round(float(row["expected_sales"]), 2),
                    "anomaly_type": row["anomaly_type"],
                    "severity": row["severity"],
                    "reason": row["reason"],
                    "z_score": round(float(row["z_score"]), 2),
                }
            )
        series = group.tail(180)[["date", "units_sold"]].copy()
        series["expected_sales"] = group.tail(180)["expected_sales"].round(2).fillna(0)
        series["date"] = pd.to_datetime(series["date"]).dt.date.astype(str)
        return {
            "product": {
                "store_id": store_id,
                "product_id": product_id,
                "product_name": group.iloc[-1]["product_name"],
                "category": group.iloc[-1]["category"],
            },
            "anomalies": records,
            "series": series.to_dict(orient="records"),
        }

