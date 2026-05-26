from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "day_of_week",
    "month",
    "weekend_flag",
    "promotion_flag",
    "holiday_flag",
    "price",
    "discount",
    "store_id_encoded",
    "product_id_encoded",
    "category_encoded",
    "inventory_level",
]

FORECAST_BLEND_WEIGHT = 0.45


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    wmape = np.sum(np.abs(y_true - y_pred)) / max(float(np.sum(np.abs(y_true))), 1.0) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "mape": round(float(mape), 3),
        "wmape": round(float(wmape), 3),
        "forecast_accuracy": round(float(max(0.0, 100 - wmape)), 3),
        "r2_score": round(float(r2), 4),
    }


def _encode_categories(df: pd.DataFrame, mappings: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    df = df.copy()
    mappings = mappings or {}
    for col in ["store_id", "product_id", "category"]:
        if col not in mappings:
            values = sorted(df[col].astype(str).unique())
            mappings[col] = {value: idx for idx, value in enumerate(values)}
        df[f"{col}_encoded"] = df[col].astype(str).map(mappings[col]).fillna(-1).astype(int)
    return df, mappings


def build_features(df: pd.DataFrame, mappings: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Create tabular time-series features for supervised demand forecasting."""

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["store_id", "product_id", "date"])
    frame["weekend_flag"] = (frame["day_of_week"] >= 5).astype(int)
    grouped = frame.groupby(["store_id", "product_id"], group_keys=False)
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = grouped["units_sold"].shift(lag)
    shifted = grouped["units_sold"].shift(1)
    frame["rolling_mean_7"] = shifted.groupby([frame["store_id"], frame["product_id"]]).rolling(7, min_periods=3).mean().reset_index(level=[0, 1], drop=True)
    frame["rolling_mean_14"] = shifted.groupby([frame["store_id"], frame["product_id"]]).rolling(14, min_periods=5).mean().reset_index(level=[0, 1], drop=True)
    frame["rolling_std_7"] = shifted.groupby([frame["store_id"], frame["product_id"]]).rolling(7, min_periods=3).std().reset_index(level=[0, 1], drop=True)
    frame, mappings = _encode_categories(frame, mappings)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["units_sold"]).reset_index(drop=True)
    return frame, mappings


def _build_model() -> tuple[Any, str]:
    try:
        from lightgbm import LGBMRegressor

        return (
            LGBMRegressor(
                objective="regression",
                n_estimators=260,
                learning_rate=0.055,
                num_leaves=48,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "LightGBM",
        )
    except Exception:
        return (
            HistGradientBoostingRegressor(
                max_iter=230,
                learning_rate=0.06,
                max_leaf_nodes=31,
                random_state=42,
                l2_regularization=0.05,
            ),
            "sklearn HistGradientBoostingRegressor",
        )


def _feature_importance(model: Any, x_val: pd.DataFrame, y_val: pd.Series) -> list[dict[str, float | str]]:
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    else:
        sample_size = min(4500, len(x_val))
        sample_x = x_val.sample(sample_size, random_state=42)
        sample_y = y_val.loc[sample_x.index]
        raw = permutation_importance(model, sample_x, sample_y, n_repeats=4, random_state=42, n_jobs=-1).importances_mean
    total = float(np.sum(np.abs(raw))) or 1.0
    rows = [
        {"feature": feature, "importance": round(float(abs(value) / total), 5)}
        for feature, value in zip(FEATURE_COLUMNS, raw)
    ]
    return sorted(rows, key=lambda item: float(item["importance"]), reverse=True)


def _recursive_feature_row(
    group: pd.DataFrame,
    history: list[float],
    target_date: pd.Timestamp,
    mappings: dict[str, dict[str, int]],
) -> dict[str, float]:
    latest = group.iloc[-1]
    recent = group.tail(90)
    lag = lambda n: history[-n] if len(history) >= n else float(np.mean(history))
    rolling_7 = np.array(history[-7:] if len(history) >= 7 else history, dtype=float)
    rolling_14 = np.array(history[-14:] if len(history) >= 14 else history, dtype=float)
    dow = int(target_date.dayofweek)
    month = int(target_date.month)
    weekend = int(dow >= 5)
    category = str(latest["category"])
    promo_rate = float(recent["promotion_flag"].mean())
    promo_gap = int(round(1 / promo_rate)) if promo_rate > 0 else 0
    future_promo = int(promo_gap >= 7 and target_date.dayofyear % promo_gap == 0)
    promo_discount = float(recent.loc[recent["promotion_flag"] == 1, "discount"].mean()) if bool((recent["promotion_flag"] == 1).any()) else 0.16
    discount = round(promo_discount if future_promo else 0.0, 2)
    base_price = float(recent["price"].median())
    price = max(0.99, base_price * (1 - discount))
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
        "store_id_encoded": int(mappings.get("store_id", {}).get(str(latest["store_id"]), -1)),
        "product_id_encoded": int(mappings.get("product_id", {}).get(str(latest["product_id"]), -1)),
        "category_encoded": int(mappings.get("category", {}).get(category, -1)),
        "inventory_level": inventory_projection,
    }


def _seasonal_prediction(group: pd.DataFrame, history: list[float], target_date: pd.Timestamp) -> float:
    recent = group.tail(180)
    dow = int(target_date.dayofweek)
    same_dow = recent[recent["day_of_week"] == dow].tail(8)["units_sold"].astype(float)
    dow_mean = float(same_dow.mean()) if len(same_dow) >= 3 else float(recent["units_sold"].tail(28).mean())
    short_level = float(np.mean(history[-14:] if len(history) >= 14 else history))
    long_level = float(recent["units_sold"].tail(90).mean())
    trend = 0.0
    if len(history) >= 28:
        trend = float(np.mean(history[-7:]) - np.mean(history[-28:-21]))
    return max(0.0, 0.55 * dow_mean + 0.35 * short_level + 0.10 * long_level + 0.08 * trend)


def _recursive_backtest(
    model: Any,
    sales_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    mappings: dict[str, dict[str, int]],
    feature_columns: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sales = sales_df.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    sales = sales.sort_values(["store_id", "product_id", "date"])

    for (store_id, product_id), group in sales.groupby(["store_id", "product_id"]):
        train_group = group[group["date"] <= cutoff].reset_index(drop=True)
        val_group = group[group["date"] > cutoff].reset_index(drop=True)
        if train_group.empty or val_group.empty:
            continue

        history = train_group["units_sold"].astype(float).tolist()
        for step, actual in enumerate(val_group.itertuples(index=False), start=1):
            row = _recursive_feature_row(train_group, history, pd.Timestamp(actual.date), mappings)
            model_prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=feature_columns))[0]))
            seasonal_prediction = _seasonal_prediction(train_group, history, pd.Timestamp(actual.date))
            prediction = FORECAST_BLEND_WEIGHT * model_prediction + (1 - FORECAST_BLEND_WEIGHT) * seasonal_prediction
            history.append(prediction)
            rows.append(
                {
                    "store_id": store_id,
                    "product_id": product_id,
                    "category": actual.category,
                    "horizon": step,
                    "actual": float(actual.units_sold),
                    "prediction": prediction,
                }
            )

    backtest = pd.DataFrame(rows)
    first_30 = backtest[backtest["horizon"] <= 30]
    grouped_30 = first_30.groupby(["store_id", "product_id", "category"]).agg(actual=("actual", "sum"), prediction=("prediction", "sum"))
    cumulative_wmape = (
        np.sum(np.abs(grouped_30["actual"].to_numpy() - grouped_30["prediction"].to_numpy()))
        / max(float(np.sum(grouped_30["actual"].to_numpy())), 1.0)
        * 100
    )
    total_error = abs(float(grouped_30["actual"].sum()) - float(grouped_30["prediction"].sum())) / max(float(grouped_30["actual"].sum()), 1.0) * 100

    return {
        "recursive_30_day_daily": _regression_metrics(first_30["actual"].to_numpy(), first_30["prediction"].to_numpy()),
        "recursive_60_day_daily": _regression_metrics(backtest["actual"].to_numpy(), backtest["prediction"].to_numpy()),
        "recursive_30_day_cumulative": {
            "wmape": round(float(cumulative_wmape), 3),
            "forecast_accuracy": round(float(max(0.0, 100 - cumulative_wmape)), 3),
            "total_demand_error": round(float(total_error), 3),
            "total_demand_accuracy": round(float(max(0.0, 100 - total_error)), 3),
            "series_count": int(len(grouped_30)),
        },
    }


def train_forecast_model(data_path: Path, model_dir: Path, metrics_dir: Path, reports_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(data_path, parse_dates=["date"])
    features, mappings = build_features(df)
    cutoff = features["date"].max() - pd.Timedelta(days=60)
    train_df = features[features["date"] <= cutoff]
    val_df = features[features["date"] > cutoff]
    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["units_sold"]
    x_val, y_val = val_df[FEATURE_COLUMNS], val_df["units_sold"]

    model, algorithm = _build_model()
    model.fit(x_train, y_train)
    val_pred = np.clip(model.predict(x_val), 0, None)
    residuals = y_val.to_numpy() - val_pred
    validation_metrics = _regression_metrics(y_val.to_numpy(), val_pred)
    recursive_metrics = _recursive_backtest(model, df, cutoff, mappings, FEATURE_COLUMNS)
    training_date = datetime.now(timezone.utc).isoformat()
    version = f"retailpulse-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    importance = _feature_importance(model, x_val, y_val)

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "algorithm": algorithm,
        "feature_columns": FEATURE_COLUMNS,
        "category_mappings": mappings,
        "training_date": training_date,
        "model_version": version,
        "residual_std": float(np.std(residuals)),
        "forecast_blend_weight": FORECAST_BLEND_WEIGHT,
    }
    joblib.dump(bundle, model_dir / "forecast_model.joblib")

    metrics = {
        "model_version": version,
        "training_date": training_date,
        "algorithm": algorithm,
        "rows_used_for_training": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "mae": validation_metrics["mae"],
        "rmse": validation_metrics["rmse"],
        "mape": validation_metrics["mape"],
        "wmape": validation_metrics["wmape"],
        "forecast_accuracy": validation_metrics["forecast_accuracy"],
        "r2_score": validation_metrics["r2_score"],
        "residual_std": round(float(np.std(residuals)), 3),
        "forecast_blend_weight": FORECAST_BLEND_WEIGHT,
        "recursive_backtest": recursive_metrics,
    }
    with (metrics_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (metrics_dir / "feature_importance.json").open("w", encoding="utf-8") as file:
        json.dump(importance, file, indent=2)

    category_errors = (
        val_df.assign(abs_error=np.abs(y_val.to_numpy() - val_pred), ape=np.abs(y_val.to_numpy() - val_pred) / np.maximum(y_val.to_numpy(), 1) * 100)
        .groupby("category")
        .agg(mae=("abs_error", "mean"), mape=("ape", "mean"), rows=("abs_error", "size"))
        .reset_index()
    )
    category_errors[["mae", "mape"]] = category_errors[["mae", "mape"]].round(3)
    category_errors.to_json(metrics_dir / "category_errors.json", orient="records", indent=2)

    sample = val_df[["date", "store_id", "product_id", "product_name", "category", "units_sold"]].copy()
    sample["prediction"] = np.round(val_pred, 2)
    sample.tail(500).to_json(reports_dir / "sample_predictions.json", orient="records", date_format="iso", indent=2)
    return metrics


if __name__ == "__main__":
    train_forecast_model(
        data_path=Path("backend/artifacts/data/retail_sales.csv"),
        model_dir=Path("backend/artifacts/models"),
        metrics_dir=Path("backend/artifacts/metrics"),
        reports_dir=Path("backend/artifacts/reports"),
    )
