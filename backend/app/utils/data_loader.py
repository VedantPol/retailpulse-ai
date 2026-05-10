from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings


def _add_backend_to_path() -> None:
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.append(str(backend_root))


def ensure_artifacts(settings: Settings) -> None:
    """Generate synthetic data and train models when artifacts are absent."""

    required_files = [
        settings.data_dir / "retail_sales.csv",
        settings.models_dir / "forecast_model.joblib",
        settings.models_dir / "recommender.joblib",
        settings.metrics_dir / "model_metrics.json",
        settings.metrics_dir / "feature_importance.json",
    ]
    for folder in [
        settings.data_dir,
        settings.models_dir,
        settings.metrics_dir,
        settings.reports_dir,
        settings.artifacts_dir / "plots",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    if all(path.exists() for path in required_files):
        return

    _add_backend_to_path()
    from training.generate_data import generate_retail_data
    from training.train_forecast_model import train_forecast_model
    from training.train_recommender import train_recommender

    data_path = settings.data_dir / "retail_sales.csv"
    if not data_path.exists():
        generate_retail_data(output_path=data_path, seed=settings.data_seed)

    train_forecast_model(
        data_path=data_path,
        model_dir=settings.models_dir,
        metrics_dir=settings.metrics_dir,
        reports_dir=settings.reports_dir,
    )
    train_recommender(
        data_path=data_path,
        model_dir=settings.models_dir,
        reports_dir=settings.reports_dir,
    )


def load_sales_data(settings: Settings) -> pd.DataFrame:
    df = pd.read_csv(settings.data_dir / "retail_sales.csv", parse_dates=["date"])
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

