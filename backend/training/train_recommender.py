from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def train_recommender(data_path: Path, model_dir: Path, reports_dir: Path) -> dict[str, object]:
    """Build product-product recommendations from demand profile similarity."""

    df = pd.read_csv(data_path, parse_dates=["date"])
    product_meta = df.groupby("product_id").agg(
        product_name=("product_name", "first"),
        category=("category", "first"),
        avg_price=("price", "mean"),
        avg_discount=("discount", "mean"),
        avg_units=("units_sold", "mean"),
        promo_rate=("promotion_flag", "mean"),
        revenue=("revenue", "sum"),
    )
    pivot = df.pivot_table(index="product_id", columns="day_of_week", values="units_sold", aggfunc="mean").fillna(0)
    pivot.columns = [f"dow_{col}" for col in pivot.columns]
    category_dummies = pd.get_dummies(product_meta["category"], prefix="cat")
    feature_frame = pd.concat([product_meta[["avg_price", "avg_discount", "avg_units", "promo_rate", "revenue"]], pivot, category_dummies], axis=1).fillna(0)
    scaled = StandardScaler().fit_transform(feature_frame)
    similarity = cosine_similarity(scaled)
    product_ids = list(feature_frame.index)

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "product_ids": product_ids,
        "product_meta": product_meta.reset_index().to_dict(orient="records"),
        "similarity": similarity,
        "feature_columns": list(feature_frame.columns),
    }
    joblib.dump(bundle, model_dir / "recommender.joblib")

    top_rows: list[dict[str, object]] = []
    for idx, product_id in enumerate(product_ids):
        order = np.argsort(similarity[idx])[::-1]
        for other_idx in order[1:6]:
            top_rows.append(
                {
                    "product_id": product_id,
                    "recommended_product_id": product_ids[other_idx],
                    "similarity_score": round(float(similarity[idx, other_idx]), 4),
                }
            )
    pd.DataFrame(top_rows).to_json(reports_dir / "recommendation_samples.json", orient="records", indent=2)
    return bundle


if __name__ == "__main__":
    train_recommender(
        data_path=Path("backend/artifacts/data/retail_sales.csv"),
        model_dir=Path("backend/artifacts/models"),
        reports_dir=Path("backend/artifacts/reports"),
    )

