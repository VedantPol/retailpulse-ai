from __future__ import annotations

from typing import Any

import numpy as np

from app.config import Settings
from app.utils.model_loader import load_joblib


class RecommendationService:
    """Returns similar, complementary, and bundle-style product suggestions."""

    def __init__(self, settings: Settings):
        self.bundle = load_joblib(settings.models_dir / "recommender.joblib")
        self.product_ids: list[str] = self.bundle["product_ids"]
        self.similarity = self.bundle["similarity"]
        self.meta = {row["product_id"]: row for row in self.bundle["product_meta"]}

    def recommend(self, product_id: str, top_k: int = 5) -> dict[str, Any]:
        if product_id not in self.product_ids:
            raise ValueError(f"Unknown product_id: {product_id}")
        idx = self.product_ids.index(product_id)
        base = self.meta[product_id]
        order = np.argsort(self.similarity[idx])[::-1]

        similar: list[dict[str, Any]] = []
        together: list[dict[str, Any]] = []
        bundles: list[dict[str, Any]] = []
        for other_idx in order:
            other_id = self.product_ids[int(other_idx)]
            if other_id == product_id:
                continue
            other = self.meta[other_id]
            score = round(float(self.similarity[idx, other_idx]), 3)
            same_category = other["category"] == base["category"]
            row = {
                "product_id": other_id,
                "product_name": other["product_name"],
                "category": other["category"],
                "similarity_score": score,
                "reason": (
                    "Similar demand curve, price tier, and category behavior"
                    if same_category
                    else "Complementary demand pattern with overlapping shopping missions"
                ),
            }
            if same_category and len(similar) < top_k:
                similar.append(row)
            if not same_category and len(together) < top_k:
                together.append(row | {"reason": f"Often pairs well with {base['category']} baskets based on demand timing"})
            if len(bundles) < top_k and (same_category or score > 0.28):
                bundles.append(
                    row
                    | {
                        "bundle_name": f"{base['product_name']} + {other['product_name']}",
                        "reason": "Bundle candidate with compatible demand profile and promotion response",
                    }
                )
            if len(similar) >= top_k and len(together) >= top_k and len(bundles) >= top_k:
                break

        return {
            "base_product": {
                "product_id": product_id,
                "product_name": base["product_name"],
                "category": base["category"],
            },
            "similar_products": similar[:top_k],
            "frequently_bought_together": together[:top_k],
            "recommended_bundles": bundles[:top_k],
        }

