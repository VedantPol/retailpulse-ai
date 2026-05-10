from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES = {
    "Grocery": 1.35,
    "Beverages": 1.18,
    "Personal Care": 0.9,
    "Household": 0.78,
    "Electronics": 0.38,
    "Apparel": 0.62,
}


@dataclass(frozen=True)
class ProductSpec:
    product_id: str
    product_name: str
    category: str
    base_price: float
    demand_multiplier: float
    price_sensitivity: float


def _build_products(rng: np.random.Generator) -> list[ProductSpec]:
    names = {
        "Grocery": ["Organic Rice", "Breakfast Cereal", "Pasta Pack", "Cooking Oil", "Snack Mix", "Premium Flour", "Soup Can", "Frozen Meal", "Granola Bar"],
        "Beverages": ["Sparkling Water", "Cold Brew", "Orange Juice", "Green Tea", "Energy Drink", "Protein Shake", "Lemon Soda", "Herbal Tea"],
        "Personal Care": ["Shampoo", "Body Wash", "Toothpaste", "Face Cream", "Hand Soap", "Deodorant", "Sunscreen", "Conditioner"],
        "Household": ["Laundry Pods", "Dish Soap", "Paper Towels", "Trash Bags", "Surface Cleaner", "Air Freshener", "Sponges", "Foil Wrap"],
        "Electronics": ["USB Charger", "Earbuds", "Smart Bulb", "Phone Cable", "Battery Pack", "Travel Adapter", "Desk Lamp", "Mouse"],
        "Apparel": ["Cotton Tee", "Athletic Socks", "Hoodie", "Denim Jeans", "Cap", "Workout Shorts", "Polo Shirt", "Scarf"],
    }
    price_ranges = {
        "Grocery": (3.0, 18.0),
        "Beverages": (2.0, 10.0),
        "Personal Care": (4.0, 24.0),
        "Household": (5.0, 28.0),
        "Electronics": (12.0, 95.0),
        "Apparel": (9.0, 70.0),
    }
    products: list[ProductSpec] = []
    sku = 1
    category_cycle = list(CATEGORIES)
    while len(products) < 50:
        category = category_cycle[(sku - 1) % len(category_cycle)]
        base_name = names[category][(sku - 1) % len(names[category])]
        low, high = price_ranges[category]
        speed = rng.choice([0.55, 0.85, 1.0, 1.25, 1.65], p=[0.12, 0.2, 0.36, 0.22, 0.1])
        products.append(
            ProductSpec(
                product_id=f"SKU_{sku:03d}",
                product_name=f"{base_name} {sku:03d}",
                category=category,
                base_price=round(float(rng.uniform(low, high)), 2),
                demand_multiplier=float(speed * rng.uniform(0.85, 1.2)),
                price_sensitivity=float(rng.uniform(0.35, 0.95)),
            )
        )
        sku += 1
    return products


def generate_retail_data(output_path: Path, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible two-year synthetic retail demand dataset."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=730, freq="D")
    stores = [f"STORE_{idx:03d}" for idx in range(1, 6)]
    store_multipliers = dict(zip(stores, rng.uniform(0.78, 1.28, size=len(stores))))
    products = _build_products(rng)
    rows: list[dict[str, object]] = []

    holiday_dates = set()
    for year in sorted({d.year for d in dates}):
        holiday_dates.update(pd.to_datetime([f"{year}-01-01", f"{year}-07-04", f"{year}-11-25", f"{year}-12-24", f"{year}-12-25"]))

    for store_id in stores:
        store_effect = store_multipliers[store_id]
        for product in products:
            inventory = int(rng.integers(80, 260) * product.demand_multiplier)
            stockout_timer = 0
            for date in dates:
                dow = int(date.dayofweek)
                month = int(date.month)
                weekend = dow >= 5
                holiday = int(date in holiday_dates or rng.random() < 0.012)
                promotion = int(rng.random() < (0.065 + (0.035 if product.category in {"Beverages", "Apparel"} else 0)))
                discount = round(float(rng.uniform(0.08, 0.32) if promotion else rng.choice([0.0, 0.03, 0.05], p=[0.75, 0.15, 0.1])), 2)
                price_noise = float(rng.normal(0, product.base_price * 0.025))
                price = max(0.99, round(product.base_price * (1 - discount) + price_noise, 2))

                category_effect = CATEGORIES[product.category]
                weekend_lift = 1.22 if weekend and product.category in {"Grocery", "Beverages", "Apparel"} else 1.06 if weekend else 1.0
                monthly = 1 + 0.12 * np.sin((month - 1) / 12 * 2 * np.pi)
                promo_lift = 1 + (0.55 + discount) * promotion
                holiday_lift = 1.18 if holiday else 1.0
                price_effect = max(0.35, 1 - product.price_sensitivity * ((price - product.base_price) / max(product.base_price, 1)))
                baseline = 8.5 * category_effect * store_effect * product.demand_multiplier
                expected = baseline * weekend_lift * monthly * promo_lift * holiday_lift * price_effect

                if rng.random() < 0.0035:
                    expected *= rng.uniform(2.1, 3.6)
                if stockout_timer > 0:
                    expected *= rng.uniform(0.05, 0.28)
                    stockout_timer -= 1
                if rng.random() < 0.0018:
                    stockout_timer = int(rng.integers(2, 6))

                noise = rng.normal(0, max(1.3, expected * 0.14))
                raw_units = max(0, int(round(expected + noise)))
                units_sold = min(raw_units, max(0, inventory))
                revenue = round(units_sold * price, 2)
                inventory = max(0, inventory - units_sold)
                if inventory < max(30, expected * 4) or date.dayofweek == 0 and rng.random() < 0.2:
                    inventory += int(rng.integers(90, 260) * product.demand_multiplier)

                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "store_id": store_id,
                        "product_id": product.product_id,
                        "product_name": product.product_name,
                        "category": product.category,
                        "price": price,
                        "discount": discount,
                        "promotion_flag": promotion,
                        "holiday_flag": holiday,
                        "day_of_week": dow,
                        "month": month,
                        "inventory_level": int(inventory),
                        "units_sold": int(units_sold),
                        "revenue": revenue,
                    }
                )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generate_retail_data(Path("backend/artifacts/data/retail_sales.csv"))

