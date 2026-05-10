"""Shared constants mirrored from training without importing heavy jobs at API import time."""

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

