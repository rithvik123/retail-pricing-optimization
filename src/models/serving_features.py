from __future__ import annotations

import numpy as np
import pandas as pd


LEAKAGE_FEATURES = (
    "num_baskets",
    "num_households",
    "total_retail_discount",
    "total_coupon_discount",
)

PRICING_BASE_NUMERIC_FEATURES = [
    "avg_unit_price",
    "median_unit_price",
    "discount_percentage",
    "is_display",
    "is_mailer",
    "week_no",
    "lag_quantity_1",
    "lag_quantity_2",
    "lag_quantity_4",
    "rolling_quantity_mean_4",
    "rolling_quantity_mean_8",
    "rolling_quantity_std_4",
    "price_change",
    "price_lag_1",
    "lag_num_baskets_1",
    "rolling_num_baskets_mean_4",
    "lag_num_households_1",
    "rolling_num_households_mean_4",
    "lag_total_retail_discount_1",
    "rolling_retail_discount_mean_4",
    "lag_total_coupon_discount_1",
    "rolling_coupon_discount_mean_4",
    "lag_discount_percentage_1",
    "rolling_discount_percentage_mean_4",
    "lag_is_display_1",
    "lag_is_mailer_1",
]

PRICING_DERIVED_FEATURES = [
    "log_avg_unit_price",
    "price_ratio_lag_1",
    "discount_x_display",
    "discount_x_mailer",
    "any_promotion",
    "lag_velocity_4",
    "lag_velocity_8",
]

PRICING_NUMERIC_FEATURES = PRICING_BASE_NUMERIC_FEATURES + PRICING_DERIVED_FEATURES

HISTORY_SOURCE_COLUMNS = {
    "num_baskets": ("lag_num_baskets_1", "rolling_num_baskets_mean_4"),
    "num_households": ("lag_num_households_1", "rolling_num_households_mean_4"),
    "total_retail_discount": ("lag_total_retail_discount_1", "rolling_retail_discount_mean_4"),
    "total_coupon_discount": ("lag_total_coupon_discount_1", "rolling_coupon_discount_mean_4"),
    "discount_percentage": ("lag_discount_percentage_1", "rolling_discount_percentage_mean_4"),
}

HISTORY_FLAG_COLUMNS = {
    "is_display": "lag_is_display_1",
    "is_mailer": "lag_is_mailer_1",
}


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        if isinstance(default, pd.Series):
            return pd.to_numeric(default, errors="coerce").fillna(0).astype(float)
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    ratio = pd.to_numeric(numerator, errors="coerce") / denominator
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def add_serving_history_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create lagged versions of outcome-adjacent fields without using same-week values."""
    frame = pd.DataFrame(features).copy()
    required = {"product_id", "store_id", "week_no"}
    if not required.issubset(frame.columns):
        return frame

    frame["__original_order"] = np.arange(len(frame))
    frame = frame.sort_values(["product_id", "store_id", "week_no"]).copy()
    group = frame.groupby(["product_id", "store_id"], observed=True, sort=False)

    for source, (lag_column, rolling_column) in HISTORY_SOURCE_COLUMNS.items():
        if source not in frame:
            frame[lag_column] = 0.0
            frame[rolling_column] = 0.0
            continue
        shifted = pd.to_numeric(group[source].shift(1), errors="coerce")
        frame[lag_column] = shifted.fillna(0)
        frame[rolling_column] = (
            shifted
            .groupby([frame["product_id"], frame["store_id"]], observed=True)
            .transform(lambda series: series.rolling(4, min_periods=1).mean())
            .fillna(0)
        )

    for source, lag_column in HISTORY_FLAG_COLUMNS.items():
        if source not in frame:
            frame[lag_column] = 0.0
            continue
        frame[lag_column] = pd.to_numeric(group[source].shift(1), errors="coerce").fillna(0)

    return frame.sort_values("__original_order").drop(columns="__original_order").reset_index(drop=True)


def augment_pricing_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add features that can be recomputed for pricing what-if rows."""
    frame = pd.DataFrame(features).copy()

    for column in PRICING_BASE_NUMERIC_FEATURES:
        default = frame.get("avg_unit_price", 0) if column == "price_lag_1" else 0
        frame[column] = _numeric(frame, column, default=default)

    safe_price = frame["avg_unit_price"].clip(lower=0)
    frame["log_avg_unit_price"] = np.log1p(safe_price)
    frame["price_ratio_lag_1"] = _safe_ratio(frame["avg_unit_price"], frame["price_lag_1"], default=1.0)
    frame["discount_x_display"] = frame["discount_percentage"] * frame["is_display"]
    frame["discount_x_mailer"] = frame["discount_percentage"] * frame["is_mailer"]
    frame["any_promotion"] = (
        (frame["discount_percentage"] > 0) | (frame["is_display"] > 0) | (frame["is_mailer"] > 0)
    ).astype(int)
    frame["lag_velocity_4"] = _safe_ratio(frame["lag_quantity_1"], frame["rolling_quantity_mean_4"], default=0.0)
    frame["lag_velocity_8"] = _safe_ratio(frame["lag_quantity_1"], frame["rolling_quantity_mean_8"], default=0.0)
    return frame


def prepare_pricing_scenario_features(
    base_features: dict | pd.Series | pd.DataFrame,
    candidate_price: float,
    discount_rate: float,
) -> pd.DataFrame:
    if isinstance(base_features, pd.DataFrame):
        row = base_features.iloc[[0]].copy()
    elif isinstance(base_features, pd.Series):
        row = base_features.to_frame().T
    else:
        row = pd.DataFrame([base_features])

    candidate_price = float(candidate_price)
    discount_rate = float(discount_rate)
    for column in ("avg_unit_price", "median_unit_price"):
        row[column] = candidate_price
    row["discount_percentage"] = discount_rate
    if "price_lag_1" not in row:
        row["price_lag_1"] = candidate_price
    price_lag = pd.to_numeric(row["price_lag_1"], errors="coerce").fillna(candidate_price).replace(0, candidate_price)
    row["price_change"] = candidate_price - price_lag
    return augment_pricing_features(row)
