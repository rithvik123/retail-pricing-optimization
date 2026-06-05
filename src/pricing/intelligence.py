from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import PROCESSED_DIR


DEFAULT_ELASTICITY = -1.0
DEFAULT_PROMOTION_EFFECT = 0.0


@lru_cache(maxsize=1)
def load_features(features_path: str | None = None) -> pd.DataFrame:
    path = Path(features_path) if features_path else default_features_path()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_elasticity_table(path: str | None = None) -> pd.DataFrame:
    table_path = Path(path) if path else PROCESSED_DIR / "price_elasticity_table.parquet"
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(table_path)


@lru_cache(maxsize=1)
def load_promotion_impact_table(path: str | None = None) -> pd.DataFrame:
    table_path = Path(path) if path else PROCESSED_DIR / "promotion_impact_table.parquet"
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(table_path)


def latest_product_store_features(product_id: int | str, store_id: int | str, features: pd.DataFrame | None = None) -> pd.Series | None:
    frame = features if features is not None else load_features()
    if frame.empty or not {"product_id", "store_id"}.issubset(frame.columns):
        return None
    product_value = int(product_id) if str(product_id).isdigit() else product_id
    store_value = int(store_id) if str(store_id).isdigit() else store_id
    rows = frame[(frame["product_id"] == product_value) & (frame["store_id"] == store_value)]
    if rows.empty:
        rows = frame[frame["product_id"] == product_value]
    if rows.empty:
        return None
    return rows.sort_values("week_no").iloc[-1]


def lookup_price_elasticity(
    product_id: int | str | None = None,
    commodity_desc: str | None = None,
    elasticity_table: pd.DataFrame | None = None,
    default: float = DEFAULT_ELASTICITY,
) -> float:
    table = elasticity_table if elasticity_table is not None else load_elasticity_table()
    if table.empty:
        return default

    if product_id is not None:
        product_rows = table[
            (table["group_type"] == "product")
            & (table["group_key"].astype(str) == str(product_id))
        ]
        if not product_rows.empty:
            return float(product_rows.iloc[0]["price_elasticity"])

    if commodity_desc:
        category_rows = table[
            (table["group_type"] == "category")
            & (table["group_key"].astype(str) == str(commodity_desc))
        ]
        if not category_rows.empty:
            return float(category_rows.iloc[0]["price_elasticity"])

    return default


def lookup_promotion_effect(
    commodity_desc: str | None = None,
    mechanism: str = "retail_discount",
    promotion_table: pd.DataFrame | None = None,
    default: float = DEFAULT_PROMOTION_EFFECT,
) -> float:
    table = promotion_table if promotion_table is not None else load_promotion_impact_table()
    if table.empty:
        return default

    if commodity_desc:
        rows = table[
            (table["commodity_desc"].astype(str) == str(commodity_desc))
            & (table["mechanism"] == mechanism)
        ]
        if not rows.empty:
            return float(rows.iloc[0]["lift_percentage"])

    overall = table[(table["commodity_desc"].astype(str) == "all") & (table["mechanism"] == mechanism)]
    if not overall.empty:
        return float(overall.iloc[0]["lift_percentage"])
    return default


def build_recommendation_context(product_id: int | str, store_id: int | str) -> dict[str, Any]:
    row = latest_product_store_features(product_id, store_id)
    if row is None:
        return {
            "base_features": None,
            "base_quantity": 100.0,
            "price_elasticity": DEFAULT_ELASTICITY,
            "promotion_effect": DEFAULT_PROMOTION_EFFECT,
            "commodity_desc": None,
            "source": "fallback_defaults",
        }

    commodity_desc = row.get("commodity_desc")
    elasticity = lookup_price_elasticity(product_id=product_id, commodity_desc=commodity_desc)
    promotion_effect = lookup_promotion_effect(commodity_desc=commodity_desc, mechanism="retail_discount")
    return {
        "base_features": row.to_dict(),
        "base_quantity": float(row.get("quantity_sold", 100.0)),
        "latest_price": float(row.get("avg_unit_price", 0.0)),
        "price_elasticity": elasticity,
        "promotion_effect": promotion_effect,
        "commodity_desc": commodity_desc,
        "source": "artifact_lookup",
    }
