from __future__ import annotations

from pathlib import Path

from src.config.paths import PROCESSED_DIR


RAW_FEATURES_PATH = PROCESSED_DIR / "product_store_week_features.parquet"
MODELING_READY_FEATURES_PATH = PROCESSED_DIR / "retail_modeling_features.parquet"


def default_features_path() -> Path:
    """Prefer the curated retail modeling table, falling back to raw features."""
    return MODELING_READY_FEATURES_PATH if MODELING_READY_FEATURES_PATH.exists() else RAW_FEATURES_PATH

