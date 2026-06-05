from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config.paths import PROCESSED_DIR, ensure_project_dirs
from src.data.load_data import standardize_columns


PRODUCT_COLUMNS = [
    "product_id",
    "department",
    "commodity_desc",
    "sub_commodity_desc",
    "brand",
    "curr_size_of_product",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def build_transaction_price_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create guide-based transaction-level price and discount features."""
    frame = standardize_columns(transactions)
    frame = frame.copy()

    for column in ("sales_value", "retail_disc", "coupon_disc", "coupon_match_disc", "quantity"):
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    quantity = frame["quantity"]
    sales = frame["sales_value"]
    retail_disc = frame["retail_disc"]
    coupon_disc = frame["coupon_disc"]
    coupon_match_disc = frame["coupon_match_disc"]

    frame["effective_unit_price"] = _safe_divide(sales, quantity)
    frame["loyalty_card_price"] = _safe_divide(sales - (retail_disc + coupon_match_disc), quantity)
    frame["non_loyalty_card_price"] = _safe_divide(sales - coupon_match_disc, quantity)
    frame["shelf_price_estimate"] = _safe_divide(sales - retail_disc - coupon_match_disc, quantity)

    frame["retail_discount_amount"] = retail_disc.abs()
    frame["coupon_discount_amount"] = coupon_disc.abs() + coupon_match_disc.abs()
    frame["total_discount_amount"] = (
        retail_disc.abs() + coupon_disc.abs() + coupon_match_disc.abs()
    )
    frame["estimated_shelf_value"] = sales - retail_disc - coupon_match_disc
    frame["discount_percentage"] = _safe_divide(
        frame["total_discount_amount"], frame["estimated_shelf_value"]
    ).clip(lower=0, upper=1.5)
    frame["has_retail_discount"] = (frame["retail_discount_amount"] > 0).astype(int)
    frame["has_coupon_discount"] = (frame["coupon_discount_amount"] > 0).astype(int)

    return frame


def build_time_features(transactions: pd.DataFrame, campaign_desc: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = transactions.copy()
    frame["day"] = pd.to_numeric(frame.get("day"), errors="coerce")
    frame["week_no"] = pd.to_numeric(frame.get("week_no"), errors="coerce")
    frame["day_of_week_proxy"] = ((frame["day"] - 1) % 7 + 1).astype("Int64")
    frame["month_proxy"] = ((frame["week_no"] - 1) // 4 + 1).astype("Int64")
    frame["quarter_proxy"] = ((frame["week_no"] - 1) // 13 + 1).astype("Int64")
    frame["campaign_active"] = 0

    if campaign_desc is not None and {"start_day", "end_day"}.issubset(campaign_desc.columns):
        intervals = campaign_desc[["start_day", "end_day"]].dropna().drop_duplicates()
        active = pd.Series(False, index=frame.index)
        for row in intervals.itertuples(index=False):
            active = active | frame["day"].between(row.start_day, row.end_day)
        frame["campaign_active"] = active.astype(int)

    return frame


def normalize_promotion_flag(series: pd.Series) -> pd.Series:
    normalized = series.fillna("0").astype(str).str.strip().str.upper()
    inactive_values = {"", "0", "N", "NO", "NONE", "NAN", "NA"}
    return (~normalized.isin(inactive_values)).astype(int)


def aggregate_causal_promotions(causal_data: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["week_no", "store_id", "product_id"]
    causal = standardize_columns(causal_data)
    for column in group_cols:
        causal[column] = pd.to_numeric(causal[column], errors="coerce").astype("Int64")
    causal = causal.dropna(subset=group_cols)
    causal["is_display"] = normalize_promotion_flag(causal.get("display", pd.Series(index=causal.index)))
    causal["is_mailer"] = normalize_promotion_flag(causal.get("mailer", pd.Series(index=causal.index)))
    return (
        causal.groupby(group_cols, observed=True)
        .agg(
            display=("display", "first"),
            mailer=("mailer", "first"),
            is_display=("is_display", "max"),
            is_mailer=("is_mailer", "max"),
        )
        .reset_index()
    )


def aggregate_causal_promotions_from_parquet(causal_path: str | Path, batch_size: int = 2_000_000) -> pd.DataFrame:
    """Aggregate the large causal table in parquet batches."""
    path = Path(causal_path)
    if not path.exists():
        return pd.DataFrame(columns=["week_no", "store_id", "product_id", "display", "mailer", "is_display", "is_mailer"])

    chunks: list[pd.DataFrame] = []
    parquet_file = pq.ParquetFile(path)
    columns = [column for column in ["week_no", "store_id", "product_id", "display", "mailer"] if column in parquet_file.schema.names]
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        chunk = batch.to_pandas()
        chunks.append(aggregate_causal_promotions(chunk))
        print(f"Aggregated causal batch ({len(chunk):,} rows)")
        del chunk

    if not chunks:
        return pd.DataFrame(columns=["week_no", "store_id", "product_id", "display", "mailer", "is_display", "is_mailer"])

    combined = pd.concat(chunks, ignore_index=True)
    result = (
        combined.groupby(["week_no", "store_id", "product_id"], observed=True)
        .agg(
            display=("display", "first"),
            mailer=("mailer", "first"),
            is_display=("is_display", "max"),
            is_mailer=("is_mailer", "max"),
        )
        .reset_index()
    )
    return result


def merge_causal_promotions(features: pd.DataFrame, promo: pd.DataFrame | None) -> pd.DataFrame:
    frame = features.copy()
    if promo is not None and not promo.empty:
        frame = frame.drop(columns=[column for column in ("display", "mailer", "is_display", "is_mailer") if column in frame])
        frame = frame.merge(promo, on=["week_no", "store_id", "product_id"], how="left")
    for column in ("display", "mailer"):
        if column not in frame:
            frame[column] = "0"
        frame[column] = frame[column].fillna("0")
    for column in ("is_display", "is_mailer"):
        if column not in frame:
            frame[column] = 0
        frame[column] = frame[column].fillna(0).astype(int)
    return frame


def aggregate_product_store_week(
    transactions: pd.DataFrame,
    products: pd.DataFrame | None = None,
    causal_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = transactions.copy()
    group_cols = ["week_no", "store_id", "product_id"]

    aggregated = (
        frame.groupby(group_cols, observed=True)
        .agg(
            quantity_sold=("quantity", "sum"),
            sales_value=("sales_value", "sum"),
            avg_unit_price=("effective_unit_price", "mean"),
            median_unit_price=("effective_unit_price", "median"),
            total_retail_discount=("retail_discount_amount", "sum"),
            total_coupon_discount=("coupon_discount_amount", "sum"),
            total_discount_amount=("total_discount_amount", "sum"),
            discount_percentage=("discount_percentage", "mean"),
            num_baskets=("basket_id", "nunique"),
            num_households=("household_key", "nunique"),
            coupon_sales_share=("has_coupon_discount", "mean"),
            campaign_active=("campaign_active", "max"),
        )
        .reset_index()
    )

    if products is not None and not products.empty:
        product_frame = standardize_columns(products)
        existing_cols = [column for column in PRODUCT_COLUMNS if column in product_frame.columns]
        aggregated = aggregated.merge(
            product_frame[existing_cols].drop_duplicates("product_id"),
            on="product_id",
            how="left",
        )

    if causal_data is not None and not causal_data.empty:
        promo = aggregate_causal_promotions(causal_data)
        aggregated = aggregated.merge(promo, on=group_cols, how="left")

    for column in ("department", "commodity_desc", "sub_commodity_desc", "brand", "curr_size_of_product"):
        if column in aggregated:
            aggregated[column] = aggregated[column].fillna("Unknown")
    for column in ("display", "mailer"):
        if column not in aggregated:
            aggregated[column] = "0"
        aggregated[column] = aggregated[column].fillna("0")
    for column in ("is_display", "is_mailer"):
        if column not in aggregated:
            aggregated[column] = 0
        aggregated[column] = aggregated[column].fillna(0).astype(int)

    return add_lag_features(aggregated)


def add_lag_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.sort_values(["product_id", "store_id", "week_no"]).reset_index(drop=True).copy()
    group = frame.groupby(["product_id", "store_id"], observed=True, sort=False)

    for lag in (1, 2, 4):
        frame[f"lag_quantity_{lag}"] = group["quantity_sold"].shift(lag)

    shifted_quantity = group["quantity_sold"].shift(1)
    frame["rolling_quantity_mean_4"] = shifted_quantity.groupby(
        [frame["product_id"], frame["store_id"]], observed=True
    ).transform(lambda values: values.rolling(4, min_periods=1).mean())
    frame["rolling_quantity_mean_8"] = shifted_quantity.groupby(
        [frame["product_id"], frame["store_id"]], observed=True
    ).transform(lambda values: values.rolling(8, min_periods=1).mean())
    frame["rolling_quantity_std_4"] = shifted_quantity.groupby(
        [frame["product_id"], frame["store_id"]], observed=True
    ).transform(lambda values: values.rolling(4, min_periods=2).std())

    frame["price_lag_1"] = group["avg_unit_price"].shift(1)
    frame["price_change"] = frame["avg_unit_price"] - frame["price_lag_1"]

    fill_zero_columns = [
        "lag_quantity_1",
        "lag_quantity_2",
        "lag_quantity_4",
        "rolling_quantity_mean_4",
        "rolling_quantity_mean_8",
        "rolling_quantity_std_4",
        "price_change",
    ]
    for column in fill_zero_columns:
        frame[column] = frame[column].fillna(0)
    frame["price_lag_1"] = frame["price_lag_1"].fillna(frame["avg_unit_price"])
    return frame


def build_modeling_table(
    transactions: pd.DataFrame,
    products: pd.DataFrame | None = None,
    causal_data: pd.DataFrame | None = None,
    campaign_desc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    transactions = build_transaction_price_features(transactions)
    transactions = build_time_features(transactions, campaign_desc=campaign_desc)
    return aggregate_product_store_week(transactions, products=products, causal_data=causal_data)


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build product-store-week modeling features.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    ensure_project_dirs()
    processed_dir = Path(args.processed_dir)
    transactions = pd.read_parquet(processed_dir / "transactions_clean.parquet")
    products = _read_optional(processed_dir / "products_clean.parquet")
    campaign_desc = _read_optional(processed_dir / "campaign_desc_clean.parquet")
    transactions = build_transaction_price_features(transactions)
    transactions = build_time_features(transactions, campaign_desc=campaign_desc)
    features = aggregate_product_store_week(transactions, products=products, causal_data=None)
    causal_path = processed_dir / "causal_data_clean.parquet"
    if causal_path.exists():
        promo = aggregate_causal_promotions_from_parquet(causal_path)
        features = merge_causal_promotions(features, promo)
    output_path = processed_dir / "product_store_week_features.parquet"
    features.to_parquet(output_path, index=False)
    print(f"Saved modeling table: {output_path} ({len(features):,} rows)")


if __name__ == "__main__":
    main()
