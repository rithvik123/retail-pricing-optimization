from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR, ensure_project_dirs
from src.data.load_data import standardize_columns
from src.features.build_features import build_transaction_price_features
from src.utils.validation import validate_transactions, validation_issues_to_frame


ID_COLUMNS = ("product_id", "household_key", "store_id", "basket_id")
PRODUCT_TEXT_COLUMNS = ("department", "commodity_desc", "sub_commodity_desc", "brand")


def cast_identifier_columns(frame: pd.DataFrame, columns: tuple[str, ...] = ID_COLUMNS) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    products = standardize_columns(products)
    products = cast_identifier_columns(products, ("product_id",))
    for column in PRODUCT_TEXT_COLUMNS:
        if column in products:
            products[column] = products[column].fillna("Unknown").astype(str).str.strip()
            products[column] = products[column].replace({"": "Unknown"})
    if "curr_size_of_product" in products:
        products["curr_size_of_product"] = (
            products["curr_size_of_product"].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
        )
    return products.drop_duplicates(subset=["product_id"])


def clean_transactions(
    transactions: pd.DataFrame,
    quantity_upper_quantile: float = 0.999,
    price_upper_quantile: float = 0.999,
) -> pd.DataFrame:
    transactions = standardize_columns(transactions)
    transactions = cast_identifier_columns(transactions)

    numeric_columns = (
        "day",
        "quantity",
        "sales_value",
        "retail_disc",
        "coupon_disc",
        "coupon_match_disc",
        "trans_time",
        "week_no",
    )
    for column in numeric_columns:
        if column in transactions:
            transactions[column] = pd.to_numeric(transactions[column], errors="coerce")

    transactions = transactions[transactions["quantity"] > 0].copy()
    transactions = transactions[transactions["sales_value"] >= 0].copy()
    transactions = transactions.dropna(subset=["product_id", "store_id", "week_no"])
    transactions = build_transaction_price_features(transactions)
    transactions = transactions[transactions["effective_unit_price"] > 0].copy()

    if len(transactions):
        quantity_cap = transactions["quantity"].quantile(quantity_upper_quantile)
        price_cap = transactions["effective_unit_price"].replace([np.inf, -np.inf], np.nan).quantile(
            price_upper_quantile
        )
        transactions = transactions[transactions["quantity"] <= quantity_cap]
        if pd.notna(price_cap) and price_cap > 0:
            transactions = transactions[transactions["effective_unit_price"] <= price_cap]

    return transactions.reset_index(drop=True)


def clean_table_collection(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cleaned: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        frame = standardize_columns(frame)
        if name == "transactions":
            cleaned[name] = clean_transactions(frame)
        elif name == "products":
            cleaned[name] = clean_products(frame)
        else:
            cleaned[name] = frame.drop_duplicates().reset_index(drop=True)
    return cleaned


def clean_generic_table(frame: pd.DataFrame, table_name: str, duplicate_limit: int = 5_000_000) -> pd.DataFrame:
    frame = standardize_columns(frame)
    if table_name == "causal_data":
        for column in ("product_id", "store_id", "week_no"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        return frame.reset_index(drop=True)
    if len(frame) <= duplicate_limit:
        frame = frame.drop_duplicates()
    return frame.reset_index(drop=True)


def load_interim_tables(interim_dir: str | Path = INTERIM_DIR) -> dict[str, pd.DataFrame]:
    interim_path = Path(interim_dir)
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(interim_path.glob("*.parquet")):
        tables[path.stem] = pd.read_parquet(path)
    if not tables:
        raise FileNotFoundError(f"No interim parquet files found in {interim_path}")
    return tables


def save_clean_tables(cleaned: dict[str, pd.DataFrame], processed_dir: str | Path = PROCESSED_DIR) -> None:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    for name, frame in cleaned.items():
        frame.to_parquet(processed_path / f"{name}_clean.parquet", index=False)

    if "transactions" in cleaned:
        issues = validate_transactions(cleaned["transactions"])
        validation_issues_to_frame(issues).to_csv(processed_path / "data_validation_issues.csv", index=False)


def clean_interim_tables_sequential(
    interim_dir: str | Path = INTERIM_DIR,
    processed_dir: str | Path = PROCESSED_DIR,
) -> None:
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(interim_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No interim parquet files found in {interim_path}")

    for path in parquet_files:
        table_name = path.stem
        frame = pd.read_parquet(path)
        if table_name == "transactions":
            cleaned = clean_transactions(frame)
        elif table_name == "products":
            cleaned = clean_products(frame)
        else:
            cleaned = clean_generic_table(frame, table_name)
        cleaned.to_parquet(processed_path / f"{table_name}_clean.parquet", index=False)
        if table_name == "transactions":
            issues = validate_transactions(cleaned)
            validation_issues_to_frame(issues).to_csv(
                processed_path / "data_validation_issues.csv",
                index=False,
            )
        print(f"Saved {table_name}_clean.parquet ({len(cleaned):,} rows)")
        del frame, cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean interim Dunnhumby tables.")
    parser.add_argument("--interim-dir", default=str(INTERIM_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    ensure_project_dirs()
    clean_interim_tables_sequential(args.interim_dir, args.processed_dir)
    print(f"Saved cleaned tables to {args.processed_dir}")


if __name__ == "__main__":
    main()
