from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import INTERIM_DIR, ensure_project_dirs, resolve_raw_dir


TABLE_FILES = {
    "transactions": "transaction_data.csv",
    "products": "product.csv",
    "causal_data": "causal_data.csv",
    "coupons": "coupon.csv",
    "coupon_redemptions": "coupon_redempt.csv",
    "campaign_table": "campaign_table.csv",
    "campaign_desc": "campaign_desc.csv",
    "hh_demographic": "hh_demographic.csv",
}


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = (
        frame.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return frame


def profile_dataframe(frame: pd.DataFrame, duplicate_limit: int = 5_000_000) -> dict[str, Any]:
    duplicate_count: int | str
    if len(frame) <= duplicate_limit:
        duplicate_count = int(frame.duplicated().sum())
    else:
        duplicate_count = "skipped_large_table"

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "duplicate_rows": duplicate_count,
    }


def load_csv_table(path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    return standardize_columns(frame)


def load_all_tables(
    raw_dir: str | Path | None = None,
    sample_rows: int | None = None,
    include_causal: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_path = resolve_raw_dir(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    profiles: dict[str, Any] = {"raw_dir": str(raw_path), "tables": {}}

    for table_name, filename in TABLE_FILES.items():
        if table_name == "causal_data" and not include_causal:
            continue
        frame = load_csv_table(raw_path / filename, sample_rows=sample_rows)
        tables[table_name] = frame
        profiles["tables"][table_name] = profile_dataframe(frame)

    return tables, profiles


def save_interim_tables(tables: dict[str, pd.DataFrame], interim_dir: str | Path = INTERIM_DIR) -> None:
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)
    for table_name, frame in tables.items():
        frame.to_parquet(interim_path / f"{table_name}.parquet", index=False)


def load_profile_and_save_tables(
    raw_dir: str | Path | None = None,
    sample_rows: int | None = None,
    include_causal: bool = True,
    interim_dir: str | Path = INTERIM_DIR,
) -> dict[str, Any]:
    """Load each CSV and save it immediately to reduce peak memory use."""
    raw_path = resolve_raw_dir(raw_dir)
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, Any] = {"raw_dir": str(raw_path), "tables": {}}

    for table_name, filename in TABLE_FILES.items():
        if table_name == "causal_data" and not include_causal:
            continue
        frame = load_csv_table(raw_path / filename, sample_rows=sample_rows)
        profiles["tables"][table_name] = profile_dataframe(frame)
        frame.to_parquet(interim_path / f"{table_name}.parquet", index=False)
        del frame

    return profiles


def print_profiles(profiles: dict[str, Any]) -> None:
    print(f"Raw data directory: {profiles['raw_dir']}")
    for table_name, profile in profiles["tables"].items():
        print(f"\n{table_name}")
        print(f"  shape: ({profile['rows']:,}, {profile['columns']:,})")
        print(f"  duplicate rows: {profile['duplicate_rows']}")
        missing = {key: value for key, value in profile["missing_values"].items() if value}
        print(f"  missing values: {missing if missing else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Dunnhumby CSV files into interim parquet tables.")
    parser.add_argument("--raw-dir", default=None, help="Directory containing Dunnhumby CSV files.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Read only the first N rows per CSV.")
    parser.add_argument("--skip-causal", action="store_true", help="Skip the large causal_data.csv file.")
    args = parser.parse_args()

    ensure_project_dirs()
    profiles = load_profile_and_save_tables(
        raw_dir=args.raw_dir,
        sample_rows=args.sample_rows,
        include_causal=not args.skip_causal,
    )
    print_profiles(profiles)
    print(f"\nSaved {len(profiles['tables'])} parquet tables to {INTERIM_DIR}")


if __name__ == "__main__":
    main()
