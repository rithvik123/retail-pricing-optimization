from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, PROCESSED_DIR, ensure_project_dirs


def classify_elasticity(value: float) -> str:
    if value < -1.5:
        return "highly_price_sensitive"
    if value < -0.5:
        return "moderately_price_sensitive"
    return "low_price_sensitive"


def estimate_group_elasticity(group: pd.DataFrame, min_observations: int = 20) -> float | None:
    data = group.copy()
    data = data[(data["avg_unit_price"] > 0) & (data["quantity_sold"] >= 0)]
    if len(data) < min_observations or data["avg_unit_price"].nunique() < 2:
        return None

    x = pd.DataFrame(
        {
            "log_price": np.log1p(data["avg_unit_price"]),
            "discount_percentage": data.get("discount_percentage", 0),
            "is_display": data.get("is_display", 0),
            "is_mailer": data.get("is_mailer", 0),
            "week_no": data.get("week_no", 0),
        }
    ).fillna(0)
    y = np.log1p(data["quantity_sold"])
    model = LinearRegression()
    model.fit(x, y)
    return float(model.coef_[0])


def calculate_elasticity_table(features: pd.DataFrame, min_observations: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for group_type, column in (("product", "product_id"), ("category", "commodity_desc")):
        if column not in features:
            continue
        for key, group in features.groupby(column, dropna=False):
            elasticity = estimate_group_elasticity(group, min_observations=min_observations)
            if elasticity is None:
                continue
            rows.append(
                {
                    "group_type": group_type,
                    "group_key": str(key),
                    "price_elasticity": elasticity,
                    "sensitivity_class": classify_elasticity(elasticity),
                    "observations": int(len(group)),
                    "avg_price": float(group["avg_unit_price"].mean()),
                    "avg_quantity": float(group["quantity_sold"].mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(["group_type", "price_elasticity"])


def plot_category_elasticity(elasticity_table: pd.DataFrame, output_path: Path = FIGURES_DIR / "elasticity_by_category.png") -> None:
    category = elasticity_table[elasticity_table["group_type"] == "category"].copy()
    if category.empty:
        return
    category = category.sort_values("price_elasticity").head(25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    plt.barh(category["group_key"], category["price_elasticity"], color="#2F6F73")
    plt.axvline(-1.5, color="#B94E48", linestyle="--", linewidth=1)
    plt.axvline(-0.5, color="#D99B36", linestyle="--", linewidth=1)
    plt.title("Price Elasticity by Category")
    plt.xlabel("Elasticity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate product/category price elasticity.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--output-path", default=str(PROCESSED_DIR / "price_elasticity_table.parquet"))
    parser.add_argument("--min-observations", type=int, default=20)
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    table = calculate_elasticity_table(features, min_observations=args.min_observations)
    table.to_parquet(args.output_path, index=False)
    plot_category_elasticity(table)
    print(f"Saved elasticity table: {args.output_path} ({len(table):,} rows)")


if __name__ == "__main__":
    main()
