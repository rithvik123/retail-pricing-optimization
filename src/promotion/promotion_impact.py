from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, PROCESSED_DIR, ensure_project_dirs


PROMOTION_FLAGS = {
    "display": "is_display",
    "mailer": "is_mailer",
    "coupon": "has_coupon",
    "retail_discount": "has_retail_discount",
    "campaign": "campaign_active",
}


def _lift(active: float, inactive: float) -> float:
    if pd.isna(active) or pd.isna(inactive) or inactive == 0:
        return 0.0
    return float((active - inactive) / inactive)


def add_promotion_flags(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["has_coupon"] = (frame.get("total_coupon_discount", 0) > 0).astype(int)
    frame["has_retail_discount"] = (frame.get("total_retail_discount", 0) > 0).astype(int)
    if "campaign_active" not in frame:
        frame["campaign_active"] = 0
    return frame


def calculate_promotion_impact(features: pd.DataFrame) -> pd.DataFrame:
    frame = add_promotion_flags(features)
    group_columns = ["commodity_desc"] if "commodity_desc" in frame else []
    rows: list[dict[str, object]] = []

    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [(None, frame)]
    for group_key, group in grouped:
        if group_columns:
            category = group_key[0] if isinstance(group_key, tuple) else group_key
        else:
            category = "all"
        for mechanism, flag in PROMOTION_FLAGS.items():
            if flag not in group:
                continue
            active = group.loc[group[flag] == 1, "quantity_sold"].mean()
            inactive = group.loc[group[flag] == 0, "quantity_sold"].mean()
            rows.append(
                {
                    "commodity_desc": category,
                    "mechanism": mechanism,
                    "average_quantity_when_active": float(active) if pd.notna(active) else 0.0,
                    "average_quantity_when_inactive": float(inactive) if pd.notna(inactive) else 0.0,
                    "lift_percentage": _lift(active, inactive),
                    "active_rows": int((group[flag] == 1).sum()),
                    "inactive_rows": int((group[flag] == 0).sum()),
                }
            )

    overall_rows = []
    for mechanism, flag in PROMOTION_FLAGS.items():
        if flag not in frame:
            continue
        active = frame.loc[frame[flag] == 1, "quantity_sold"].mean()
        inactive = frame.loc[frame[flag] == 0, "quantity_sold"].mean()
        overall_rows.append(
            {
                "commodity_desc": "all",
                "mechanism": mechanism,
                "average_quantity_when_active": float(active) if pd.notna(active) else 0.0,
                "average_quantity_when_inactive": float(inactive) if pd.notna(inactive) else 0.0,
                "lift_percentage": _lift(active, inactive),
                "active_rows": int((frame[flag] == 1).sum()),
                "inactive_rows": int((frame[flag] == 0).sum()),
            }
        )

    return pd.DataFrame(overall_rows + rows)


def plot_promotion_lift(
    impact_table: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "promotion_lift_by_category.png",
    mechanism: str = "display",
) -> None:
    data = impact_table[
        (impact_table["mechanism"] == mechanism)
        & (impact_table["commodity_desc"] != "all")
        & (impact_table["active_rows"] >= 5)
        & (impact_table["inactive_rows"] >= 5)
    ].copy()
    if data.empty:
        return
    data = data.sort_values("lift_percentage", ascending=False).head(20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    plt.barh(data["commodity_desc"], data["lift_percentage"], color="#2F6F73")
    plt.title(f"{mechanism.title()} Lift by Category")
    plt.xlabel("Lift %")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze promotion and coupon impact.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--output-path", default=str(PROCESSED_DIR / "promotion_impact_table.parquet"))
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    impact = calculate_promotion_impact(features)
    impact.to_parquet(args.output_path, index=False)
    plot_promotion_lift(impact)
    print(f"Saved promotion impact table: {args.output_path} ({len(impact):,} rows)")


if __name__ == "__main__":
    main()
