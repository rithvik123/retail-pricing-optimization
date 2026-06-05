from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config.paths import PROCESSED_DIR, ensure_project_dirs


NON_MERCHANDISE_DEPARTMENTS = {
    "KIOSK-GAS",
    "MISC SALES TRAN",
    "COUP/STR & MFG",
    "CNTRL/STORE SUP",
    "CHARITABLE CONT",
}

NON_MERCHANDISE_COMMODITIES = {
    "COUPON/MISC ITEMS",
    "NO COMMODITY DESCRIPTION",
}


def build_modeling_ready_features(
    features: pd.DataFrame,
    min_unit_price: float = 0.05,
    max_unit_price: float = 100.0,
    max_discount_percentage: float = 1.0,
    min_quantity_cap: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove non-merchandise and anomalous rows for demand modeling.

    The raw Dunnhumby transaction table contains fuel/coupon/miscellaneous
    lines where `quantity` is not a normal retail unit count. These rows are
    useful audit context, but they distort product demand modeling.
    """
    frame = features.copy()
    reasons = pd.Series("", index=frame.index, dtype="object")

    department = frame.get("department", pd.Series("", index=frame.index)).fillna("").astype(str)
    commodity = frame.get("commodity_desc", pd.Series("", index=frame.index)).fillna("").astype(str)
    avg_price = pd.to_numeric(frame.get("avg_unit_price", 0), errors="coerce")
    discount = pd.to_numeric(frame.get("discount_percentage", 0), errors="coerce").fillna(0)
    quantity = pd.to_numeric(frame.get("quantity_sold", 0), errors="coerce")

    quantity_cap = quantity[
        (~department.isin(NON_MERCHANDISE_DEPARTMENTS))
        & (~commodity.isin(NON_MERCHANDISE_COMMODITIES))
        & (avg_price >= min_unit_price)
        & (avg_price <= max_unit_price)
    ].quantile(0.999)
    if pd.isna(quantity_cap) or quantity_cap < 1:
        quantity_cap = quantity.quantile(0.999)
    quantity_cap = max(float(quantity_cap), float(min_quantity_cap))

    rule_map = {
        "non_merchandise_department": department.isin(NON_MERCHANDISE_DEPARTMENTS),
        "non_merchandise_commodity": commodity.isin(NON_MERCHANDISE_COMMODITIES),
        "unit_price_below_minimum": avg_price < min_unit_price,
        "unit_price_above_maximum": avg_price > max_unit_price,
        "discount_above_100_percent": discount > max_discount_percentage,
        "quantity_above_99_9_percentile": quantity > quantity_cap,
    }

    keep = pd.Series(True, index=frame.index)
    for reason, mask in rule_map.items():
        mask = mask.fillna(False)
        keep &= ~mask
        reasons.loc[mask] = reasons.loc[mask].where(reasons.loc[mask] == "", reasons.loc[mask] + ";") + reason

    filtered = frame.loc[keep].reset_index(drop=True)
    exclusions = (
        pd.DataFrame({"exclusion_reason": reasons.loc[~keep]})
        .assign(exclusion_reason=lambda data: data["exclusion_reason"].str.split(";"))
        .explode("exclusion_reason")
        .groupby("exclusion_reason", as_index=False)
        .size()
        .rename(columns={"size": "excluded_rows"})
        .sort_values("excluded_rows", ascending=False)
    )

    summary_rows = [
        {"metric": "input_rows", "value": len(frame)},
        {"metric": "output_rows", "value": len(filtered)},
        {"metric": "excluded_rows", "value": len(frame) - len(filtered)},
        {"metric": "quantity_cap", "value": float(quantity_cap)},
        {"metric": "input_units", "value": float(quantity.sum())},
        {"metric": "output_units", "value": float(filtered["quantity_sold"].sum())},
        {"metric": "input_revenue", "value": float(frame["sales_value"].sum())},
        {"metric": "output_revenue", "value": float(filtered["sales_value"].sum())},
    ]
    summary = pd.concat([pd.DataFrame(summary_rows), exclusions.rename(columns={"exclusion_reason": "metric", "excluded_rows": "value"})], ignore_index=True)
    return filtered, summary


def write_modeling_ready_report(summary: pd.DataFrame, output_path: Path) -> None:
    values = dict(zip(summary["metric"], summary["value"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Modeling-Ready Dataset Report",
                "",
                "The raw product-store-week feature table is preserved, but demand modeling uses a retail-focused table that removes non-merchandise rows and extreme anomalies.",
                "",
                f"- Input rows: {int(values.get('input_rows', 0)):,}",
                f"- Output rows: {int(values.get('output_rows', 0)):,}",
                f"- Excluded rows: {int(values.get('excluded_rows', 0)):,}",
                f"- Quantity cap: {values.get('quantity_cap', 0):,.0f}",
                f"- Input units: {values.get('input_units', 0):,.0f}",
                f"- Output units: {values.get('output_units', 0):,.0f}",
                f"- Input revenue: ${values.get('input_revenue', 0):,.2f}",
                f"- Output revenue: ${values.get('output_revenue', 0):,.2f}",
                "",
                "## Exclusion Rules",
                "",
                "| Rule | Rows |",
                "| --- | ---: |",
                *[
                    f"| {row.metric} | {int(row.value):,} |"
                    for row in summary.itertuples(index=False)
                    if row.metric not in {
                        "input_rows",
                        "output_rows",
                        "excluded_rows",
                        "quantity_cap",
                        "input_units",
                        "output_units",
                        "input_revenue",
                        "output_revenue",
                    }
                ],
                "",
                "Primary rationale: fuel and coupon/miscellaneous rows encode quantity in non-standard units, producing huge demand values at near-zero prices. They should not train product pricing recommendations.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a retail modeling-ready feature table.")
    parser.add_argument("--features-path", default=str(PROCESSED_DIR / "product_store_week_features.parquet"))
    parser.add_argument("--output-path", default=str(PROCESSED_DIR / "retail_modeling_features.parquet"))
    parser.add_argument("--summary-path", default=str(PROCESSED_DIR / "retail_modeling_filter_summary.csv"))
    parser.add_argument("--report-path", default="reports/modeling_ready_dataset.md")
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    filtered, summary = build_modeling_ready_features(features)
    filtered.to_parquet(args.output_path, index=False)
    summary.to_csv(args.summary_path, index=False)
    write_modeling_ready_report(summary, Path(args.report_path))
    print(f"Saved modeling-ready features: {args.output_path} ({len(filtered):,} rows)")


if __name__ == "__main__":
    main()
