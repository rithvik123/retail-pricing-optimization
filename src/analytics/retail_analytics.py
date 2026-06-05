from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs


def _save_bar(data: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    sns.barplot(x=data.values, y=data.index, color="#2F6F73")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_line(data: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y, color="#2F6F73", linewidth=2)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def lift_percentage(frame: pd.DataFrame, flag_column: str, value_column: str = "quantity_sold") -> float:
    if flag_column not in frame or frame[flag_column].nunique(dropna=True) < 2:
        return 0.0
    treated = frame.loc[frame[flag_column] == 1, value_column].mean()
    control = frame.loc[frame[flag_column] == 0, value_column].mean()
    if pd.isna(treated) or pd.isna(control) or control == 0:
        return 0.0
    return float((treated - control) / control)


def category_price_sensitivity(features: pd.DataFrame, min_rows: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category, group in features.groupby("commodity_desc", dropna=False):
        if len(group) < min_rows or group["avg_unit_price"].nunique() < 2:
            continue
        corr = group[["avg_unit_price", "quantity_sold"]].corr().iloc[0, 1]
        rows.append({"commodity_desc": category, "price_quantity_corr": corr, "rows": len(group)})
    return pd.DataFrame(rows).sort_values("price_quantity_corr")


def create_business_summary(features: pd.DataFrame, output_path: Path = REPORTS_DIR / "business_summary.md") -> dict[str, object]:
    total_revenue = float(features["sales_value"].sum())
    total_units = float(features["quantity_sold"].sum())
    avg_discount = float(features["discount_percentage"].fillna(0).mean())
    coupon_share = float(features.get("coupon_sales_share", pd.Series([0])).fillna(0).mean())
    display_lift = lift_percentage(features, "is_display")
    mailer_lift = lift_percentage(features, "is_mailer")

    top_category = "Unknown"
    if "commodity_desc" in features:
        top_category = str(features.groupby("commodity_desc")["sales_value"].sum().idxmax())

    summary = {
        "total_revenue": total_revenue,
        "total_units_sold": total_units,
        "average_discount": avg_discount,
        "coupon_sales_share": coupon_share,
        "display_lift": display_lift,
        "mailer_lift": mailer_lift,
        "top_revenue_category": top_category,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "# Retail Pricing Optimization Business Summary",
                "",
                "This summary is generated from the product-store-week feature table.",
                "",
                f"- Total revenue: ${total_revenue:,.2f}",
                f"- Total units sold: {total_units:,.0f}",
                f"- Average discount: {avg_discount:.2%}",
                f"- Coupon sales share: {coupon_share:.2%}",
                f"- Display promotion lift: {display_lift:.2%}",
                f"- Mailer promotion lift: {mailer_lift:.2%}",
                f"- Top revenue category: {top_category}",
                "",
                "Business interpretation should be reviewed with merchandising context before deployment.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def run_retail_analytics(
    features: pd.DataFrame,
    transactions: pd.DataFrame | None = None,
    figures_dir: str | Path = FIGURES_DIR,
    reports_dir: str | Path = REPORTS_DIR,
) -> dict[str, object]:
    figures_path = Path(figures_dir)
    reports_path = Path(reports_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    weekly = features.groupby("week_no", as_index=False).agg(
        sales_value=("sales_value", "sum"),
        quantity_sold=("quantity_sold", "sum"),
    )
    _save_line(weekly, "week_no", "sales_value", "Weekly Revenue Trend", figures_path / "weekly_revenue.png")
    _save_line(weekly, "week_no", "quantity_sold", "Weekly Quantity Trend", figures_path / "weekly_quantity.png")

    if "department" in features:
        department_sales = features.groupby("department")["sales_value"].sum().sort_values(ascending=False).head(15)
        _save_bar(department_sales, "Top Departments by Revenue", "Revenue", "Department", figures_path / "top_departments_revenue.png")

    if "commodity_desc" in features:
        category_units = features.groupby("commodity_desc")["quantity_sold"].sum().sort_values(ascending=False).head(15)
        _save_bar(category_units, "Top Categories by Quantity", "Units", "Category", figures_path / "top_categories_quantity.png")

    product_sales = (
        features.groupby("product_id")["sales_value"].sum().sort_values(ascending=False).head(15)
    )
    _save_bar(product_sales.astype(float), "Top Products by Revenue", "Revenue", "Product ID", figures_path / "top_products_revenue.png")

    store_sales = features.groupby("store_id")["sales_value"].sum().sort_values(ascending=False).head(15)
    _save_bar(store_sales.astype(float), "Top Stores by Revenue", "Revenue", "Store ID", figures_path / "top_stores_revenue.png")

    if "brand" in features:
        brand_summary = features.groupby("brand")["sales_value"].sum().sort_values(ascending=False).head(10)
        _save_bar(brand_summary, "Private Label vs National Brand Revenue", "Revenue", "Brand", figures_path / "brand_revenue.png")

    promo_lifts = pd.Series(
        {
            "Display": lift_percentage(features, "is_display"),
            "Mailer": lift_percentage(features, "is_mailer"),
            "Coupon": lift_percentage(features.assign(has_coupon=(features["total_coupon_discount"] > 0).astype(int)), "has_coupon"),
        }
    )
    _save_bar(promo_lifts, "Promotion Lift by Mechanic", "Lift %", "Promotion", figures_path / "promotion_lifts.png")

    if transactions is not None and "household_key" in transactions:
        spending = transactions.groupby("household_key")["sales_value"].sum()
        plt.figure(figsize=(10, 6))
        sns.histplot(spending, bins=50, color="#2F6F73")
        plt.title("Household Spending Distribution")
        plt.xlabel("Spend")
        plt.tight_layout()
        plt.savefig(figures_path / "household_spending_distribution.png", dpi=160)
        plt.close()

    sensitivity = category_price_sensitivity(features)
    if not sensitivity.empty:
        _save_bar(
            sensitivity.set_index("commodity_desc")["price_quantity_corr"].head(20),
            "Category Price Sensitivity Overview",
            "Price-Quantity Correlation",
            "Category",
            figures_path / "category_price_sensitivity.png",
        )

    return create_business_summary(features, reports_path / "business_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate retail analytics charts and business summary.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--transactions-path", default=str(PROCESSED_DIR / "transactions_clean.parquet"))
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    transactions = pd.read_parquet(args.transactions_path) if Path(args.transactions_path).exists() else None
    summary = run_retail_analytics(features, transactions=transactions)
    print(summary)


if __name__ == "__main__":
    main()
