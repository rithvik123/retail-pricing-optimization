from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyarrow.parquet as pq

from src.config.artifacts import default_features_path
from src.config.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.models.serving_features import add_serving_history_features
from src.optimization.profit_optimizer import (
    candidate_discounts_from_price_history,
    format_price_action,
    load_demand_model,
    load_pricing_model,
    recommend_price,
    simulate_price_candidates,
)
from src.pricing.intelligence import lookup_price_elasticity, lookup_promotion_effect
from src.pricing.elasticity import calculate_elasticity_table
from src.promotion.promotion_impact import calculate_promotion_impact


st.set_page_config(
    page_title="Retail Pricing Optimization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #1f2933;
      --muted: #667085;
      --panel: #ffffff;
      --line: #d9dee2;
      --teal: #1f6f6a;
      --berry: #b5525c;
      --amber: #d89c38;
      --green: #4f7d4b;
      --blue: #315f8c;
      --violet: #6b5b95;
      --mist: #f6f8f7;
    }
    .stApp { background: var(--mist); color: var(--ink); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [role="radiogroup"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
      color: var(--ink) !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      min-height: 30px;
      padding: 3px 0;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label {
      color: var(--ink) !important;
      opacity: 1 !important;
    }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
    .metric-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px 13px 16px;
      min-height: 98px;
      box-shadow: 0 8px 22px rgba(31, 41, 51, 0.05);
      position: relative;
      overflow: hidden;
    }
    .metric-card:before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      height: 4px;
      width: 100%;
      background: var(--accent, var(--teal));
    }
    .metric-label { color: var(--muted); font-size: 0.82rem; margin-bottom: 8px; }
    .metric-value {
      color: var(--ink);
      font-size: clamp(1.05rem, 1.15vw, 1.45rem);
      font-weight: 700;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .metric-sub { color: var(--muted); font-size: 0.78rem; margin-top: 6px; }
    .section-band {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      margin: 8px 0 16px 0;
      box-shadow: 0 8px 22px rgba(31, 41, 51, 0.04);
    }
    .section-title {
      color: var(--ink);
      font-size: 1.02rem;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .section-note {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }
    .status-pill {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      margin: 0 6px 6px 0;
      background: #ffffff;
      color: var(--ink);
      font-size: 0.78rem;
      font-weight: 600;
    }
    .recommendation {
      background: #ffffff;
      border-left: 5px solid var(--teal);
      border-radius: 8px;
      border-top: 1px solid var(--line);
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 18px 20px;
      box-shadow: 0 10px 26px rgba(31, 41, 51, 0.06);
    }
    .decision-number {
      color: var(--teal);
      font-weight: 800;
      font-size: 2rem;
      line-height: 1.15;
    }
    .risk-note {
      border-left: 4px solid var(--amber);
      padding: 10px 14px;
      background: #fffaf0;
      border-radius: 6px;
      color: #684b16;
      font-size: 0.88rem;
    }
    .empty-state {
      background: #ffffff;
      border: 1px solid var(--line);
      border-left: 5px solid var(--amber);
      border-radius: 8px;
      padding: 18px 20px;
      margin-top: 14px;
      box-shadow: 0 10px 26px rgba(31, 41, 51, 0.06);
      color: var(--ink);
    }
    .empty-state h3 {
      margin: 0 0 8px 0;
      font-size: 1.05rem;
    }
    .empty-state p {
      margin: 0 0 8px 0;
      color: var(--muted);
    }
    [data-testid="stAlert"],
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div {
      color: var(--ink) !important;
    }
    div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def demo_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    weeks = np.arange(1, 53)
    products = [1004906, 1033142, 1036325, 1082185, 8160430, 26190]
    stores = [286, 327, 364, 406]
    rows = []
    for product in products:
        for store in stores:
            base_price = rng.uniform(1.2, 6.5)
            base_qty = rng.uniform(20, 120)
            department = rng.choice(["GROCERY", "PASTRY", "DRUG GM", "MEAT"])
            category = rng.choice(["COOKIES/CONES", "FRUIT - SHELF STABLE", "BREAD", "YOGURT"])
            brand = rng.choice(["Private", "National"])
            for week in weeks:
                display = int(rng.random() < 0.08)
                mailer = int(rng.random() < 0.12)
                discount = rng.choice([0, 0.05, 0.10, 0.15], p=[0.65, 0.15, 0.13, 0.07])
                price = base_price * (1 - discount)
                qty = base_qty * (1 + 1.8 * discount + 0.35 * display + 0.18 * mailer) + rng.normal(0, 7)
                qty = max(qty, 1)
                rows.append(
                    {
                        "week_no": week,
                        "store_id": store,
                        "product_id": product,
                        "department": department,
                        "commodity_desc": category,
                        "sub_commodity_desc": category,
                        "brand": brand,
                        "quantity_sold": qty,
                        "sales_value": qty * price,
                        "avg_unit_price": price,
                        "median_unit_price": price,
                        "discount_percentage": discount,
                        "total_retail_discount": qty * base_price * discount,
                        "total_coupon_discount": 0,
                        "num_baskets": int(qty / rng.uniform(1.2, 3.2)),
                        "num_households": int(qty / rng.uniform(1.5, 4.0)),
                        "is_display": display,
                        "is_mailer": mailer,
                        "coupon_sales_share": 0,
                        "campaign_active": int(week in range(12, 20)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["product_id", "store_id", "week_no"])
    group = frame.groupby(["product_id", "store_id"], sort=False)
    for lag in (1, 2, 4):
        frame[f"lag_quantity_{lag}"] = group["quantity_sold"].shift(lag).fillna(0)
    frame["rolling_quantity_mean_4"] = group["quantity_sold"].shift(1).groupby(
        [frame["product_id"], frame["store_id"]]
    ).transform(lambda values: values.rolling(4, min_periods=1).mean()).fillna(0)
    frame["rolling_quantity_mean_8"] = group["quantity_sold"].shift(1).groupby(
        [frame["product_id"], frame["store_id"]]
    ).transform(lambda values: values.rolling(8, min_periods=1).mean()).fillna(0)
    frame["rolling_quantity_std_4"] = group["quantity_sold"].shift(1).groupby(
        [frame["product_id"], frame["store_id"]]
    ).transform(lambda values: values.rolling(4, min_periods=2).std()).fillna(0)
    frame["price_lag_1"] = group["avg_unit_price"].shift(1).fillna(frame["avg_unit_price"])
    frame["price_change"] = frame["avg_unit_price"] - frame["price_lag_1"]
    return frame


def pricing_model_is_accepted() -> bool:
    path = REPORTS_DIR / "pricing_model_champion.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return bool(report.get("accepted_for_pricing", report.get("serving_safe") and report.get("near_current_accuracy")))


@st.cache_data(show_spinner=False)
def load_features() -> tuple[pd.DataFrame, bool]:
    path = default_features_path()
    if path.exists():
        features = pd.read_parquet(path)
        return (add_serving_history_features(features) if pricing_model_is_accepted() else features), False
    features = demo_features()
    return (add_serving_history_features(features) if pricing_model_is_accepted() else features), True


@st.cache_resource(show_spinner=False)
def load_dashboard_model():
    return load_demand_model()


@st.cache_resource(show_spinner=False)
def load_dashboard_pricing_model():
    return load_pricing_model()


@st.cache_data(show_spinner=False)
def load_json_artifact(path: str) -> dict:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {}
    return json.loads(artifact_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_table_metadata() -> pd.DataFrame:
    tables = [
        ("Curated Modeling Features", PROCESSED_DIR / "retail_modeling_features.parquet", "Decision table"),
        ("Raw Product-Store-Week Features", PROCESSED_DIR / "product_store_week_features.parquet", "Audit table"),
        ("Clean Transactions", PROCESSED_DIR / "transactions_clean.parquet", "Transaction lines"),
        ("Causal Promotions", PROCESSED_DIR / "causal_data_clean.parquet", "Display and mailer"),
        ("Products", PROCESSED_DIR / "products_clean.parquet", "Product catalog"),
        ("Price Elasticity", PROCESSED_DIR / "price_elasticity_table.parquet", "Pricing response"),
        ("Promotion Impact", PROCESSED_DIR / "promotion_impact_table.parquet", "Promotion lift"),
    ]
    rows = []
    for name, path, role in tables:
        exists = path.exists()
        row_count = None
        column_count = None
        size_mb = None
        if exists:
            try:
                metadata = pq.ParquetFile(path).metadata
                row_count = metadata.num_rows
                column_count = metadata.num_columns
                size_mb = path.stat().st_size / 1024 / 1024
            except Exception:
                size_mb = path.stat().st_size / 1024 / 1024
        rows.append(
            {
                "Dataset": name,
                "Role": role,
                "Rows": row_count,
                "Columns": column_count,
                "Size MB": size_mb,
                "Status": "Available" if exists else "Missing",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_filter_summary() -> pd.DataFrame:
    path = PROCESSED_DIR / "retail_modeling_filter_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["metric", "value"])


@st.cache_data(show_spinner=False)
def load_champion_report() -> dict:
    return load_json_artifact(str(REPORTS_DIR / "model_champion.json"))


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def signed_pct(value: float) -> str:
    return f"{value:+.1%}"


def metric_card(label: str, value: str, sub: str = "", accent: str = "#1f6f6a") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent};">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, note: str | None = None) -> None:
    note_html = f'<div class="section-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-band">
          <div class="section-title">{title}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pills(items: list[tuple[str, str]]) -> None:
    pills = "".join(f'<span class="status-pill">{label}: {value}</span>' for label, value in items)
    st.markdown(pills, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def price_rich_product_options(frame: pd.DataFrame) -> list:
    if frame.empty or "product_id" not in frame:
        return []
    summary = (
        frame.groupby("product_id", observed=True)
        .agg(
            price_points=("avg_unit_price", "nunique"),
            rows=("week_no", "size"),
            revenue=("sales_value", "sum"),
        )
        .reset_index()
        .sort_values(["price_points", "rows", "revenue"], ascending=[False, False, False])
    )
    return summary["product_id"].tolist()


def selected_filter_summary() -> str:
    parts = []
    labels = [
        ("Department", "department_filter"),
        ("Category", "category_filter"),
        ("Brand", "brand_filter"),
    ]
    for label, key in labels:
        values = st.session_state.get(key, [])
        if values:
            parts.append(f"{label}: {', '.join(map(str, values))}")
    return " | ".join(parts) if parts else "No sidebar filters selected"


def empty_filter_state(page: str, full_frame: pd.DataFrame) -> None:
    st.title(page)
    total_rows = len(full_frame)
    st.markdown(
        f"""
        <div class="empty-state">
          <h3>No matching rows for the current filters</h3>
          <p><b>Current selection:</b> {selected_filter_summary()}</p>
          <p>Clear one filter, or choose Category after Department so the category belongs to that department. The full dataset has {total_rows:,} modeling rows available.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.warning("No rows match this filter combination. Clear Category or Brand to continue.")


def valid_options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame or frame.empty:
        return []
    return sorted(frame[column].dropna().astype(str).unique())


def multiselect_filtered(label: str, options: list[str], key: str) -> list[str]:
    existing = [value for value in st.session_state.get(key, []) if value in options]
    if key not in st.session_state or existing != st.session_state.get(key, []):
        st.session_state[key] = existing
    return st.sidebar.multiselect(label, options, key=key)


def filter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Retail Pricing")
    selected_departments = multiselect_filtered("Department", valid_options(frame, "department"), "department_filter")
    filtered = frame.copy()
    if selected_departments:
        filtered = filtered[filtered["department"].astype(str).isin(selected_departments)]

    selected_categories = multiselect_filtered("Category", valid_options(filtered, "commodity_desc"), "category_filter")
    if selected_categories:
        filtered = filtered[filtered["commodity_desc"].astype(str).isin(selected_categories)]

    selected_brands = multiselect_filtered("Brand", valid_options(filtered, "brand"), "brand_filter")
    if selected_brands:
        filtered = filtered[filtered["brand"].astype(str).isin(selected_brands)]

    week_min, week_max = int(frame["week_no"].min()), int(frame["week_no"].max())
    if week_min == week_max:
        st.sidebar.caption(f"Week {week_min}")
        week_range = (week_min, week_max)
    else:
        week_range = st.sidebar.slider("Week range", week_min, week_max, (week_min, week_max))

    filtered = filtered[filtered["week_no"].between(*week_range)].copy()
    return filtered


def executive_overview(frame: pd.DataFrame) -> None:
    st.title("Retail Pricing Optimization Intelligence Platform")
    if frame.empty:
        st.warning("No rows match the selected filters.")
        return
    champion = load_champion_report().get("champion", {})
    if champion:
        status_pills(
            [
                ("Serving model", str(champion.get("name", "unknown")).replace("_", " ").title()),
                ("Test WAPE", f"{champion.get('score', 0):.3f}"),
                ("Rows in view", f"{len(frame):,}"),
            ]
        )

    display_lift = lift(frame, "is_display")
    mailer_lift = lift(frame, "is_mailer")
    top_category = frame.groupby("commodity_desc")["sales_value"].sum().idxmax()
    primary_values = [
        ("Total Revenue", money(frame["sales_value"].sum()), "net sales", "#1f6f6a"),
        ("Units Sold", f"{frame['quantity_sold'].sum():,.0f}", "retail units", "#4f7d4b"),
        ("Avg Discount", pct(frame["discount_percentage"].fillna(0).mean()), "mean line discount", "#d89c38"),
        ("Coupon Share", pct(frame.get("coupon_sales_share", pd.Series([0])).fillna(0).mean()), "basket-line share", "#6b5b95"),
    ]
    secondary_values = [
        ("Display Lift", pct(display_lift), "quantity uplift", "#315f8c"),
        ("Mailer Lift", pct(mailer_lift), "quantity uplift", "#b5525c"),
        ("Top Category", str(top_category)[:28], "by revenue", "#1f6f6a"),
    ]
    for col, item in zip(st.columns(4), primary_values):
        with col:
            metric_card(*item)
    for col, item in zip(st.columns(3), secondary_values):
        with col:
            metric_card(*item)

    section_header("Trading Performance")
    weekly = frame.groupby("week_no", as_index=False).agg(sales_value=("sales_value", "sum"), quantity_sold=("quantity_sold", "sum"))
    left, right = st.columns([1.4, 1])
    with left:
        fig = px.line(weekly, x="week_no", y="sales_value", title="Weekly Revenue Trend")
        fig.add_bar(x=weekly["week_no"], y=weekly["quantity_sold"], name="Units", yaxis="y2", opacity=0.22)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Units"))
        st.plotly_chart(fig, width="stretch")
    with right:
        dept = frame.groupby("department", as_index=False)["sales_value"].sum().sort_values("sales_value", ascending=False).head(12)
        st.plotly_chart(px.bar(dept, x="sales_value", y="department", orientation="h", title="Department Revenue"), width="stretch")

    left, middle, right = st.columns(3)
    with left:
        brand = frame.groupby("brand", as_index=False)["sales_value"].sum()
        st.plotly_chart(px.pie(brand, names="brand", values="sales_value", hole=0.45, title="Private vs National Revenue"), width="stretch")
    with middle:
        promo_mix = pd.DataFrame(
            {
                "mechanic": ["Retail Discount", "Coupon", "Display", "Mailer"],
                "share": [
                    float((frame["total_retail_discount"] > 0).mean()),
                    float((frame["total_coupon_discount"] > 0).mean()),
                    float(frame["is_display"].mean()),
                    float(frame["is_mailer"].mean()),
                ],
            }
        )
        st.plotly_chart(px.bar(promo_mix, x="mechanic", y="share", title="Promotion Coverage"), width="stretch")
    with right:
        price_bands = frame.assign(
            price_band=pd.cut(
                frame["avg_unit_price"],
                bins=[0, 1, 2, 3, 5, 10, 100],
                labels=["<$1", "$1-$2", "$2-$3", "$3-$5", "$5-$10", "$10+"],
            ).astype(str)
        )
        band_summary = price_bands.groupby("price_band", as_index=False, observed=True)["sales_value"].sum()
        st.plotly_chart(px.bar(band_summary, x="price_band", y="sales_value", title="Revenue by Price Band"), width="stretch")


def lift(frame: pd.DataFrame, flag: str) -> float:
    if flag not in frame or frame[flag].nunique() < 2:
        return 0.0
    active = frame.loc[frame[flag] == 1, "quantity_sold"].mean()
    inactive = frame.loc[frame[flag] == 0, "quantity_sold"].mean()
    return 0.0 if inactive == 0 or pd.isna(inactive) else float((active - inactive) / inactive)


@st.cache_data(show_spinner=False)
def load_static_elasticity() -> pd.DataFrame:
    path = PROCESSED_DIR / "price_elasticity_table.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["group_type", "group_key", "price_elasticity", "sensitivity_class"])
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_static_promotion_lift() -> pd.DataFrame:
    path = PROCESSED_DIR / "promotion_impact_table.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["commodity_desc", "mechanism", "lift_percentage"])
    return pd.read_parquet(path)


def latest_product_store_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    keys = ["product_id", "store_id"]
    latest_idx = frame.groupby(keys, sort=False)["week_no"].idxmax()
    latest = frame.loc[latest_idx].copy()
    history = (
        frame.groupby(keys, as_index=False)
        .agg(
            history_rows=("week_no", "size"),
            history_weeks=("week_no", "nunique"),
            total_revenue=("sales_value", "sum"),
            total_units=("quantity_sold", "sum"),
            mean_units=("quantity_sold", "mean"),
            avg_observed_discount=("discount_percentage", "mean"),
            avg_observed_price=("avg_unit_price", "mean"),
        )
    )
    return latest.merge(history, on=keys, how="left")


def enrich_opportunity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    latest = latest_product_store_snapshot(frame)
    if latest.empty:
        return latest

    elasticity = load_static_elasticity()
    category_elasticity = elasticity[elasticity["group_type"] == "category"][["group_key", "price_elasticity", "sensitivity_class"]].copy()
    category_elasticity = category_elasticity.rename(
        columns={
            "group_key": "commodity_desc",
            "price_elasticity": "category_elasticity",
            "sensitivity_class": "sensitivity",
        }
    )
    latest = latest.merge(category_elasticity, on="commodity_desc", how="left")

    promotion = load_static_promotion_lift()
    retail_lift = promotion[promotion["mechanism"] == "retail_discount"][["commodity_desc", "lift_percentage"]].copy()
    retail_lift = retail_lift.rename(columns={"lift_percentage": "retail_discount_lift"})
    latest = latest.merge(retail_lift, on="commodity_desc", how="left")

    latest["category_elasticity"] = latest["category_elasticity"].fillna(-1.0).clip(-3, 1)
    latest["retail_discount_lift"] = latest["retail_discount_lift"].fillna(0).clip(-1, 3)
    latest["abs_elasticity"] = latest["category_elasticity"].abs()
    latest["markdown_exposure"] = latest["avg_observed_discount"].fillna(0).clip(0, 1)
    latest["revenue_rank_score"] = latest["total_revenue"].rank(pct=True).fillna(0)
    latest["margin_recovery_score"] = latest["total_revenue"].fillna(0) * latest["markdown_exposure"] * (1 - latest["abs_elasticity"].clip(0, 1))
    latest["markdown_test_score"] = latest["total_revenue"].fillna(0) * latest["abs_elasticity"].clip(0, 2) * (1 + latest["retail_discount_lift"].clip(0, 2))

    action_conditions = [
        (latest["markdown_exposure"] >= 0.08) & (latest["abs_elasticity"] <= 0.35),
        (latest["category_elasticity"] <= -0.75) & (latest["retail_discount_lift"] >= 0.10),
        (latest["quantity_sold"] < latest["mean_units"] * 0.65) & (latest["retail_discount_lift"] >= 0.10),
    ]
    action_choices = [
        "Recover margin",
        "Test targeted markdown",
        "Demand recovery promo",
    ]
    latest["recommended_action"] = np.select(action_conditions, action_choices, default="Protect current price")
    latest["priority_score"] = np.select(
        [
            latest["recommended_action"].eq("Recover margin"),
            latest["recommended_action"].eq("Test targeted markdown"),
            latest["recommended_action"].eq("Demand recovery promo"),
        ],
        [
            latest["margin_recovery_score"],
            latest["markdown_test_score"],
            latest["total_revenue"].fillna(0) * latest["retail_discount_lift"].clip(0, 2),
        ],
        default=latest["total_revenue"].fillna(0) * (1 - latest["abs_elasticity"].clip(0, 1)) * 0.05,
    )
    latest["decision_cue"] = latest["recommended_action"].map(
        {
            "Recover margin": "Discounting exists, but category price sensitivity is low.",
            "Test targeted markdown": "Category is more price responsive and promo lift is positive.",
            "Demand recovery promo": "Latest units trail history; promo signal may help recovery.",
            "Protect current price": "Weak evidence that a markdown will beat current-price profit.",
        }
    )
    return latest.sort_values("priority_score", ascending=False)


def data_explorer_page(frame: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    st.title("Data Explorer")
    section_header(
        "Dataset Inventory",
        "The pricing engine uses the curated modeling table for decisions while preserving raw feature and transaction tables for audit.",
    )
    metadata = load_table_metadata()
    display_metadata = metadata.copy()
    display_metadata["Rows"] = display_metadata["Rows"].apply(lambda value: f"{int(value):,}" if pd.notna(value) else "")
    display_metadata["Columns"] = display_metadata["Columns"].apply(lambda value: f"{int(value):,}" if pd.notna(value) else "")
    display_metadata["Size MB"] = display_metadata["Size MB"].apply(lambda value: f"{value:,.1f}" if pd.notna(value) else "")
    st.dataframe(display_metadata, width="stretch", hide_index=True)

    summary = load_filter_summary()
    if not summary.empty:
        values = dict(zip(summary["metric"], summary["value"]))
        cols = st.columns(4)
        with cols[0]:
            metric_card("Curated Rows", f"{int(values.get('output_rows', len(full_frame))):,}", "modeling-ready", "#1f6f6a")
        with cols[1]:
            metric_card("Excluded Rows", f"{int(values.get('excluded_rows', 0)):,}", "quality filter", "#b5525c")
        with cols[2]:
            metric_card("Curated Units", f"{values.get('output_units', full_frame['quantity_sold'].sum()):,.0f}", "retail units", "#4f7d4b")
        with cols[3]:
            metric_card("Curated Revenue", money(values.get("output_revenue", full_frame["sales_value"].sum())), "decision table", "#d89c38")

        exclusions = summary[
            ~summary["metric"].isin(
                ["input_rows", "output_rows", "excluded_rows", "quantity_cap", "input_units", "output_units", "input_revenue", "output_revenue"]
            )
        ].copy()
        if not exclusions.empty:
            st.plotly_chart(
                px.bar(exclusions, x="value", y="metric", orientation="h", title="Rows Removed by Modeling-Ready Filter"),
                width="stretch",
            )

    section_header("Coverage", "Product, store, category, and week coverage in the selected data slice.")
    coverage_cols = st.columns(6)
    coverage_values = [
        ("Weeks", f"{int(frame['week_no'].min())}-{int(frame['week_no'].max())}", "time range", "#315f8c"),
        ("Products", f"{frame['product_id'].nunique():,}", "unique SKUs", "#1f6f6a"),
        ("Stores", f"{frame['store_id'].nunique():,}", "locations", "#4f7d4b"),
        ("Departments", f"{frame['department'].nunique():,}", "groups", "#d89c38"),
        ("Categories", f"{frame['commodity_desc'].nunique():,}", "commodities", "#6b5b95"),
        ("Brands", f"{frame['brand'].nunique():,}", "brand types", "#b5525c"),
    ]
    for col, item in zip(coverage_cols, coverage_values):
        with col:
            metric_card(*item)

    weekly_coverage = frame.groupby("week_no", as_index=False).agg(
        rows=("quantity_sold", "size"),
        products=("product_id", "nunique"),
        stores=("store_id", "nunique"),
        revenue=("sales_value", "sum"),
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(weekly_coverage, x="week_no", y="products", title="Active Products by Week"), width="stretch")
    with right:
        st.plotly_chart(px.line(weekly_coverage, x="week_no", y="stores", title="Active Stores by Week"), width="stretch")

    section_header("Data Distributions")
    sample = frame.sample(min(len(frame), 120_000), random_state=42) if len(frame) else frame
    left, middle, right = st.columns(3)
    with left:
        st.plotly_chart(px.histogram(sample, x="quantity_sold", nbins=30, title="Quantity Sold"), width="stretch")
    with middle:
        st.plotly_chart(px.histogram(sample, x="avg_unit_price", nbins=40, title="Average Unit Price"), width="stretch")
    with right:
        st.plotly_chart(px.histogram(sample, x="discount_percentage", nbins=40, title="Discount Percentage"), width="stretch")

    section_header("Column Health")
    missing = frame.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_share"]
    st.plotly_chart(px.bar(missing.head(20), x="missing_share", y="column", orientation="h", title="Top Missingness Checks"), width="stretch")
    st.dataframe(frame.head(200), width="stretch", hide_index=True)


def product_category_analytics(frame: pd.DataFrame) -> None:
    st.title("Product and Category Analytics")
    left, right = st.columns(2)
    with left:
        products = frame.groupby("product_id", as_index=False)["sales_value"].sum().sort_values("sales_value", ascending=False).head(20)
        st.plotly_chart(px.bar(products, x="sales_value", y="product_id", orientation="h", title="Top Products by Revenue"), width="stretch")
    with right:
        units = frame.groupby("product_id", as_index=False)["quantity_sold"].sum().sort_values("quantity_sold", ascending=False).head(20)
        st.plotly_chart(px.bar(units, x="quantity_sold", y="product_id", orientation="h", title="Top Products by Quantity"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.scatter(frame, x="avg_unit_price", y="quantity_sold", color="department", title="Price vs Quantity"), width="stretch")
    with right:
        st.plotly_chart(px.scatter(frame, x="discount_percentage", y="quantity_sold", color="brand", title="Discount vs Demand"), width="stretch")

    category = frame.groupby(["week_no", "commodity_desc"], as_index=False)["sales_value"].sum()
    st.plotly_chart(px.line(category, x="week_no", y="sales_value", color="commodity_desc", title="Category Revenue Trend"), width="stretch")


def opportunity_scanner_page(frame: pd.DataFrame) -> None:
    st.title("Opportunity Scanner")
    section_header(
        "Portfolio Actions",
        "Rank product-store pairs by pricing opportunity using revenue, observed markdown exposure, category elasticity, and promotion lift signals.",
    )
    scanner = enrich_opportunity_frame(frame)
    if scanner.empty:
        st.warning("No product-store rows are available for the selected filters.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    min_weeks = c1.slider("Minimum history weeks", 1, 52, 8, step=1)
    lens = c2.selectbox(
        "Opportunity lens",
        ["All actions", "Recover margin", "Test targeted markdown", "Demand recovery promo", "Protect current price"],
    )
    top_n = c3.slider("Rows to show", 10, 100, 30, step=10)

    scanner = scanner[scanner["history_weeks"] >= min_weeks].copy()
    if lens != "All actions":
        scanner = scanner[scanner["recommended_action"] == lens]
    if scanner.empty:
        st.warning("No opportunities match the current lens and history threshold.")
        return

    revenue_covered = scanner["total_revenue"].sum()
    action_counts = scanner["recommended_action"].value_counts()
    top_action = action_counts.idxmax()
    low_sensitivity_share = float((scanner["abs_elasticity"] <= 0.35).mean())
    cols = st.columns(4)
    with cols[0]:
        metric_card("Pairs Scanned", f"{len(scanner):,}", "product-store pairs", "#1f6f6a")
    with cols[1]:
        metric_card("Revenue Covered", money(revenue_covered), "selected slice", "#4f7d4b")
    with cols[2]:
        metric_card("Top Action", top_action, f"{action_counts.max():,} pairs", "#d89c38")
    with cols[3]:
        metric_card("Low Sensitivity", pct(low_sensitivity_share), "category signal", "#315f8c")

    left, right = st.columns([1, 1.25])
    with left:
        action_df = action_counts.rename_axis("action").reset_index(name="pairs")
        st.plotly_chart(px.bar(action_df, x="pairs", y="action", orientation="h", title="Action Mix"), width="stretch")
    with right:
        category_score = (
            scanner.groupby("commodity_desc", as_index=False)["priority_score"]
            .sum()
            .sort_values("priority_score", ascending=False)
            .head(15)
        )
        st.plotly_chart(px.bar(category_score, x="priority_score", y="commodity_desc", orientation="h", title="Opportunity Score by Category"), width="stretch")

    sample = scanner.head(2_000)
    st.plotly_chart(
        px.scatter(
            sample,
            x="category_elasticity",
            y="markdown_exposure",
            size="total_revenue",
            color="recommended_action",
            hover_data=["product_id", "store_id", "commodity_desc", "brand", "decision_cue"],
            title="Elasticity vs Markdown Exposure",
        ),
        width="stretch",
    )

    display_cols = [
        "recommended_action",
        "decision_cue",
        "product_id",
        "store_id",
        "department",
        "commodity_desc",
        "brand",
        "avg_unit_price",
        "quantity_sold",
        "total_revenue",
        "avg_observed_discount",
        "category_elasticity",
        "retail_discount_lift",
        "history_weeks",
        "priority_score",
    ]
    display = scanner.head(top_n)[display_cols].copy()
    display = display.rename(
        columns={
            "avg_unit_price": "latest_price",
            "quantity_sold": "latest_units",
            "avg_observed_discount": "avg_discount",
            "retail_discount_lift": "discount_lift",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)


def product_drilldown_page(frame: pd.DataFrame) -> None:
    st.title("Product Drilldown")
    section_header(
        "SKU and Store View",
        "Inspect one product-store pair across price, demand, promotion exposure, peer prices, and a compact profit curve.",
    )
    product_order = frame.groupby("product_id")["sales_value"].sum().sort_values(ascending=False).index.tolist()
    product = st.selectbox("Product ID", product_order, key="drill_product")
    product_frame = frame[frame["product_id"] == product]
    store_order = product_frame.groupby("store_id")["sales_value"].sum().sort_values(ascending=False).index.tolist()
    store = st.selectbox("Store ID", store_order, key=f"drill_store_{product}")
    series = product_frame[product_frame["store_id"] == store].sort_values("week_no").copy()
    if series.empty:
        st.warning("No history is available for this product-store pair.")
        return

    latest = series.iloc[-1]
    category = str(latest.get("commodity_desc", "Unknown"))
    price_elasticity = lookup_price_elasticity(product_id=product, commodity_desc=category)
    promo_lift = lookup_promotion_effect(commodity_desc=category, mechanism="retail_discount")
    total_revenue = float(series["sales_value"].sum())
    total_units = float(series["quantity_sold"].sum())
    avg_discount = float(series["discount_percentage"].fillna(0).mean())
    current_price = float(latest["avg_unit_price"])
    current_units = float(latest["quantity_sold"])

    cols = st.columns(6)
    values = [
        ("Latest Price", f"${current_price:,.2f}", f"week {int(latest['week_no'])}", "#1f6f6a"),
        ("Latest Units", f"{current_units:,.0f}", "current demand", "#4f7d4b"),
        ("Revenue", money(total_revenue), "selected history", "#d89c38"),
        ("Units", f"{total_units:,.0f}", "selected history", "#315f8c"),
        ("Avg Discount", pct(avg_discount), "observed", "#6b5b95"),
        ("Elasticity", f"{price_elasticity:.2f}", "price response", "#b5525c"),
    ]
    for col, item in zip(cols, values):
        with col:
            metric_card(*item)

    left, right = st.columns([1.4, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series["week_no"], y=series["quantity_sold"], mode="lines+markers", name="Units"))
        fig.add_trace(go.Scatter(x=series["week_no"], y=series["avg_unit_price"], mode="lines+markers", name="Price", yaxis="y2"))
        fig.update_layout(
            title="Price and Demand History",
            xaxis_title="Week",
            yaxis_title="Units",
            yaxis2=dict(title="Price", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.plotly_chart(px.bar(series, x="week_no", y="discount_percentage", title="Observed Discount by Week"), width="stretch")

    left, right = st.columns(2)
    with left:
        peer = latest_product_store_snapshot(frame[frame["commodity_desc"].astype(str) == category])
        st.plotly_chart(px.histogram(peer, x="avg_unit_price", nbins=30, title=f"Peer Price Distribution: {category}"), width="stretch")
    with right:
        st.plotly_chart(px.scatter(series, x="avg_unit_price", y="quantity_sold", color="discount_percentage", title="This SKU: Price vs Units"), width="stretch")

    base = series.tail(1)
    candidate_discounts = [10, 5, 0, -5, -10, -15, -20]
    simulations = simulate_price_candidates(
        product_id=product,
        store_id=store,
        current_price=current_price,
        estimated_unit_cost=current_price * 0.6,
        candidate_discounts=candidate_discounts,
        demand_model=load_dashboard_pricing_model(),
        base_features=base.iloc[0],
        base_quantity=current_units,
        price_elasticity=price_elasticity,
        promotion_effect=promo_lift,
        inventory_limit=max(current_units * 3, 100),
        promotion_cost=0,
        price_history=series,
    )
    sim_table = pd.DataFrame([asdict(row) for row in simulations])
    sim_table["price_change_pct"] = -sim_table["discount"]
    section_header("Compact Profit Curve")
    st.plotly_chart(px.line(sim_table, x="price_change_pct", y="expected_profit", markers=True, title="Price Change vs Expected Profit"), width="stretch")
    history_cols = [
        "week_no",
        "avg_unit_price",
        "quantity_sold",
        "sales_value",
        "discount_percentage",
        "is_display",
        "is_mailer",
        "num_baskets",
        "num_households",
    ]
    st.dataframe(series[history_cols].tail(100), width="stretch", hide_index=True)


def promotion_page(frame: pd.DataFrame) -> None:
    st.title("Promotion and Coupon Intelligence")
    impact = calculate_promotion_impact(frame)
    overall = impact[impact["commodity_desc"] == "all"]
    cols = st.columns(5)
    for col, mechanism in zip(cols, ["display", "mailer", "coupon", "retail_discount", "campaign"]):
        row = overall[overall["mechanism"] == mechanism].iloc[0]
        with col:
            metric_card(mechanism.replace("_", " ").title(), pct(row["lift_percentage"]), "quantity lift")

    chart_data = impact[(impact["commodity_desc"] != "all") & (impact["mechanism"].isin(["display", "mailer", "coupon"]))]
    st.plotly_chart(px.bar(chart_data, x="commodity_desc", y="lift_percentage", color="mechanism", title="Promotion Lift by Category"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.box(frame, x="is_display", y="quantity_sold", title="Display vs No Display Demand"), width="stretch")
    with right:
        st.plotly_chart(px.box(frame, x="is_mailer", y="quantity_sold", title="Mailer vs No Mailer Demand"), width="stretch")


def elasticity_page(frame: pd.DataFrame) -> None:
    st.title("Price Elasticity")
    elasticity = calculate_elasticity_table(frame, min_observations=8)
    if elasticity.empty:
        st.warning("Elasticity estimates need more price variation in the selected slice.")
        return
    category = elasticity[elasticity["group_type"] == "category"].sort_values("price_elasticity")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(px.bar(category.head(30), x="price_elasticity", y="group_key", orientation="h", color="sensitivity_class", title="Elasticity by Category"), width="stretch")
    with right:
        st.plotly_chart(px.histogram(elasticity, x="price_elasticity", color="group_type", title="Elasticity Distribution"), width="stretch")

    sensitive = elasticity.sort_values("price_elasticity").head(20)
    st.dataframe(sensitive, width="stretch", hide_index=True)


def model_performance_page(frame: pd.DataFrame) -> None:
    st.title("Model Performance")
    champion_report = load_champion_report()
    pricing_report = load_json_artifact(str(REPORTS_DIR / "pricing_model_champion.json"))
    diagnostics = load_json_artifact(str(REPORTS_DIR / "model_diagnostics.json"))
    champion = champion_report.get("champion", {})
    candidates = pd.DataFrame(champion_report.get("candidates", []))

    if champion:
        cols = st.columns(5)
        with cols[0]:
            metric_card("Champion", str(champion.get("name", "")).replace("_", " ").title(), "serving model", "#1f6f6a")
        with cols[1]:
            metric_card("Test WAPE", f"{champion.get('score', 0):.3f}", "selection metric", "#4f7d4b")
        with cols[2]:
            metric_card("Test MAE", f"{champion.get('test_metrics', {}).get('mae', 0):.3f}", "unit error", "#315f8c")
        with cols[3]:
            metric_card("Test SMAPE", f"{champion.get('test_metrics', {}).get('smape', 0):.3f}", "relative error", "#d89c38")
        with cols[4]:
            metric_card("Test Rows", "389,041", "time split", "#6b5b95")

    if not candidates.empty:
        score_rows = []
        for row in candidates.to_dict("records"):
            score_rows.append(
                {
                    "model": str(row["name"]).replace("_", " ").title(),
                    "test_wape": row.get("test_metrics", {}).get("wape"),
                    "test_mae": row.get("test_metrics", {}).get("mae"),
                    "test_smape": row.get("test_metrics", {}).get("smape"),
                }
            )
        scores = pd.DataFrame(score_rows)
        st.plotly_chart(px.bar(scores, x="model", y="test_wape", title="Model Competition by Test WAPE"), width="stretch")
        st.dataframe(scores, width="stretch", hide_index=True)

    pricing_champion = pricing_report.get("champion", {})
    if pricing_champion:
        pricing_accepted = bool(pricing_report.get("accepted_for_pricing", pricing_report.get("serving_safe") and pricing_report.get("near_current_accuracy")))
        section_header(
            "Pricing Model Safety",
            "A serving-safe pricing model was trained without same-week outcome leakage. It is used only when it stays close to the demand champion; otherwise the optimizer falls back to the demand champion and shows guardrails.",
        )
        pcols = st.columns(5)
        with pcols[0]:
            metric_card("Pricing Candidate", str(pricing_champion.get("name", "")).replace("_", " ").title(), "scenario model", "#1f6f6a")
        with pcols[1]:
            metric_card("Pricing WAPE", f"{pricing_champion.get('score', 0):.3f}", "test split", "#4f7d4b")
        with pcols[2]:
            metric_card("Accepted", "Yes" if pricing_accepted else "No", "optimizer model", "#4f7d4b" if pricing_accepted else "#b5525c")
        with pcols[3]:
            metric_card("Near Current Accuracy", "Yes" if pricing_report.get("near_current_accuracy") else "No", "within tolerance", "#d89c38")
        with pcols[4]:
            metric_card("Leakage Fields", str(len(pricing_champion.get("leakage_features_present", []))), "present in model", "#6b5b95")

    if diagnostics:
        quantity_bins = pd.DataFrame(diagnostics.get("quantity_bins", []))
        categories = pd.DataFrame(diagnostics.get("categories_by_wape", []))
        if not quantity_bins.empty:
            section_header(
                "Error by Demand Size",
                "This shows where forecast error concentrates. High-demand rows are especially important for pricing because a small unit error can move profit.",
            )
            st.plotly_chart(px.bar(quantity_bins, x="quantity_bin", y="wape", title="WAPE by Actual Demand Bin"), width="stretch")
            st.dataframe(quantity_bins, width="stretch", hide_index=True)
        if not categories.empty:
            section_header("Highest-Error Categories")
            st.dataframe(categories.head(10), width="stretch", hide_index=True)

    section_header(
        "How Recommendations Are Calculated",
        "For each candidate price, the dashboard recomputes price-derived features, predicts demand, calculates expected profit as (price - unit cost) x predicted demand - promotion cost, then applies margin, inventory, price-history, and profit-lift guardrails.",
    )

    section_header("Diagnostics")
    left, right = st.columns(2)
    actual_vs_pred = REPORTS_DIR / "figures" / "actual_vs_predicted_quantity.png"
    residuals = REPORTS_DIR / "figures" / "residual_distribution.png"
    with left:
        if actual_vs_pred.exists():
            st.image(str(actual_vs_pred), caption="Actual vs predicted quantity")
    with right:
        if residuals.exists():
            st.image(str(residuals), caption="Residual distribution")

    section_header("Feature Importance")
    catboost_importance = REPORTS_DIR / "figures" / "catboost_feature_importance.png"
    production_importance = REPORTS_DIR / "figures" / "production_feature_importance.png"
    baseline_importance = REPORTS_DIR / "figures" / "feature_importance.png"
    for path in (catboost_importance, production_importance, baseline_importance):
        if path.exists():
            st.image(str(path), caption=path.stem.replace("_", " ").title())
            break


def forecasting_page(frame: pd.DataFrame) -> None:
    st.title("Demand Forecasting")
    section_header(
        "Forecast Scope",
        "Choose a product first; the store list is then limited to stores with history for that product.",
    )
    product_options = sorted(frame["product_id"].dropna().unique())
    if not product_options:
        st.warning("No products match the selected sidebar filters.")
        return
    product = st.selectbox("Product", product_options, key="forecast_product")
    product_frame = frame[frame["product_id"] == product]
    store_options = sorted(product_frame["store_id"].dropna().unique())
    if not store_options:
        st.warning("No stores have demand history for the selected product under the current filters.")
        return
    store = st.selectbox("Store", store_options, key=f"forecast_store_{product}")
    series = product_frame[product_frame["store_id"] == store].sort_values("week_no").copy()
    if series.empty:
        st.warning("No rows for the selected product-store pair.")
        return
    status_pills(
        [
            ("Rows", f"{len(series):,}"),
            ("Weeks", f"{series['week_no'].nunique():,}"),
            ("Category", str(series["commodity_desc"].iloc[-1]) if "commodity_desc" in series else "Unknown"),
            ("Brand", str(series["brand"].iloc[-1]) if "brand" in series else "Unknown"),
        ]
    )
    model = load_dashboard_model()
    if model is not None:
        series["model_prediction"] = pd.Series(model.predict(series), index=series.index).clip(lower=0)
        prediction_label = "Trained demand model"
    else:
        lag_prediction = series["lag_quantity_1"].astype(float)
        fallback_prediction = series["quantity_sold"].expanding().mean().shift(1).fillna(series["quantity_sold"])
        series["model_prediction"] = lag_prediction.mask(lag_prediction <= 0, fallback_prediction)
        prediction_label = "Lag fallback"
    series["absolute_error"] = (series["quantity_sold"] - series["model_prediction"]).abs()
    series_wape = series["absolute_error"].sum() / max(series["quantity_sold"].abs().sum(), 0.0001)
    series_smape = (
        2
        * series["absolute_error"]
        / (series["quantity_sold"].abs() + series["model_prediction"].abs()).replace(0, np.nan)
    ).fillna(0).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["week_no"], y=series["quantity_sold"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=series["week_no"], y=series["model_prediction"], mode="lines", name=prediction_label))
    fig.update_layout(title="Actual vs Predicted Demand", xaxis_title="Week", yaxis_title="Quantity")
    st.plotly_chart(fig, width="stretch")
    cols = st.columns(4)
    with cols[0]:
        metric_card("Mean Absolute Error", f"{series['absolute_error'].mean():,.2f}", prediction_label)
    with cols[1]:
        metric_card("WAPE", f"{series_wape:.3f}", "selected SKU-store")
    with cols[2]:
        metric_card("SMAPE", f"{series_smape:.3f}", "selected SKU-store")
    with cols[3]:
        metric_card("Latest Prediction", f"{series['model_prediction'].iloc[-1]:,.1f}", f"actual {series['quantity_sold'].iloc[-1]:,.0f}")


def simulator_page(frame: pd.DataFrame) -> None:
    st.title("Scenario Simulator")
    section_header("Scenario Inputs", "Compare demand, revenue, profit, and margin across candidate discounts for the selected product-store pair.")
    product = st.selectbox("Product ID", price_rich_product_options(frame), key="sim_product")
    product_frame = frame[frame["product_id"] == product]
    store = st.selectbox("Store ID", sorted(product_frame["store_id"].unique()), key=f"sim_store_{product}")
    pair_history = product_frame[product_frame["store_id"] == store]
    base = pair_history.sort_values("week_no").tail(1)
    default_price = float(base["avg_unit_price"].iloc[0]) if not base.empty else 2.5
    default_qty = float(base["quantity_sold"].iloc[0]) if not base.empty else 100
    commodity_desc = str(base["commodity_desc"].iloc[0]) if not base.empty and "commodity_desc" in base else None
    price_elasticity = lookup_price_elasticity(product_id=product, commodity_desc=commodity_desc)
    promotion_effect = lookup_promotion_effect(commodity_desc=commodity_desc, mechanism="retail_discount")

    status_pills(
        [
            ("Category", str(commodity_desc or "Unknown")),
            ("Elasticity", f"{price_elasticity:.2f}"),
            ("Discount lift", pct(promotion_effect)),
            ("Latest units", f"{default_qty:,.0f}"),
        ]
    )

    c1, c2, c3, c4 = st.columns(4)
    current_price = c1.number_input("Current price", min_value=0.01, value=round(default_price, 2), step=0.05)
    unit_cost = c2.number_input("Unit cost", min_value=0.01, value=round(default_price * 0.6, 2), step=0.05)
    inventory = c3.number_input("Inventory limit", min_value=1, value=int(max(default_qty * 3, 100)), step=10)
    promo_cost = c4.number_input("Promotion cost", min_value=0.0, value=0.0, step=10.0)
    max_discount = st.slider("Maximum discount", 0, 30, 20, step=5)
    discounts = list(range(0, max_discount + 1, 5))

    simulations = simulate_price_candidates(
        product_id=product,
        store_id=store,
        current_price=current_price,
        estimated_unit_cost=unit_cost,
        candidate_discounts=discounts,
        demand_model=load_dashboard_pricing_model(),
        base_features=base.iloc[0] if not base.empty else None,
        base_quantity=default_qty,
        price_elasticity=price_elasticity,
        promotion_effect=promotion_effect,
        inventory_limit=inventory,
        promotion_cost=promo_cost,
        price_history=pair_history,
    )
    table = pd.DataFrame([asdict(row) for row in simulations])
    best = table.sort_values("expected_profit", ascending=False).iloc[0]
    cols = st.columns(4)
    with cols[0]:
        metric_card("Best Action", format_price_action(best["discount"] / 100).capitalize(), "highest profit", "#1f6f6a")
    with cols[1]:
        metric_card("Best Price", f"${best['candidate_price']:,.2f}", "candidate", "#4f7d4b")
    with cols[2]:
        metric_card("Expected Profit", money(best["expected_profit"]), "scenario", "#d89c38")
    with cols[3]:
        metric_card("Feasible", "Yes" if bool(best["feasible"]) else "No", str(best["constraint_message"]), "#315f8c")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.line(table, x="discount", y="predicted_quantity", markers=True, title="Discount vs Predicted Demand"), width="stretch")
    with right:
        st.plotly_chart(px.line(table, x="discount", y="expected_profit", markers=True, title="Discount vs Expected Profit"), width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)


def recommendation_page(frame: pd.DataFrame) -> None:
    st.title("Profit Optimizer")
    section_header(
        "Pricing Decision",
        "Select a product-store pair, set business constraints, and compare price increases, no change, and discounts by expected demand, revenue, margin, and profit.",
    )
    product = st.selectbox("Product ID", price_rich_product_options(frame), key="rec_product", help="SKU to price.")
    product_frame = frame[frame["product_id"] == product]
    store = st.selectbox("Store ID", sorted(product_frame["store_id"].unique()), key=f"rec_store_{product}", help="Store where the price will be evaluated.")
    pair_history = product_frame[product_frame["store_id"] == store]
    base = pair_history.sort_values("week_no").tail(1)
    default_price = float(base["avg_unit_price"].iloc[0]) if not base.empty else 2.5
    default_qty = float(base["quantity_sold"].iloc[0]) if not base.empty else 100
    commodity_desc = str(base["commodity_desc"].iloc[0]) if not base.empty and "commodity_desc" in base else None
    price_elasticity = lookup_price_elasticity(product_id=product, commodity_desc=commodity_desc)
    promotion_effect = lookup_promotion_effect(commodity_desc=commodity_desc, mechanism="retail_discount")
    c1, c2, c3, c4 = st.columns(4)
    current_price = c1.number_input("Current price", min_value=0.01, value=round(default_price, 2), step=0.05, key="rec_price", help="Shelf or current effective unit price.")
    unit_cost = c2.number_input("Unit cost", min_value=0.01, value=round(default_price * 0.6, 2), step=0.05, key="rec_cost", help="Estimated product cost used to calculate gross margin.")
    inventory = c3.number_input("Inventory limit", min_value=1, value=int(max(default_qty * 3, 100)), step=10, key="rec_inventory", help="Maximum units available to sell.")
    promo_cost = c4.number_input("Promotion cost", min_value=0.0, value=0.0, step=10.0, key="rec_promo_cost", help="Fixed cost for the promotion scenario.")
    decision_mode = st.radio(
        "Optimization mode",
        ["Profit actions", "Promotion discount only"],
        horizontal=True,
        help="Profit actions can recommend price increases, no change, or discounts. Promotion discount only compares no-change against discount candidates.",
    )
    st.caption(
        "Recommendations maximize profit, not discount depth. A discount wins only when the model and elasticity table predict enough extra units to offset the lower margin."
    )
    if decision_mode == "Promotion discount only":
        max_discount = st.slider(
            "Maximum discount to test (%)",
            0,
            30,
            20,
            step=5,
            key="rec_discount_only_range",
            help="Discount mode compares no-change against observed-history-supported markdowns.",
        )
        min_change, max_change = -max_discount, 0
    else:
        min_change, max_change = st.slider(
            "Allowed price change from current (%)",
            -20,
            30,
            (-15, 15),
            step=5,
            key="rec_price_change_range",
            help="Negative values test price cuts; positive values test price increases. Extend this only when you want to evaluate larger moves.",
        )

    context_cols = st.columns(5)
    latest_week = int(base["week_no"].iloc[0]) if not base.empty else 0
    latest_units = float(base["quantity_sold"].iloc[0]) if not base.empty else default_qty
    latest_discount = float(base["discount_percentage"].iloc[0]) if not base.empty else 0
    candidate_discounts = candidate_discounts_from_price_history(
        current_price=current_price,
        price_history=product_frame,
        min_price_change=min_change / 100,
        max_price_change=max_change / 100,
        step=0.05,
    )
    candidate_actions = [format_price_action(discount / 100 if abs(discount) > 1 else discount) for discount in candidate_discounts]
    with context_cols[0]:
        metric_card("Category", str(commodity_desc or "Unknown")[:24], "selected SKU", "#1f6f6a")
    with context_cols[1]:
        metric_card("Latest Units", f"{latest_units:,.0f}", f"week {latest_week}", "#4f7d4b")
    with context_cols[2]:
        metric_card("Elasticity", f"{price_elasticity:.2f}", "price response", "#315f8c")
    with context_cols[3]:
        metric_card("Discount Lift", pct(promotion_effect), "category signal", "#d89c38")
    with context_cols[4]:
        metric_card("Candidate Actions", f"{len(candidate_discounts)}", "supported by product history", "#6b5b95")

    result = recommend_price(
        product_id=product,
        store_id=store,
        current_price=current_price,
        estimated_unit_cost=unit_cost,
        candidate_discounts=candidate_discounts,
        demand_model=load_dashboard_pricing_model(),
        base_features=base.iloc[0] if not base.empty else None,
        base_quantity=default_qty,
        price_elasticity=price_elasticity,
        promotion_effect=promotion_effect,
        inventory_limit=inventory,
        promotion_cost=promo_cost,
        price_history=product_frame,
    )
    raw_table = pd.DataFrame([asdict(row) for row in result.simulations])
    raw_table["price_change_pct"] = raw_table.get("price_change_percentage", -raw_table["discount"])
    raw_table["price_action"] = raw_table["discount"].map(lambda value: format_price_action(value / 100).capitalize())
    baseline_rows = raw_table[raw_table["discount"] == 0]
    baseline_quantity = float(baseline_rows["predicted_quantity"].iloc[0]) if not baseline_rows.empty else float(result.predicted_quantity)
    baseline_profit = float(baseline_rows["expected_profit"].iloc[0]) if not baseline_rows.empty else float(result.expected_profit)
    raw_table["predicted_demand_lift"] = raw_table.get(
        "demand_lift", raw_table["predicted_quantity"] / max(baseline_quantity, 0.0001) - 1
    )
    raw_table["required_lift_to_match_current"] = raw_table.get(
        "required_demand_lift",
        raw_table.apply(
            lambda row: ((baseline_profit + promo_cost) / max(row["candidate_price"] - unit_cost, 0.0001)) / max(baseline_quantity, 0.0001) - 1,
            axis=1,
        ),
    )
    table = raw_table.copy()
    table["margin"] = table["margin_percentage"].map(lambda value: f"{value:.1%}")
    table["predicted_demand_lift"] = table["predicted_demand_lift"].map(lambda value: f"{value:.1%}")
    table["revenue_lift"] = table["revenue_lift"].map(lambda value: f"{value:.1%}") if "revenue_lift" in table else ""
    table["profit_lift"] = table["profit_lift"].map(lambda value: f"{value:.1%}") if "profit_lift" in table else ""
    table["required_lift_to_match_current"] = table["required_lift_to_match_current"].map(lambda value: f"{value:.1%}")
    recommended_matches = raw_table[(raw_table["candidate_price"] - result.recommended_price).abs() < 0.0001]
    recommended_row = recommended_matches.iloc[0] if not recommended_matches.empty else raw_table.sort_values("expected_profit", ascending=False).iloc[0]
    profit_delta = result.expected_profit - baseline_profit
    profit_lift = profit_delta / max(abs(baseline_profit), 0.0001)
    economics_cols = st.columns(6)
    with economics_cols[0]:
        metric_card("Current Profit", money(baseline_profit), "0% price change", "#315f8c")
    with economics_cols[1]:
        metric_card("Recommended Profit", money(result.expected_profit), "best candidate", "#1f6f6a")
    with economics_cols[2]:
        metric_card("Profit Lift", signed_pct(profit_lift), "vs current", "#4f7d4b" if profit_delta >= 0 else "#b5525c")
    with economics_cols[3]:
        metric_card("Price Change", signed_pct(-result.recommended_discount / 100), format_price_action(result.recommended_discount / 100), "#d89c38")
    with economics_cols[4]:
        metric_card("Confidence", result.confidence.title(), result.guardrail_message.replace("_", " "), "#4f7d4b" if result.confidence == "high" else "#b5525c")
    with economics_cols[5]:
        metric_card("Risk Adj Profit", money(float(recommended_row.get("risk_adjusted_profit", result.expected_profit))), "selection score", "#6b5b95")
    st.markdown(
        f"""
          <div class="recommendation">
          <div class="metric-label">Profit-maximizing price</div>
          <div class="decision-number">${result.recommended_price:,.2f}</div>
          <p><b>Price action:</b> {format_price_action(result.recommended_discount / 100).capitalize()} &nbsp;
          <b>Expected Demand:</b> {result.predicted_quantity:,.1f} &nbsp;
          <b>Expected Revenue:</b> ${result.expected_revenue:,.0f} &nbsp;
          <b>Expected Profit:</b> ${result.expected_profit:,.0f}</p>
          <p>{result.business_reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.guardrail_message != "ok":
        st.markdown(
            f'<div class="risk-note">Confidence guardrail: {result.guardrail_message.replace("_", " ")}. Treat this recommendation as decision support and review the scenario before action.</div>',
            unsafe_allow_html=True,
        )
    st.caption("Candidate actions tested from observed product price history: " + ", ".join(action.capitalize() for action in candidate_actions))
    recommended_change = -result.recommended_discount
    if abs(recommended_change) > 1e-9 and (
        abs(recommended_change - min_change) < 1e-9 or abs(recommended_change - max_change) < 1e-9
    ):
        st.markdown(
            '<div class="risk-note">The selected action is at the edge of the tested price range. This means the curve is still moving at the boundary, not that the boundary is automatically a safe real-world optimum.</div>',
            unsafe_allow_html=True,
        )

    if not bool(recommended_row.get("feasible", True)):
        st.markdown(
            '<div class="risk-note">Highest-profit scenario violates one or more constraints. Review margin, inventory, or discount policy before using it.</div>',
            unsafe_allow_html=True,
        )

    if abs(result.recommended_discount) < 1e-9:
        discount_rows = raw_table[raw_table["discount"] > 0].sort_values("expected_profit", ascending=False)
        if not discount_rows.empty:
            alternative = discount_rows.iloc[0]
            st.markdown(
                f"""
                <div class="risk-note">
                  No-change wins because the best discount scenario predicts {alternative['predicted_demand_lift']:.1%} demand lift,
                  but needs about {alternative['required_lift_to_match_current']:.1%} lift to beat current-price profit.
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_header("Scenario Comparison")
    section_header(
        "Decision Math",
        "Demand: Q1 = Q0 x (P1/P0)^elasticity x promotion lift. Revenue: P x Q. Profit: (P - unit cost) x Q - promo cost. The selected action maximizes risk-adjusted incremental profit after margin, inventory, and price-history checks.",
    )
    left, right = st.columns([1.25, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_table["price_change_pct"], y=raw_table["expected_profit"], mode="lines+markers", name="Expected profit"))
        fig.add_trace(go.Scatter(x=raw_table["price_change_pct"], y=raw_table["expected_revenue"], mode="lines+markers", name="Expected revenue", yaxis="y2"))
        fig.update_layout(
            title="Price Change vs Profit and Revenue",
            xaxis_title="Price change %",
            yaxis_title="Expected profit",
            yaxis2=dict(title="Expected revenue", overlaying="y", side="right", showgrid=False),
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.plotly_chart(px.line(raw_table, x="candidate_price", y="predicted_quantity", markers=True, title="Price-Demand Curve"), width="stretch")

    display_cols = [
        "price_action",
        "price_change_pct",
        "candidate_price",
        "contribution_margin",
        "predicted_quantity",
        "predicted_demand_lift",
        "revenue_lift",
        "profit_lift",
        "required_lift_to_match_current",
        "expected_revenue",
        "expected_profit",
        "risk_adjusted_profit",
        "margin",
        "confidence",
        "guardrail_message",
        "feasible",
        "constraint_message",
    ]
    st.dataframe(table[display_cols], width="stretch", hide_index=True)


features, demo_mode = load_features()
filtered = filter_frame(features)
if demo_mode:
    st.sidebar.info("Using generated demo data until the processed feature table is built.")

page = st.sidebar.radio(
    "Page",
    [
        "Executive Overview",
        "Data Explorer",
        "Opportunity Scanner",
        "Product Drilldown",
        "Product and Category Analytics",
        "Promotion and Coupon Intelligence",
        "Price Elasticity",
        "Model Performance",
        "Demand Forecasting",
        "Scenario Simulator",
        "Profit Optimizer",
    ],
)

if filtered.empty:
    empty_filter_state(page, features)
    st.stop()

if page == "Executive Overview":
    executive_overview(filtered)
elif page == "Data Explorer":
    data_explorer_page(filtered, features)
elif page == "Opportunity Scanner":
    opportunity_scanner_page(filtered)
elif page == "Product Drilldown":
    product_drilldown_page(filtered)
elif page == "Product and Category Analytics":
    product_category_analytics(filtered)
elif page == "Promotion and Coupon Intelligence":
    promotion_page(filtered)
elif page == "Price Elasticity":
    elasticity_page(filtered)
elif page == "Model Performance":
    model_performance_page(filtered)
elif page == "Demand Forecasting":
    forecasting_page(filtered)
elif page == "Scenario Simulator":
    simulator_page(filtered)
else:
    recommendation_page(filtered)

