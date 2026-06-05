from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.train_baseline import TARGET, available_features, prepare_model_frame, time_based_split
from src.utils.metrics import regression_report, wape


def quantity_bin(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-0.1, 1, 2, 5, 10, float("inf")],
        labels=["1", "2", "3-5", "6-10", "11+"],
    ).astype(str)


def group_metrics(frame: pd.DataFrame, group_col: str, min_rows: int = 250) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(group_col, dropna=False):
        if len(group) < min_rows:
            continue
        rows.append(
            {
                group_col: str(key),
                "rows": int(len(group)),
                "actual_units": float(group["actual"].sum()),
                "predicted_units": float(group["predicted"].sum()),
                "mae": float((group["actual"] - group["predicted"]).abs().mean()),
                "wape": wape(group["actual"], group["predicted"]),
            }
        )
    return sorted(rows, key=lambda row: row["wape"], reverse=True)


def save_diagnostic_plots(diagnostics: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample = diagnostics.sample(min(len(diagnostics), 100_000), random_state=42)

    plt.figure(figsize=(8, 8))
    plt.scatter(sample["actual"], sample["predicted"], s=4, alpha=0.2, color="#2F6F73")
    max_value = max(float(sample["actual"].max()), float(sample["predicted"].max()), 1)
    plt.plot([0, max_value], [0, max_value], color="#B94E48", linewidth=1)
    plt.xlabel("Actual quantity")
    plt.ylabel("Predicted quantity")
    plt.title("Actual vs Predicted Demand")
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted_quantity.png", dpi=160)
    plt.close()

    residual = diagnostics["actual"] - diagnostics["predicted"]
    plt.figure(figsize=(10, 6))
    residual.clip(lower=-10, upper=10).hist(bins=60, color="#2F6F73")
    plt.title("Residual Distribution (Clipped to +/- 10)")
    plt.xlabel("Actual - predicted")
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_distribution.png", dpi=160)
    plt.close()


def write_diagnostics_report(payload: dict[str, object], output_path: Path = REPORTS_DIR / "model_diagnostics.md") -> None:
    overall = payload["overall"]
    quantity_bins = payload["quantity_bins"][:8]
    departments = payload["departments_by_wape"][:10]
    categories = payload["categories_by_wape"][:10]

    lines = [
        "# Demand Model Diagnostics",
        "",
        "Diagnostics are calculated on the time-based test split using the retail modeling-ready feature table.",
        "",
        "## Overall Test Metrics",
        "",
        f"- MAE: {overall['mae']:.4f}",
        f"- RMSE: {overall['rmse']:.4f}",
        f"- SMAPE: {overall['smape']:.4f}",
        f"- WAPE: {overall['wape']:.4f}",
        "",
        "## Quantity Bin Metrics",
        "",
        "| Actual Quantity Bin | Rows | Actual Units | Predicted Units | MAE | WAPE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in quantity_bins:
        lines.append(
            f"| {row['quantity_bin']} | {row['rows']:,} | {row['actual_units']:,.0f} | "
            f"{row['predicted_units']:,.0f} | {row['mae']:.4f} | {row['wape']:.4f} |"
        )
    lines.extend(["", "## Highest-WAPE Departments", "", "| Department | Rows | WAPE | MAE |", "| --- | ---: | ---: | ---: |"])
    for row in departments:
        lines.append(f"| {row['department']} | {row['rows']:,} | {row['wape']:.4f} | {row['mae']:.4f} |")
    lines.extend(["", "## Highest-WAPE Categories", "", "| Category | Rows | WAPE | MAE |", "| --- | ---: | ---: | ---: |"])
    for row in categories:
        lines.append(f"| {row['commodity_desc']} | {row['rows']:,} | {row['wape']:.4f} | {row['mae']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The curated model is now modeling ordinary retail unit demand rather than fuel/coupon pseudo-units.",
            "- Remaining error should be reviewed by high-WAPE departments/categories before using recommendations in production.",
            "- The next model improvement should focus on better treatment of sparse product-store histories and promotion timing.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model_diagnostics(
    features_path: str | Path = default_features_path(),
    model_path: str | Path = MODELS_DIR / "baseline_demand_model.pkl",
) -> dict[str, object]:
    features = pd.read_parquet(features_path)
    frame = prepare_model_frame(features)
    _, _, test = time_based_split(frame)
    numeric, categorical = available_features(frame)
    feature_columns = numeric + categorical
    model = joblib.load(model_path)
    predictions = model.predict(test[feature_columns])
    diagnostics = test[["department", "commodity_desc", TARGET]].copy()
    diagnostics["actual"] = diagnostics[TARGET].astype(float)
    diagnostics["predicted"] = pd.Series(predictions, index=diagnostics.index).clip(lower=0)
    diagnostics["quantity_bin"] = quantity_bin(diagnostics["actual"])

    payload: dict[str, object] = {
        "features_path": str(features_path),
        "model_path": str(model_path),
        "test_rows": int(len(test)),
        "overall": regression_report(diagnostics["actual"], diagnostics["predicted"]),
        "quantity_bins": group_metrics(diagnostics, "quantity_bin", min_rows=1),
        "departments_by_wape": group_metrics(diagnostics, "department"),
        "categories_by_wape": group_metrics(diagnostics, "commodity_desc"),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_diagnostics_report(payload)
    save_diagnostic_plots(diagnostics)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create baseline demand model diagnostics.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "baseline_demand_model.pkl"))
    args = parser.parse_args()

    ensure_project_dirs()
    payload = run_model_diagnostics(args.features_path, args.model_path)
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
