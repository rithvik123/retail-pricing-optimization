from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs
from src.utils.io import write_json
from src.utils.metrics import regression_report


NUMERIC_FEATURES = [
    "avg_unit_price",
    "median_unit_price",
    "discount_percentage",
    "is_display",
    "is_mailer",
    "week_no",
    "lag_quantity_1",
    "lag_quantity_2",
    "lag_quantity_4",
    "rolling_quantity_mean_4",
    "rolling_quantity_mean_8",
    "rolling_quantity_std_4",
    "price_change",
    "price_lag_1",
    "total_retail_discount",
    "total_coupon_discount",
    "num_baskets",
    "num_households",
]

CATEGORICAL_FEATURES = [
    "department",
    "commodity_desc",
    "brand",
    "store_id",
    "product_id",
]

TARGET = "quantity_sold"


def make_regressor():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)


def available_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in NUMERIC_FEATURES if feature in frame.columns]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in frame.columns]
    return numeric, categorical


def prepare_model_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame = frame[frame[TARGET].notna()].copy()
    for column in NUMERIC_FEATURES:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    for column in CATEGORICAL_FEATURES:
        if column in frame:
            frame[column] = frame[column].fillna("Unknown").astype(str)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce").fillna(0).clip(lower=0)
    return frame


def time_based_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(frame["week_no"].dropna().unique())
    if len(weeks) < 4:
        train = frame.sample(frac=0.7, random_state=42)
        remaining = frame.drop(train.index)
        valid = remaining.sample(frac=0.5, random_state=42) if len(remaining) else remaining
        test = remaining.drop(valid.index)
        return train, valid, test

    train_cut = weeks[int(len(weeks) * 0.7)]
    valid_cut = weeks[int(len(weeks) * 0.85)]
    train = frame[frame["week_no"] <= train_cut]
    valid = frame[(frame["week_no"] > train_cut) & (frame["week_no"] <= valid_cut)]
    test = frame[frame["week_no"] > valid_cut]
    return train, valid, test


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", max_categories=100),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_regressor()),
        ]
    )


def save_feature_importance(model: Pipeline, output_path: Path) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False).head(25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    importance.sort_values().plot(kind="barh", color="#2F6F73")
    plt.title("Baseline Demand Model Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_baseline_model(
    features: pd.DataFrame,
    model_path: Path = MODELS_DIR / "baseline_demand_model.pkl",
    metrics_path: Path = REPORTS_DIR / "baseline_metrics.json",
    feature_importance_path: Path = FIGURES_DIR / "feature_importance.png",
) -> dict[str, object]:
    frame = prepare_model_frame(features)
    numeric, categorical = available_features(frame)
    train, valid, test = time_based_split(frame)
    feature_columns = numeric + categorical

    model = build_pipeline(numeric, categorical)
    model.fit(train[feature_columns], train[TARGET])

    metrics = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "features": feature_columns,
    }
    if len(valid):
        metrics["validation"] = regression_report(valid[TARGET], model.predict(valid[feature_columns]))
    if len(test):
        metrics["test"] = regression_report(test[TARGET], model.predict(test[feature_columns]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    write_json(metrics, metrics_path)
    save_feature_importance(model, feature_importance_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline demand model.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "baseline_demand_model.pkl"))
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    metrics = train_baseline_model(features, Path(args.model_path))
    print(metrics)


if __name__ == "__main__":
    main()
