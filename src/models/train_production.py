from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.train_baseline import CATEGORICAL_FEATURES, TARGET, prepare_model_frame, time_based_split
from src.utils.io import write_json
from src.utils.metrics import regression_report


BASE_NUMERIC_FEATURES = [
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

AUGMENTED_NUMERIC_FEATURES = [
    "log_avg_unit_price",
    "price_ratio_lag_1",
    "discount_x_display",
    "discount_x_mailer",
    "any_promotion",
    "lag_velocity_4",
    "lag_velocity_8",
]


class RetailDemandFeatureAugmenter(BaseEstimator, TransformerMixin):
    """Add price and promotion interaction features without peeking at the target."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in BASE_NUMERIC_FEATURES:
            if column not in frame:
                frame[column] = 0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

        safe_price = frame["avg_unit_price"].clip(lower=0)
        safe_price_lag = frame["price_lag_1"].replace(0, np.nan)
        frame["log_avg_unit_price"] = np.log1p(safe_price)
        frame["price_ratio_lag_1"] = (frame["avg_unit_price"] / safe_price_lag).replace([np.inf, -np.inf], np.nan).fillna(1)
        frame["discount_x_display"] = frame["discount_percentage"] * frame["is_display"]
        frame["discount_x_mailer"] = frame["discount_percentage"] * frame["is_mailer"]
        frame["any_promotion"] = (
            (frame["discount_percentage"] > 0)
            | (frame["is_display"] > 0)
            | (frame["is_mailer"] > 0)
            | (frame["total_coupon_discount"] > 0)
        ).astype(int)
        frame["lag_velocity_4"] = frame["lag_quantity_1"] / frame["rolling_quantity_mean_4"].replace(0, np.nan)
        frame["lag_velocity_8"] = frame["lag_quantity_1"] / frame["rolling_quantity_mean_8"].replace(0, np.nan)
        frame[["lag_velocity_4", "lag_velocity_8"]] = frame[["lag_velocity_4", "lag_velocity_8"]].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        return frame


def make_poisson_regressor():
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError("LightGBM is required for the production count-demand model.") from exc

    return LGBMRegressor(
        objective="poisson",
        metric="poisson",
        n_estimators=550,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def available_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in BASE_NUMERIC_FEATURES if feature in frame.columns]
    numeric += AUGMENTED_NUMERIC_FEATURES
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in frame.columns]
    return numeric, categorical


def build_production_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", max_categories=150),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("feature_augmenter", RetailDemandFeatureAugmenter()),
            ("preprocessor", preprocessor),
            ("model", make_poisson_regressor()),
        ]
    )


def save_feature_importance(model: Pipeline, output_path: Path) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names).sort_values(ascending=False).head(30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 8))
    importance.sort_values().plot(kind="barh", color="#2F6F73")
    plt.title("Production Count-Demand Model Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_production_model(
    features: pd.DataFrame,
    model_path: Path = MODELS_DIR / "production_demand_model.pkl",
    metrics_path: Path = REPORTS_DIR / "production_metrics.json",
    feature_importance_path: Path = FIGURES_DIR / "production_feature_importance.png",
) -> dict[str, object]:
    frame = prepare_model_frame(features)
    train, valid, test = time_based_split(frame)
    numeric, categorical = available_columns(frame)
    feature_columns = [feature for feature in BASE_NUMERIC_FEATURES if feature in frame.columns] + categorical

    model = build_production_pipeline(numeric, categorical)
    model.fit(train[feature_columns], train[TARGET].clip(lower=0))

    metrics: dict[str, object] = {
        "model_type": "LightGBM Poisson count-demand regressor",
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "base_features": feature_columns,
        "augmented_features": AUGMENTED_NUMERIC_FEATURES,
    }
    if len(valid):
        valid_pred = pd.Series(model.predict(valid[feature_columns]), index=valid.index).clip(lower=0)
        metrics["validation"] = regression_report(valid[TARGET], valid_pred)
    if len(test):
        test_pred = pd.Series(model.predict(test[feature_columns]), index=test.index).clip(lower=0)
        metrics["test"] = regression_report(test[TARGET], test_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    write_json(metrics, metrics_path)
    save_feature_importance(model, feature_importance_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production LightGBM Poisson demand model.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "production_demand_model.pkl"))
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    metrics = train_production_model(features, Path(args.model_path))
    print(metrics)


if __name__ == "__main__":
    main()

