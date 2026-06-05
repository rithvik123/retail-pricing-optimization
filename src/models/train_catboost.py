from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.model_wrappers import CatBoostDemandModel
from src.models.train_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, prepare_model_frame, time_based_split
from src.utils.io import write_json
from src.utils.metrics import regression_report


CATBOOST_NUMERIC_FEATURES = [
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


def prepare_catboost_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = prepare_model_frame(features)
    numeric_features = [feature for feature in CATBOOST_NUMERIC_FEATURES if feature in frame.columns]
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature in frame.columns]

    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    for column in categorical_features:
        frame[column] = frame[column].fillna("Unknown").astype(str)

    return frame, numeric_features, categorical_features


def make_catboost_regressor(iterations: int = 800):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=0.06,
        depth=8,
        l2_leaf_reg=6,
        random_seed=42,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        verbose=100,
        thread_count=-1,
    )


def save_feature_importance(model, feature_columns: list[str], output_path: Path) -> None:
    importance = pd.Series(model.get_feature_importance(), index=feature_columns).sort_values(ascending=False).head(30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 8))
    importance.sort_values().plot(kind="barh", color="#2F6F73")
    plt.title("CatBoost Demand Model Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def train_catboost_model(
    features: pd.DataFrame,
    model_path: Path = MODELS_DIR / "catboost_demand_model.pkl",
    native_model_path: Path = MODELS_DIR / "catboost_demand_model.cbm",
    metrics_path: Path = REPORTS_DIR / "catboost_metrics.json",
    feature_importance_path: Path = FIGURES_DIR / "catboost_feature_importance.png",
    iterations: int = 800,
) -> dict[str, object]:
    frame, numeric_features, categorical_features = prepare_catboost_frame(features)
    train, valid, test = time_based_split(frame)
    feature_columns = numeric_features + categorical_features

    model = make_catboost_regressor(iterations=iterations)
    model.fit(
        train[feature_columns],
        train[TARGET],
        cat_features=categorical_features,
        eval_set=(valid[feature_columns], valid[TARGET]) if len(valid) else None,
        use_best_model=bool(len(valid)),
    )

    metrics: dict[str, object] = {
        "model_type": "CatBoostRegressor",
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "features": feature_columns,
        "categorical_features": categorical_features,
        "iterations_requested": iterations,
        "best_iteration": int(model.get_best_iteration() or iterations),
    }
    if len(valid):
        valid_pred = pd.Series(model.predict(valid[feature_columns]), index=valid.index).clip(lower=0)
        metrics["validation"] = regression_report(valid[TARGET], valid_pred)
    if len(test):
        test_pred = pd.Series(model.predict(test[feature_columns]), index=test.index).clip(lower=0)
        metrics["test"] = regression_report(test[TARGET], test_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(CatBoostDemandModel(model, feature_columns, categorical_features), model_path)
    model.save_model(native_model_path)
    write_json(metrics, metrics_path)
    save_feature_importance(model, feature_columns, feature_importance_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost demand model.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "catboost_demand_model.pkl"))
    parser.add_argument("--iterations", type=int, default=800)
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    metrics = train_catboost_model(features, model_path=Path(args.model_path), iterations=args.iterations)
    print(metrics)


if __name__ == "__main__":
    main()
