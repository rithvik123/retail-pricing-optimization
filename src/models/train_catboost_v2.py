from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.model_wrappers import CatBoostDemandModel
from src.models.serving_features import (
    LEAKAGE_FEATURES,
    PRICING_NUMERIC_FEATURES,
    add_serving_history_features,
    augment_pricing_features,
)
from src.models.train_baseline import CATEGORICAL_FEATURES, TARGET, prepare_model_frame, time_based_split
from src.utils.io import write_json
from src.utils.metrics import regression_report


@dataclass(frozen=True)
class CatBoostV2Candidate:
    name: str
    model_path: Path
    native_model_path: Path
    metrics_path: Path
    feature_importance_path: Path
    loss_function: str
    eval_metric: str = "MAE"
    learning_rate: float = 0.05
    depth: int = 8
    l2_leaf_reg: float = 8
    use_high_demand_weights: bool = False


CANDIDATES = [
    CatBoostV2Candidate(
        name="catboost_v2_rmse",
        model_path=MODELS_DIR / "catboost_v2_rmse_model.pkl",
        native_model_path=MODELS_DIR / "catboost_v2_rmse_model.cbm",
        metrics_path=REPORTS_DIR / "catboost_v2_rmse_metrics.json",
        feature_importance_path=FIGURES_DIR / "catboost_v2_rmse_feature_importance.png",
        loss_function="RMSE",
    ),
    CatBoostV2Candidate(
        name="catboost_v2_weighted_mae",
        model_path=MODELS_DIR / "catboost_v2_weighted_mae_model.pkl",
        native_model_path=MODELS_DIR / "catboost_v2_weighted_mae_model.cbm",
        metrics_path=REPORTS_DIR / "catboost_v2_weighted_mae_metrics.json",
        feature_importance_path=FIGURES_DIR / "catboost_v2_weighted_mae_feature_importance.png",
        loss_function="MAE",
        learning_rate=0.04,
        l2_leaf_reg=10,
        use_high_demand_weights=True,
    ),
    CatBoostV2Candidate(
        name="catboost_v2_tweedie",
        model_path=MODELS_DIR / "catboost_v2_tweedie_model.pkl",
        native_model_path=MODELS_DIR / "catboost_v2_tweedie_model.cbm",
        metrics_path=REPORTS_DIR / "catboost_v2_tweedie_metrics.json",
        feature_importance_path=FIGURES_DIR / "catboost_v2_tweedie_feature_importance.png",
        loss_function="Tweedie:variance_power=1.3",
        learning_rate=0.04,
        l2_leaf_reg=10,
    ),
]


def prepare_v2_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = augment_pricing_features(add_serving_history_features(prepare_model_frame(features)))
    numeric_features = [feature for feature in PRICING_NUMERIC_FEATURES if feature in frame.columns]
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature in frame.columns]

    for column in numeric_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    for column in categorical_features:
        frame[column] = frame[column].fillna("Unknown").astype(str)

    leakage_present = sorted(set(numeric_features + categorical_features).intersection(LEAKAGE_FEATURES))
    if leakage_present:
        raise ValueError(f"Pricing v2 feature list includes leakage fields: {leakage_present}")

    return frame, numeric_features, categorical_features


def make_catboost_regressor(candidate: CatBoostV2Candidate, iterations: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function=candidate.loss_function,
        eval_metric=candidate.eval_metric,
        iterations=iterations,
        learning_rate=candidate.learning_rate,
        depth=candidate.depth,
        l2_leaf_reg=candidate.l2_leaf_reg,
        random_seed=42,
        od_type="Iter",
        od_wait=70,
        allow_writing_files=False,
        verbose=100,
        thread_count=-1,
    )


def high_demand_weights(target: pd.Series) -> pd.Series:
    y = pd.to_numeric(target, errors="coerce").fillna(0).clip(lower=0)
    return 1.0 + np.sqrt(y.clip(upper=25))


def quantity_bin_report(actual: pd.Series, predicted: pd.Series) -> list[dict[str, object]]:
    frame = pd.DataFrame({"actual": actual.astype(float), "predicted": predicted.astype(float)})
    conditions = [
        frame["actual"] >= 11,
        frame["actual"].between(6, 10),
        frame["actual"].between(3, 5),
        frame["actual"].between(2, 2),
        frame["actual"].between(1, 1),
    ]
    frame["quantity_bin"] = np.select(conditions, ["11+", "6-10", "3-5", "2", "1"], default="0")
    rows = []
    for quantity_bin, group in frame.groupby("quantity_bin", sort=False):
        rows.append(
            {
                "quantity_bin": str(quantity_bin),
                "rows": int(len(group)),
                "actual_units": float(group["actual"].sum()),
                "predicted_units": float(group["predicted"].sum()),
                "mae": float((group["actual"] - group["predicted"]).abs().mean()),
                "wape": float((group["actual"] - group["predicted"]).abs().sum() / max(group["actual"].abs().sum(), 1e-9)),
            }
        )
    return rows


def segment_wape_report(frame: pd.DataFrame, actual: pd.Series, predicted: pd.Series, column: str, top_n: int = 10) -> list[dict[str, object]]:
    if column not in frame:
        return []
    scored = frame[[column]].copy()
    scored["actual"] = actual.astype(float)
    scored["predicted"] = predicted.astype(float)
    rows = []
    for value, group in scored.groupby(column, observed=True):
        actual_units = float(group["actual"].sum())
        if actual_units <= 0:
            continue
        rows.append(
            {
                column: str(value),
                "rows": int(len(group)),
                "actual_units": actual_units,
                "predicted_units": float(group["predicted"].sum()),
                "mae": float((group["actual"] - group["predicted"]).abs().mean()),
                "wape": float((group["actual"] - group["predicted"]).abs().sum() / actual_units),
            }
        )
    return sorted(rows, key=lambda row: row["wape"], reverse=True)[:top_n]


def save_feature_importance(model, feature_columns: list[str], output_path: Path) -> None:
    importance = pd.Series(model.get_feature_importance(), index=feature_columns).sort_values(ascending=False).head(30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 8))
    importance.sort_values().plot(kind="barh", color="#2F6F73")
    plt.title("CatBoost v2 Pricing-Safe Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_predictions(split: pd.DataFrame, predicted: pd.Series) -> dict[str, object]:
    actual = split[TARGET]
    return {
        **regression_report(actual, predicted),
        "quantity_bins": quantity_bin_report(actual, predicted),
        "departments_by_wape": segment_wape_report(split, actual, predicted, "department"),
        "categories_by_wape": segment_wape_report(split, actual, predicted, "commodity_desc"),
    }


def train_candidate(
    candidate: CatBoostV2Candidate,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    iterations: int,
) -> dict[str, object]:
    model = make_catboost_regressor(candidate, iterations=iterations)
    sample_weight = high_demand_weights(train[TARGET]) if candidate.use_high_demand_weights else None
    model.fit(
        train[feature_columns],
        train[TARGET],
        cat_features=categorical_features,
        sample_weight=sample_weight,
        eval_set=(valid[feature_columns], valid[TARGET]) if len(valid) else None,
        use_best_model=bool(len(valid)),
    )

    metrics: dict[str, object] = {
        "model_type": "CatBoostRegressor v2 pricing-safe",
        "candidate": candidate.name,
        "loss_function": candidate.loss_function,
        "weighted_high_demand": candidate.use_high_demand_weights,
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "features": feature_columns,
        "categorical_features": categorical_features,
        "excluded_leakage_features": list(LEAKAGE_FEATURES),
        "leakage_features_present": sorted(set(feature_columns).intersection(LEAKAGE_FEATURES)),
        "scenario_safe": True,
        "iterations_requested": iterations,
        "best_iteration": int(model.get_best_iteration() or iterations),
    }
    if len(valid):
        valid_pred = pd.Series(model.predict(valid[feature_columns]), index=valid.index).clip(lower=0)
        metrics["validation"] = evaluate_predictions(valid, valid_pred)
    if len(test):
        test_pred = pd.Series(model.predict(test[feature_columns]), index=test.index).clip(lower=0)
        metrics["test"] = evaluate_predictions(test, test_pred)

    candidate.model_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = CatBoostDemandModel(model, feature_columns, categorical_features, feature_augmenter="pricing_v2")
    joblib.dump(wrapper, candidate.model_path)
    model.save_model(candidate.native_model_path)
    write_json(metrics, candidate.metrics_path)
    save_feature_importance(model, feature_columns, candidate.feature_importance_path)
    return metrics


def incumbent_champion_wape() -> float | None:
    champion_report_path = REPORTS_DIR / "model_champion.json"
    if not champion_report_path.exists():
        return None
    report = json.loads(champion_report_path.read_text(encoding="utf-8"))
    score = report.get("champion", {}).get("score")
    return float(score) if score is not None else None


def promote_v2_champions(results: list[dict[str, object]], incumbent_wape: float | None, near_accuracy_tolerance: float) -> dict[str, object]:
    evaluated = [
        {
            "name": result["candidate"],
            "model_path": str(next(candidate.model_path for candidate in CANDIDATES if candidate.name == result["candidate"])),
            "metrics_path": str(next(candidate.metrics_path for candidate in CANDIDATES if candidate.name == result["candidate"])),
            "selection_metric": "wape",
            "selection_split": "test",
            "score": float(result.get("test", {}).get("wape", np.inf)),
            "test_metrics": {
                key: result.get("test", {}).get(key)
                for key in ("mae", "rmse", "smape", "wape")
            },
            "validation_metrics": {
                key: result.get("validation", {}).get(key)
                for key in ("mae", "rmse", "smape", "wape")
            },
            "scenario_safe": bool(result.get("scenario_safe")),
            "leakage_features_present": result.get("leakage_features_present", []),
        }
        for result in results
        if result.get("test", {}).get("wape") is not None
    ]
    if not evaluated:
        raise RuntimeError("No CatBoost v2 candidates produced test WAPE.")

    best_pricing = min(evaluated, key=lambda row: row["score"])
    best_pricing_path = Path(best_pricing["model_path"])
    pricing_champion_path = MODELS_DIR / "champion_pricing_model.pkl"

    near_current_accuracy = True
    if incumbent_wape is not None:
        near_current_accuracy = best_pricing["score"] <= incumbent_wape * (1 + near_accuracy_tolerance)
    accepted_for_pricing = bool(best_pricing.get("scenario_safe") and near_current_accuracy)
    if accepted_for_pricing:
        shutil.copy2(best_pricing_path, pricing_champion_path)

    pricing_report = {
        "champion": best_pricing,
        "candidates": sorted(evaluated, key=lambda row: row["score"]),
        "champion_model_path": str(pricing_champion_path) if accepted_for_pricing else None,
        "best_candidate_model_path": str(best_pricing_path),
        "incumbent_demand_wape": incumbent_wape,
        "near_accuracy_tolerance": near_accuracy_tolerance,
        "near_current_accuracy": near_current_accuracy,
        "serving_safe": True,
        "accepted_for_pricing": accepted_for_pricing,
    }
    write_json(pricing_report, REPORTS_DIR / "pricing_model_champion.json")

    promoted_demand = False
    demand_report_path = REPORTS_DIR / "model_champion.json"
    if incumbent_wape is None or best_pricing["score"] <= incumbent_wape:
        demand_champion_path = MODELS_DIR / "champion_demand_model.pkl"
        shutil.copy2(best_pricing_path, demand_champion_path)
        demand_report = {
            "champion": best_pricing,
            "candidates": sorted(evaluated, key=lambda row: row["score"]),
            "champion_model_path": str(demand_champion_path),
            "promotion_reason": "CatBoost v2 matched or beat incumbent demand WAPE.",
        }
        write_json(demand_report, demand_report_path)
        promoted_demand = True

    return {
        "pricing": pricing_report,
        "demand_promoted": promoted_demand,
        "incumbent_demand_wape": incumbent_wape,
    }


def train_catboost_v2_models(
    features: pd.DataFrame,
    iterations: int = 700,
    near_accuracy_tolerance: float = 0.10,
) -> dict[str, object]:
    frame, numeric_features, categorical_features = prepare_v2_frame(features)
    train, valid, test = time_based_split(frame)
    feature_columns = numeric_features + categorical_features

    results = [
        train_candidate(
            candidate=candidate,
            train=train,
            valid=valid,
            test=test,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            iterations=iterations,
        )
        for candidate in CANDIDATES
    ]
    write_json({"candidates": results}, REPORTS_DIR / "catboost_v2_candidates.json")
    selection = promote_v2_champions(results, incumbent_champion_wape(), near_accuracy_tolerance)
    return {"candidates": results, "selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost v2 pricing-safe demand models.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--near-accuracy-tolerance", type=float, default=0.10)
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    result = train_catboost_v2_models(
        features,
        iterations=args.iterations,
        near_accuracy_tolerance=args.near_accuracy_tolerance,
    )
    summary = {
        "pricing_champion": result["selection"]["pricing"]["champion"],
        "demand_promoted": result["selection"]["demand_promoted"],
        "near_current_accuracy": result["selection"]["pricing"]["near_current_accuracy"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
