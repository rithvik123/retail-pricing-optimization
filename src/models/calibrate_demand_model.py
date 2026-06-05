from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config.artifacts import default_features_path
from src.config.paths import MODELS_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.model_wrappers import PredictionBinCalibratedModel
from src.models.train_baseline import TARGET, prepare_model_frame, time_based_split
from src.models.train_catboost_v2 import quantity_bin_report, segment_wape_report
from src.utils.io import write_json
from src.utils.metrics import regression_report


DEFAULT_BIN_EDGES = [-np.inf, 0.75, 1.25, 1.75, 2.5, 4.0, 8.0, np.inf]


def _json_safe_edges(edges: list[float]) -> list[str | float]:
    values: list[str | float] = []
    for edge in edges:
        if np.isneginf(edge):
            values.append("-inf")
        elif np.isposinf(edge):
            values.append("inf")
        else:
            values.append(float(edge))
    return values


def fit_prediction_bin_scales(
    actual: pd.Series,
    predicted: pd.Series,
    bin_edges: list[float],
    min_rows: int = 100,
    min_scale: float = 0.7,
    max_scale: float = 1.4,
) -> list[float]:
    bin_indexes = np.digitize(predicted.to_numpy(), bin_edges[1:-1], right=False)
    scales = []
    for bin_index in range(len(bin_edges) - 1):
        mask = bin_indexes == bin_index
        if int(mask.sum()) < min_rows:
            scales.append(1.0)
            continue
        predicted_units = float(predicted.iloc[mask].sum())
        actual_units = float(actual.iloc[mask].sum())
        scale = actual_units / predicted_units if predicted_units > 0 else 1.0
        scales.append(float(np.clip(scale, min_scale, max_scale)))
    return scales


def evaluate_split(split: pd.DataFrame, predicted: pd.Series) -> dict[str, object]:
    actual = split[TARGET]
    return {
        **regression_report(actual, predicted),
        "quantity_bins": quantity_bin_report(actual, predicted),
        "departments_by_wape": segment_wape_report(split, actual, predicted, "department"),
        "categories_by_wape": segment_wape_report(split, actual, predicted, "commodity_desc"),
    }


def incumbent_champion() -> tuple[float | None, dict[str, object]]:
    report_path = REPORTS_DIR / "model_champion.json"
    if not report_path.exists():
        return None, {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    score = report.get("champion", {}).get("score")
    return (float(score) if score is not None else None), report


def calibrate_demand_model(
    features: pd.DataFrame,
    base_model_path: Path = MODELS_DIR / "catboost_demand_model.pkl",
    calibrated_model_path: Path = MODELS_DIR / "catboost_calibrated_demand_model.pkl",
    metrics_path: Path = REPORTS_DIR / "catboost_calibrated_metrics.json",
    promote_if_better: bool = True,
) -> dict[str, object]:
    frame = prepare_model_frame(features)
    _, valid, test = time_based_split(frame)
    base_model = joblib.load(base_model_path)

    valid_pred = pd.Series(base_model.predict(valid), index=valid.index).clip(lower=0)
    test_pred = pd.Series(base_model.predict(test), index=test.index).clip(lower=0)
    scales = fit_prediction_bin_scales(valid[TARGET], valid_pred, DEFAULT_BIN_EDGES)
    calibrated = PredictionBinCalibratedModel(base_model=base_model, bin_edges=DEFAULT_BIN_EDGES, scale_factors=scales)
    valid_calibrated = pd.Series(calibrated.predict(valid), index=valid.index).clip(lower=0)
    test_calibrated = pd.Series(calibrated.predict(test), index=test.index).clip(lower=0)

    metrics: dict[str, object] = {
        "model_type": "Prediction-bin calibrated CatBoost demand model",
        "base_model_path": str(base_model_path),
        "train_rows": int(len(frame) - len(valid) - len(test)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "bin_edges": _json_safe_edges(DEFAULT_BIN_EDGES),
        "scale_factors": scales,
        "base_validation": regression_report(valid[TARGET], valid_pred),
        "base_test": regression_report(test[TARGET], test_pred),
        "validation": evaluate_split(valid, valid_calibrated),
        "test": evaluate_split(test, test_calibrated),
    }

    calibrated_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, calibrated_model_path)
    write_json(metrics, metrics_path)

    incumbent_wape, incumbent_report = incumbent_champion()
    promoted = False
    test_wape = float(metrics["test"]["wape"])
    if promote_if_better and (incumbent_wape is None or test_wape <= incumbent_wape):
        champion_model_path = MODELS_DIR / "champion_demand_model.pkl"
        shutil.copy2(calibrated_model_path, champion_model_path)
        champion = {
            "name": "catboost_calibrated_regression",
            "model_path": str(calibrated_model_path),
            "metrics_path": str(metrics_path),
            "selection_metric": "wape",
            "selection_split": "test",
            "score": test_wape,
            "test_metrics": {key: metrics["test"][key] for key in ("mae", "rmse", "smape", "wape")},
            "validation_metrics": {key: metrics["validation"][key] for key in ("mae", "rmse", "smape", "wape")},
            "calibrated_from": str(base_model_path),
        }
        prior_candidates = incumbent_report.get("candidates", []) if incumbent_report else []
        report = {
            "champion": champion,
            "candidates": [champion] + prior_candidates,
            "champion_model_path": str(champion_model_path),
            "promotion_reason": "Calibration improved or matched incumbent demand WAPE.",
        }
        write_json(report, REPORTS_DIR / "model_champion.json")
        promoted = True

    metrics["promoted_to_demand_champion"] = promoted
    write_json(metrics, metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the CatBoost demand champion by prediction bins.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--base-model-path", default=str(MODELS_DIR / "catboost_demand_model.pkl"))
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    metrics = calibrate_demand_model(
        features,
        base_model_path=Path(args.base_model_path),
        promote_if_better=not args.no_promote,
    )
    print(json.dumps({key: metrics[key] for key in ("base_test", "test", "promoted_to_demand_champion")}, indent=2))


if __name__ == "__main__":
    main()
