from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.config.paths import MODELS_DIR, REPORTS_DIR, ensure_project_dirs


MODEL_CANDIDATES = {
    "baseline_lightgbm_regression": {
        "model_path": MODELS_DIR / "baseline_demand_model.pkl",
        "metrics_path": REPORTS_DIR / "baseline_metrics.json",
    },
    "production_lightgbm_poisson": {
        "model_path": MODELS_DIR / "production_demand_model.pkl",
        "metrics_path": REPORTS_DIR / "production_metrics.json",
    },
    "catboost_regression": {
        "model_path": MODELS_DIR / "catboost_demand_model.pkl",
        "metrics_path": REPORTS_DIR / "catboost_metrics.json",
    },
    "catboost_calibrated_regression": {
        "model_path": MODELS_DIR / "catboost_calibrated_demand_model.pkl",
        "metrics_path": REPORTS_DIR / "catboost_calibrated_metrics.json",
    },
    "catboost_v2_rmse": {
        "model_path": MODELS_DIR / "catboost_v2_rmse_model.pkl",
        "metrics_path": REPORTS_DIR / "catboost_v2_rmse_metrics.json",
    },
    "catboost_v2_weighted_mae": {
        "model_path": MODELS_DIR / "catboost_v2_weighted_mae_model.pkl",
        "metrics_path": REPORTS_DIR / "catboost_v2_weighted_mae_metrics.json",
    },
    "catboost_v2_tweedie": {
        "model_path": MODELS_DIR / "catboost_v2_tweedie_model.pkl",
        "metrics_path": REPORTS_DIR / "catboost_v2_tweedie_metrics.json",
    },
}


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_champion(
    candidates: dict[str, dict[str, Path]] = MODEL_CANDIDATES,
    metric_name: str = "wape",
    split: str = "test",
    champion_model_path: Path = MODELS_DIR / "champion_demand_model.pkl",
    champion_report_path: Path = REPORTS_DIR / "model_champion.json",
) -> dict[str, object]:
    evaluated: list[dict[str, object]] = []
    for name, candidate in candidates.items():
        model_path = candidate["model_path"]
        metrics_path = candidate["metrics_path"]
        if not model_path.exists() or not metrics_path.exists():
            continue
        metrics = _load_metrics(metrics_path)
        score = metrics.get(split, {}).get(metric_name)
        if score is None:
            continue
        evaluated.append(
            {
                "name": name,
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
                "selection_metric": metric_name,
                "selection_split": split,
                "score": float(score),
                "test_metrics": metrics.get("test", {}),
                "validation_metrics": metrics.get("validation", {}),
            }
        )

    if not evaluated:
        raise FileNotFoundError("No model candidates with usable metrics were found.")

    champion = min(evaluated, key=lambda row: row["score"])
    shutil.copy2(champion["model_path"], champion_model_path)
    report = {
        "champion": champion,
        "candidates": sorted(evaluated, key=lambda row: row["score"]),
        "champion_model_path": str(champion_model_path),
    }
    champion_report_path.parent.mkdir(parents=True, exist_ok=True)
    champion_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best demand model candidate.")
    parser.add_argument("--metric", default="wape")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    ensure_project_dirs()
    report = select_champion(metric_name=args.metric, split=args.split)
    print(json.dumps(report["champion"], indent=2))


if __name__ == "__main__":
    main()
