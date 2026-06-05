from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.serving_features import augment_pricing_features


@dataclass
class CatBoostDemandModel:
    model: object
    feature_columns: list[str]
    categorical_features: list[str]
    feature_augmenter: str | None = None

    def predict(self, X):
        frame = pd.DataFrame(X).copy()
        if getattr(self, "feature_augmenter", None) == "pricing_v2":
            frame = augment_pricing_features(frame)
        for column in self.feature_columns:
            if column not in frame:
                frame[column] = "Unknown" if column in self.categorical_features else 0
        frame = frame[self.feature_columns]
        for column in self.categorical_features:
            frame[column] = frame[column].fillna("Unknown").astype(str)
        for column in self.feature_columns:
            if column not in self.categorical_features:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return self.model.predict(frame)


@dataclass
class PredictionBinCalibratedModel:
    base_model: object
    bin_edges: list[float]
    scale_factors: list[float]

    def predict(self, X):
        predictions = pd.Series(self.base_model.predict(X), dtype=float).clip(lower=0)
        bin_indexes = np.digitize(predictions.to_numpy(), self.bin_edges[1:-1], right=False)
        scales = np.asarray(self.scale_factors, dtype=float)
        calibrated = predictions.to_numpy() * scales[bin_indexes]
        return np.clip(calibrated, 0, None)
