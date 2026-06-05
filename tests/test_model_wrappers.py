import numpy as np
import pandas as pd

from src.models.model_wrappers import PredictionBinCalibratedModel


class ConstantRampModel:
    def predict(self, X):
        return np.array([0.5, 1.0, 3.0])


def test_prediction_bin_calibrated_model_scales_by_prediction_bin():
    model = PredictionBinCalibratedModel(
        base_model=ConstantRampModel(),
        bin_edges=[-np.inf, 0.75, 1.25, np.inf],
        scale_factors=[1.4, 1.0, 0.8],
    )

    predictions = model.predict(pd.DataFrame({"x": [1, 2, 3]}))

    assert predictions.tolist() == [0.7, 1.0, 2.4000000000000004]
