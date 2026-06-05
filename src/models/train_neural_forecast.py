from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.artifacts import default_features_path
from src.config.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs
from src.models.train_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, prepare_model_frame, time_based_split
from src.utils.io import write_json
from src.utils.metrics import regression_report


class DemandLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(-1)


def encode_categoricals(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    encoded = frame.copy()
    mappings: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_FEATURES:
        if column not in encoded:
            continue
        values = encoded[column].fillna("Unknown").astype(str)
        categories = {value: index for index, value in enumerate(sorted(values.unique()))}
        mappings[column] = categories
        encoded[f"{column}_code"] = values.map(categories).fillna(-1).astype(float)
    return encoded, mappings


def build_tensor_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], StandardScaler, dict[str, dict[str, int]]]:
    frame = prepare_model_frame(frame)
    frame, mappings = encode_categoricals(frame)
    feature_columns = [column for column in NUMERIC_FEATURES if column in frame]
    feature_columns += [f"{column}_code" for column in mappings]
    scaler = StandardScaler()
    frame[feature_columns] = scaler.fit_transform(frame[feature_columns].fillna(0))
    return frame, feature_columns, scaler, mappings


def make_loader(frame: pd.DataFrame, feature_columns: list[str], batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.tensor(frame[feature_columns].to_numpy(dtype=np.float32)).unsqueeze(1)
    y = torch.tensor(np.log1p(frame[TARGET].to_numpy(dtype=np.float32)))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, frame: pd.DataFrame, feature_columns: list[str], batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    loader = make_loader(frame, feature_columns, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x_batch, _ in loader:
            batch_pred = torch.expm1(model(x_batch)).clamp(min=0).cpu().numpy()
            preds.append(batch_pred)
    return np.concatenate(preds) if preds else np.array([])


def train_neural_model(
    features: pd.DataFrame,
    model_path: Path = MODELS_DIR / "neural_demand_model.pt",
    metrics_path: Path = REPORTS_DIR / "neural_metrics.json",
    epochs: int = 20,
    batch_size: int = 2048,
    max_rows: int | None = None,
) -> dict[str, object]:
    frame, feature_columns, scaler, mappings = build_tensor_frame(features)
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=42).sort_values(["week_no", "product_id", "store_id"])

    train, valid, test = time_based_split(frame)
    model = DemandLSTM(input_size=len(feature_columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    train_loader = make_loader(train, feature_columns, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    metrics: dict[str, object] = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "features": feature_columns,
        "epochs": epochs,
    }
    if len(valid):
        metrics["validation"] = regression_report(valid[TARGET], predict(model, valid, feature_columns))
    if len(test):
        metrics["test"] = regression_report(test[TARGET], predict(model, test, feature_columns))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": len(feature_columns),
            "feature_columns": feature_columns,
            "scaler": scaler,
            "category_mappings": mappings,
        },
        model_path,
    )
    write_json(metrics, metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train neural demand forecasting model.")
    parser.add_argument("--features-path", default=str(default_features_path()))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "neural_demand_model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=250_000)
    args = parser.parse_args()

    ensure_project_dirs()
    features = pd.read_parquet(args.features_path)
    metrics = train_neural_model(
        features,
        model_path=Path(args.model_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )
    print(metrics)


if __name__ == "__main__":
    main()
