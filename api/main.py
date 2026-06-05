from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.optimization.profit_optimizer import (
    load_demand_model,
    recommend_price,
    simulate_price_candidates,
)
from src.pricing.intelligence import build_recommendation_context


app = FastAPI(
    title="Retail Pricing Optimization API",
    description="Demand prediction and profit-aware price recommendation service.",
    version="0.1.0",
)


class PredictDemandRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)
    base_quantity: float = 100.0
    current_price: float | None = None
    candidate_price: float | None = None
    price_elasticity: float = -1.0


class RecommendPriceRequest(BaseModel):
    product_id: int | str
    store_id: int | str
    current_price: float
    unit_cost: float
    candidate_discounts: list[float] = Field(default_factory=lambda: [0, 5, 10, 15, 20])
    inventory_limit: float | None = None
    promotion_cost: float = 0.0
    base_quantity: float | None = None
    price_elasticity: float | None = None
    promotion_effect: float | None = None
    base_features: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def get_model():
    return load_demand_model()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "retail-pricing-optimization", "status": "ready"}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True, "model_loaded": get_model() is not None}


@app.post("/predict-demand")
def predict_demand(payload: PredictDemandRequest) -> dict[str, float | str]:
    model = get_model()
    if model is not None and payload.features:
        predicted_quantity = float(model.predict(pd.DataFrame([payload.features]))[0])
        return {"prediction_source": "baseline_model", "predicted_quantity": max(predicted_quantity, 0.0)}

    if payload.current_price and payload.candidate_price:
        ratio = max(payload.candidate_price / payload.current_price, 0.01)
        quantity = payload.base_quantity * (ratio**payload.price_elasticity)
    else:
        quantity = payload.base_quantity
    return {"prediction_source": "elasticity_fallback", "predicted_quantity": max(float(quantity), 0.0)}


@app.post("/recommend-price")
def recommend_price_endpoint(payload: RecommendPriceRequest) -> dict[str, Any]:
    context = build_recommendation_context(payload.product_id, payload.store_id)
    result = recommend_price(
        product_id=payload.product_id,
        store_id=payload.store_id,
        current_price=payload.current_price,
        estimated_unit_cost=payload.unit_cost,
        candidate_discounts=payload.candidate_discounts,
        demand_model=get_model(),
        base_features=payload.base_features or context["base_features"],
        base_quantity=payload.base_quantity if payload.base_quantity is not None else context["base_quantity"],
        price_elasticity=payload.price_elasticity if payload.price_elasticity is not None else context["price_elasticity"],
        promotion_effect=payload.promotion_effect if payload.promotion_effect is not None else context["promotion_effect"],
        inventory_limit=payload.inventory_limit,
        promotion_cost=payload.promotion_cost,
    )
    response = asdict(result)
    response["context_source"] = context["source"]
    response["commodity_desc"] = context["commodity_desc"]
    return response


@app.post("/simulate-prices")
def simulate_prices_endpoint(payload: RecommendPriceRequest) -> dict[str, Any]:
    context = build_recommendation_context(payload.product_id, payload.store_id)
    simulations = simulate_price_candidates(
        product_id=payload.product_id,
        store_id=payload.store_id,
        current_price=payload.current_price,
        estimated_unit_cost=payload.unit_cost,
        candidate_discounts=payload.candidate_discounts,
        demand_model=get_model(),
        base_features=payload.base_features or context["base_features"],
        base_quantity=payload.base_quantity if payload.base_quantity is not None else context["base_quantity"],
        price_elasticity=payload.price_elasticity if payload.price_elasticity is not None else context["price_elasticity"],
        promotion_effect=payload.promotion_effect if payload.promotion_effect is not None else context["promotion_effect"],
        inventory_limit=payload.inventory_limit,
        promotion_cost=payload.promotion_cost,
    )
    return {
        "context_source": context["source"],
        "commodity_desc": context["commodity_desc"],
        "price_elasticity": payload.price_elasticity if payload.price_elasticity is not None else context["price_elasticity"],
        "promotion_effect": payload.promotion_effect if payload.promotion_effect is not None else context["promotion_effect"],
        "simulations": [asdict(row) for row in simulations],
    }
