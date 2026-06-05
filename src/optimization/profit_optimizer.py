from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from src.config.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.models.serving_features import prepare_pricing_scenario_features


MODEL_CATEGORICAL_COLUMNS = ("department", "commodity_desc", "brand", "store_id", "product_id")


@dataclass(frozen=True)
class SimulationResult:
    product_id: int | str
    store_id: int | str
    discount: float
    candidate_price: float
    predicted_quantity: float
    expected_revenue: float
    expected_profit: float
    margin_percentage: float
    feasible: bool
    constraint_message: str
    confidence: str = "high"
    guardrail_message: str = "ok"
    price_change_percentage: float = 0.0
    contribution_margin: float = 0.0
    demand_lift: float = 0.0
    revenue_lift: float = 0.0
    profit_lift: float = 0.0
    incremental_profit: float = 0.0
    break_even_quantity: float = 0.0
    required_demand_lift: float = 0.0
    risk_adjusted_profit: float = 0.0


@dataclass(frozen=True)
class RecommendationResult:
    recommended_price: float
    recommended_discount: float
    predicted_quantity: float
    expected_revenue: float
    expected_profit: float
    price_elasticity: float
    promotion_effect: float
    business_reason: str
    simulations: list[SimulationResult]
    baseline_profit: float = 0.0
    profit_lift: float = 0.0
    confidence: str = "high"
    guardrail_message: str = "ok"


def normalize_discounts(candidate_discounts: Iterable[float] | None) -> list[float]:
    values = list(candidate_discounts if candidate_discounts is not None else [0, 5, 10, 15, 20, 25, 30])
    normalized = []
    for value in values:
        rate = float(value)
        if abs(rate) > 1:
            rate = rate / 100
        normalized.append(rate)
    return sorted(set(normalized))


def format_price_action(discount_rate: float) -> str:
    if abs(discount_rate) < 1e-9:
        return "no price change"
    if discount_rate < 0:
        return f"{abs(discount_rate) * 100:.0f}% price increase"
    return f"{discount_rate * 100:.0f}% discount"


def _bounded_price_elasticity(price_elasticity: float) -> float:
    if not math.isfinite(float(price_elasticity)):
        return -1.0
    value = float(price_elasticity)
    if value >= -0.05:
        return -0.5
    return max(min(value, -0.05), -3.0)


def observed_price_bounds(price_history: pd.DataFrame | pd.Series | None) -> tuple[float | None, float | None]:
    if price_history is None:
        return None, None
    history = pd.DataFrame(price_history).copy()
    if history.empty or "avg_unit_price" not in history:
        return None, None
    prices = pd.to_numeric(history["avg_unit_price"], errors="coerce")
    prices = prices[prices > 0].dropna()
    if prices.empty:
        return None, None
    return float(prices.min()), float(prices.max())


def candidate_discounts_from_price_history(
    current_price: float,
    price_history: pd.DataFrame | pd.Series | None,
    min_price_change: float = -0.15,
    max_price_change: float = 0.15,
    step: float = 0.05,
) -> list[float]:
    """Build candidate discount rates from price changes supported by observed history."""
    if current_price <= 0:
        return [0.0]
    lower, upper = observed_price_bounds(price_history)
    changes: list[float] = []
    steps = int(round((max_price_change - min_price_change) / step))
    for index in range(steps + 1):
        change = round(min_price_change + index * step, 10)
        candidate_price = current_price * (1 + change)
        is_current = abs(change) < 1e-9
        is_supported = lower is None or upper is None or lower <= candidate_price <= upper
        if is_current or is_supported:
            changes.append(change)
    if 0.0 not in changes:
        changes.append(0.0)
    # Discount convention: positive means price cut, negative means price increase.
    return sorted({round(-change, 10) for change in changes})


def load_demand_model(model_path: str | Path = MODELS_DIR / "baseline_demand_model.pkl"):
    requested_path = Path(model_path)
    if requested_path == MODELS_DIR / "baseline_demand_model.pkl":
        for candidate in (
            MODELS_DIR / "champion_demand_model.pkl",
            MODELS_DIR / "baseline_demand_model.pkl",
            MODELS_DIR / "production_demand_model.pkl",
        ):
            if candidate.exists():
                return joblib.load(candidate)
    path = requested_path
    if not path.exists():
        return None
    return joblib.load(path)


def load_pricing_model(model_path: str | Path = MODELS_DIR / "champion_pricing_model.pkl"):
    requested_path = Path(model_path)
    if requested_path == MODELS_DIR / "champion_pricing_model.pkl":
        report_path = REPORTS_DIR / "pricing_model_champion.json"
        accepted_pricing_model = True
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            accepted_pricing_model = bool(
                report.get("accepted_for_pricing", report.get("serving_safe") and report.get("near_current_accuracy"))
            )
        candidates = []
        if accepted_pricing_model:
            candidates.append(MODELS_DIR / "champion_pricing_model.pkl")
        candidates.extend(
            [
                MODELS_DIR / "champion_demand_model.pkl",
                MODELS_DIR / "baseline_demand_model.pkl",
                MODELS_DIR / "production_demand_model.pkl",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return joblib.load(candidate)
    if not requested_path.exists():
        return None
    return joblib.load(requested_path)


def _model_predict_quantity(model, base_features: dict | pd.Series | pd.DataFrame, candidate_price: float, discount_rate: float) -> float:
    row = prepare_pricing_scenario_features(base_features, candidate_price, discount_rate)
    for column in MODEL_CATEGORICAL_COLUMNS:
        if column in row:
            row[column] = row[column].where(row[column].notna(), "Unknown").astype(str)

    prediction = float(model.predict(row)[0])
    return max(prediction, 0.0)


def _elasticity_predict_quantity(
    current_price: float,
    candidate_price: float,
    base_quantity: float,
    price_elasticity: float,
    promotion_effect: float,
    discount_rate: float,
) -> float:
    if current_price <= 0:
        return max(base_quantity, 0.0)
    price_ratio = max(candidate_price / current_price, 0.01)
    discount_uplift = 1 + promotion_effect * discount_rate
    return max(base_quantity * (price_ratio**price_elasticity) * discount_uplift, 0.0)


def _confidence_multiplier(confidence: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.9,
        "low": 0.75,
    }.get(str(confidence).lower(), 0.85)


def _risk_adjusted_profit(expected_profit: float, baseline_profit: float, confidence: str) -> float:
    incremental_profit = expected_profit - baseline_profit
    if incremental_profit <= 0:
        return expected_profit
    return baseline_profit + incremental_profit * _confidence_multiplier(confidence)


def _scenario_guardrails(
    candidate_price: float,
    price_history: pd.DataFrame | pd.Series | None,
    min_history_weeks: int,
) -> tuple[str, str]:
    if price_history is None:
        return "high", "ok"

    history = pd.DataFrame(price_history).copy()
    if history.empty or "avg_unit_price" not in history:
        return "medium", "missing_price_history"

    messages = []
    if "week_no" in history:
        history_weeks = int(history["week_no"].nunique())
    else:
        history_weeks = int(len(history))
    if history_weeks < min_history_weeks:
        messages.append("sparse_history")

    prices = pd.to_numeric(history["avg_unit_price"], errors="coerce")
    prices = prices[prices > 0].dropna()
    if len(prices) >= 4:
        lower = float(prices.quantile(0.05))
        upper = float(prices.quantile(0.95))
        if candidate_price < lower or candidate_price > upper:
            messages.append("outside_observed_price_range")
    elif len(prices):
        lower = float(prices.min())
        upper = float(prices.max())
        if candidate_price < lower or candidate_price > upper:
            messages.append("outside_observed_price_range")

    if not messages:
        return "high", "ok"
    if "outside_observed_price_range" in messages or "sparse_history" in messages:
        return "low", ";".join(messages)
    return "medium", ";".join(messages)


def simulate_price_candidates(
    product_id: int | str,
    store_id: int | str,
    current_price: float,
    estimated_unit_cost: float,
    candidate_discounts: Iterable[float] | None = None,
    demand_model=None,
    base_features: dict | pd.Series | pd.DataFrame | None = None,
    base_quantity: float = 100.0,
    price_elasticity: float = -1.0,
    promotion_effect: float = 0.0,
    inventory_limit: float | None = None,
    promotion_cost: float = 0.0,
    min_margin_percentage: float = 0.15,
    price_history: pd.DataFrame | pd.Series | None = None,
    min_history_weeks: int = 8,
    enforce_observed_price_range: bool = True,
    price_response_weight: float = 0.65,
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    baseline_model_quantity = None
    if demand_model is not None and base_features is not None:
        baseline_model_quantity = _model_predict_quantity(demand_model, base_features, float(current_price), 0.0)
        baseline_quantity = baseline_model_quantity
    else:
        baseline_quantity = base_quantity
    baseline_revenue = float(current_price) * max(float(baseline_quantity), 0.0)
    baseline_profit = (float(current_price) - float(estimated_unit_cost)) * max(float(baseline_quantity), 0.0)
    for discount_rate in normalize_discounts(candidate_discounts):
        candidate_price = float(current_price) * (1 - discount_rate)
        if demand_model is not None and base_features is not None:
            model_quantity = _model_predict_quantity(demand_model, base_features, candidate_price, discount_rate)
            response_quantity = _elasticity_predict_quantity(
                current_price=current_price,
                candidate_price=candidate_price,
                base_quantity=baseline_model_quantity if baseline_model_quantity is not None else base_quantity,
                price_elasticity=_bounded_price_elasticity(price_elasticity),
                promotion_effect=promotion_effect,
                discount_rate=discount_rate,
            )
            response_weight = min(max(float(price_response_weight), 0.0), 1.0)
            predicted_quantity = (1 - response_weight) * model_quantity + response_weight * response_quantity
        else:
            predicted_quantity = _elasticity_predict_quantity(
                current_price=current_price,
                candidate_price=candidate_price,
                base_quantity=base_quantity,
                price_elasticity=_bounded_price_elasticity(price_elasticity),
                promotion_effect=promotion_effect,
                discount_rate=discount_rate,
            )

        candidate_promotion_cost = float(promotion_cost) if discount_rate > 0 else 0.0
        expected_revenue = candidate_price * predicted_quantity
        expected_profit = (candidate_price - estimated_unit_cost) * predicted_quantity - candidate_promotion_cost
        margin_percentage = 0.0 if candidate_price <= 0 else (candidate_price - estimated_unit_cost) / candidate_price
        contribution_margin = candidate_price - estimated_unit_cost

        messages = []
        if candidate_price <= estimated_unit_cost:
            messages.append("price_not_above_unit_cost")
        if margin_percentage < min_margin_percentage:
            messages.append("margin_below_minimum")
        if discount_rate > 0.30:
            messages.append("discount_above_policy_limit")
        if inventory_limit is not None and predicted_quantity > inventory_limit:
            messages.append("predicted_quantity_exceeds_inventory")
        confidence, guardrail_message = _scenario_guardrails(
            candidate_price=candidate_price,
            price_history=price_history,
            min_history_weeks=min_history_weeks,
        )
        if enforce_observed_price_range and "outside_observed_price_range" in guardrail_message:
            messages.append("outside_observed_price_range")
        incremental_profit = expected_profit - baseline_profit
        risk_adjusted_profit = _risk_adjusted_profit(expected_profit, baseline_profit, confidence)
        if contribution_margin > 0:
            break_even_quantity = (baseline_profit + candidate_promotion_cost) / contribution_margin
        else:
            break_even_quantity = float("inf")
        if baseline_quantity > 0 and math.isfinite(break_even_quantity):
            required_demand_lift = break_even_quantity / baseline_quantity - 1
        else:
            required_demand_lift = 0.0
        demand_lift = predicted_quantity / max(float(baseline_quantity), 0.0001) - 1
        revenue_lift = expected_revenue / max(baseline_revenue, 0.0001) - 1
        profit_lift = incremental_profit / max(abs(baseline_profit), 0.0001)

        results.append(
            SimulationResult(
                product_id=product_id,
                store_id=store_id,
                discount=round(discount_rate * 100, 2),
                candidate_price=round(candidate_price, 4),
                predicted_quantity=round(predicted_quantity, 4),
                expected_revenue=round(expected_revenue, 4),
                expected_profit=round(expected_profit, 4),
                margin_percentage=round(margin_percentage, 4),
                feasible=not messages,
                constraint_message=";".join(messages) if messages else "ok",
                confidence=confidence,
                guardrail_message=guardrail_message,
                price_change_percentage=round(-discount_rate * 100, 2),
                contribution_margin=round(contribution_margin, 4),
                demand_lift=round(demand_lift, 4),
                revenue_lift=round(revenue_lift, 4),
                profit_lift=round(profit_lift, 4),
                incremental_profit=round(incremental_profit, 4),
                break_even_quantity=round(break_even_quantity, 4) if math.isfinite(break_even_quantity) else float("inf"),
                required_demand_lift=round(required_demand_lift, 4),
                risk_adjusted_profit=round(risk_adjusted_profit, 4),
            )
        )
    return results


def recommend_price(
    product_id: int | str,
    store_id: int | str,
    current_price: float,
    estimated_unit_cost: float,
    candidate_discounts: Iterable[float] | None = None,
    demand_model=None,
    base_features: dict | pd.Series | pd.DataFrame | None = None,
    base_quantity: float = 100.0,
    price_elasticity: float = -1.0,
    promotion_effect: float = 0.0,
    inventory_limit: float | None = None,
    promotion_cost: float = 0.0,
    min_margin_percentage: float = 0.15,
    price_history: pd.DataFrame | pd.Series | None = None,
    min_history_weeks: int = 8,
    profit_lift_threshold: float = 0.02,
    enforce_observed_price_range: bool = True,
    price_response_weight: float = 0.65,
) -> RecommendationResult:
    simulations = simulate_price_candidates(
        product_id=product_id,
        store_id=store_id,
        current_price=current_price,
        estimated_unit_cost=estimated_unit_cost,
        candidate_discounts=candidate_discounts,
        demand_model=demand_model,
        base_features=base_features,
        base_quantity=base_quantity,
        price_elasticity=price_elasticity,
        promotion_effect=promotion_effect,
        inventory_limit=inventory_limit,
        promotion_cost=promotion_cost,
        min_margin_percentage=min_margin_percentage,
        price_history=price_history,
        min_history_weeks=min_history_weeks,
        enforce_observed_price_range=enforce_observed_price_range,
        price_response_weight=price_response_weight,
    )
    feasible = [row for row in simulations if row.feasible]
    candidates = feasible if feasible else simulations
    best = max(candidates, key=lambda row: row.risk_adjusted_profit)
    baseline_candidates = [row for row in candidates if abs(row.discount) < 1e-9]
    baseline = baseline_candidates[0] if baseline_candidates else None
    threshold_applied = False
    if baseline is not None and abs(best.discount) > 1e-9:
        candidate_lift = (best.risk_adjusted_profit - baseline.risk_adjusted_profit) / max(
            abs(baseline.risk_adjusted_profit), 0.0001
        )
        if candidate_lift < profit_lift_threshold:
            best = baseline
            threshold_applied = True

    action = format_price_action(best.discount / 100)
    baseline_profit = baseline.expected_profit if baseline is not None else best.expected_profit
    profit_lift = (best.expected_profit - baseline_profit) / max(abs(baseline_profit), 0.0001)
    if threshold_applied:
        reason = (
            "No change is preferred because the best alternative profit lift is below "
            f"the {profit_lift_threshold:.0%} decision buffer."
        )
    elif feasible:
        reason = (
            f"{action.capitalize()} maximizes expected profit while keeping "
            f"margin above {min_margin_percentage:.0%}."
        )
    else:
        reason = (
            "No candidate satisfies all constraints; showing the highest-profit option "
            "so the pricing team can review policy trade-offs."
        )

    return RecommendationResult(
        recommended_price=best.candidate_price,
        recommended_discount=best.discount,
        predicted_quantity=best.predicted_quantity,
        expected_revenue=best.expected_revenue,
        expected_profit=best.expected_profit,
        price_elasticity=price_elasticity,
        promotion_effect=promotion_effect,
        business_reason=reason,
        simulations=simulations,
        baseline_profit=baseline_profit,
        profit_lift=profit_lift,
        confidence=best.confidence,
        guardrail_message=best.guardrail_message,
    )


def save_simulation_results(results: list[SimulationResult], output_path: str | Path = PROCESSED_DIR / "price_simulation_results.parquet") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in results]).to_parquet(path, index=False)
