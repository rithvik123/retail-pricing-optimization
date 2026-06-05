import numpy as np
import pandas as pd

from src.optimization.profit_optimizer import (
    candidate_discounts_from_price_history,
    normalize_discounts,
    recommend_price,
    simulate_price_candidates,
)


class CapturingModel:
    def __init__(self):
        self.rows = []

    def predict(self, X):
        frame = pd.DataFrame(X).copy()
        self.rows.append(frame)
        return np.array([100.0])


class SmallLiftModel:
    def predict(self, X):
        frame = pd.DataFrame(X)
        price = float(frame["avg_unit_price"].iloc[0])
        if price < 10:
            return np.array([127.0])
        return np.array([100.0])


def test_normalize_discounts_accepts_markups_as_negative_percentages():
    assert normalize_discounts([-10, -5, 0, 5, 10]) == [-0.1, -0.05, 0.0, 0.05, 0.1]


def test_candidate_discounts_from_history_only_keeps_supported_price_changes():
    candidates = candidate_discounts_from_price_history(
        current_price=10,
        price_history=pd.DataFrame({"avg_unit_price": [9.5, 10, 10.5, 11]}),
        min_price_change=-0.2,
        max_price_change=0.2,
        step=0.05,
    )

    assert 0.0 in candidates
    assert -0.05 in candidates
    assert 0.05 in candidates
    assert -0.2 not in candidates
    assert 0.2 not in candidates


def test_candidate_discounts_can_be_limited_to_discount_only_mode():
    candidates = candidate_discounts_from_price_history(
        current_price=10,
        price_history=pd.DataFrame({"avg_unit_price": [8, 9, 10, 11]}),
        min_price_change=-0.2,
        max_price_change=0,
        step=0.05,
    )

    assert 0.0 in candidates
    assert 0.1 in candidates
    assert all(candidate >= 0 for candidate in candidates)


def test_profit_optimizer_respects_margin_constraints():
    simulations = simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=7,
        candidate_discounts=[0, 10, 30],
        base_quantity=100,
        price_elasticity=-1.0,
    )

    assert simulations[0].feasible is True
    assert simulations[-1].feasible is False
    assert "margin_below_minimum" in simulations[-1].constraint_message


def test_recommend_price_returns_best_feasible_result():
    result = recommend_price(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=4,
        candidate_discounts=[0, 10, 20],
        base_quantity=100,
        price_elasticity=-0.8,
    )

    assert result.recommended_discount in {0, 10, 20}
    assert result.expected_profit > 0
    assert result.business_reason


def test_model_scenario_rows_recompute_price_derived_features():
    model = CapturingModel()

    simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=4,
        candidate_discounts=[10],
        demand_model=model,
        base_features={
            "avg_unit_price": 10,
            "median_unit_price": 10,
            "price_lag_1": 12,
            "discount_percentage": 0,
            "is_display": 1,
            "is_mailer": 0,
            "lag_quantity_1": 8,
            "rolling_quantity_mean_4": 4,
            "rolling_quantity_mean_8": 8,
        },
    )

    row = model.rows[-1]
    assert row["avg_unit_price"].iloc[0] == 9
    assert row["price_change"].iloc[0] == -3
    assert row["price_ratio_lag_1"].iloc[0] == 0.75
    assert row["discount_x_display"].iloc[0] == 0.1


def test_model_predictions_are_blended_with_elasticity_response():
    model = CapturingModel()

    simulations = simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=4,
        candidate_discounts=[-10],
        demand_model=model,
        base_features={
            "avg_unit_price": 10,
            "median_unit_price": 10,
            "price_lag_1": 10,
            "discount_percentage": 0,
        },
        price_elasticity=-1.0,
        price_response_weight=1.0,
        enforce_observed_price_range=False,
    )

    assert simulations[0].predicted_quantity < 100


def test_simulation_outputs_economic_break_even_metrics():
    simulations = simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=5,
        candidate_discounts=[0, 10],
        base_quantity=100,
        price_elasticity=-1.0,
        promotion_cost=10,
        enforce_observed_price_range=False,
    )

    discount_row = [row for row in simulations if row.discount == 10][0]
    assert discount_row.contribution_margin == 4
    assert discount_row.break_even_quantity > 100
    assert discount_row.required_demand_lift > 0
    assert discount_row.incremental_profit < 0


def test_low_confidence_positive_profit_is_risk_adjusted():
    simulations = simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=4,
        candidate_discounts=[0, -10],
        base_quantity=100,
        price_elasticity=-0.2,
        price_history=pd.DataFrame({"week_no": [1, 2], "avg_unit_price": [10, 11]}),
        enforce_observed_price_range=True,
    )

    increase_row = [row for row in simulations if row.discount == -10][0]
    baseline_row = [row for row in simulations if row.discount == 0][0]
    assert increase_row.confidence == "low"
    assert increase_row.expected_profit > baseline_row.expected_profit
    assert increase_row.risk_adjusted_profit < increase_row.expected_profit


def test_recommend_price_prefers_no_change_below_profit_lift_threshold():
    result = recommend_price(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=5,
        candidate_discounts=[0, 10],
        demand_model=SmallLiftModel(),
        base_features={
            "avg_unit_price": 10,
            "median_unit_price": 10,
            "price_lag_1": 10,
            "discount_percentage": 0,
        },
        profit_lift_threshold=0.02,
        price_response_weight=0.0,
    )

    assert result.recommended_discount == 0
    assert "decision buffer" in result.business_reason


def test_price_history_guardrail_flags_out_of_range_candidates():
    simulations = simulate_price_candidates(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=4,
        candidate_discounts=[30],
        base_quantity=100,
        price_elasticity=-1.0,
        price_history=pd.DataFrame({"week_no": list(range(1, 13)), "avg_unit_price": [9.5, 10, 10.5] * 4}),
    )

    assert simulations[0].confidence == "low"
    assert "outside_observed_price_range" in simulations[0].guardrail_message
    assert simulations[0].feasible is False
    assert "outside_observed_price_range" in simulations[0].constraint_message


def test_recommend_price_does_not_select_out_of_range_boundary_move():
    result = recommend_price(
        product_id=100,
        store_id=20,
        current_price=10,
        estimated_unit_cost=5,
        candidate_discounts=[0, -30],
        demand_model=SmallLiftModel(),
        base_features={
            "avg_unit_price": 10,
            "median_unit_price": 10,
            "price_lag_1": 10,
            "discount_percentage": 0,
        },
        price_history=pd.DataFrame({"week_no": list(range(1, 13)), "avg_unit_price": [9.5, 10, 10.5] * 4}),
    )

    assert result.recommended_discount == 0
