import pandas as pd

from src.models.serving_features import (
    LEAKAGE_FEATURES,
    PRICING_NUMERIC_FEATURES,
    add_serving_history_features,
    augment_pricing_features,
    prepare_pricing_scenario_features,
)


def test_pricing_features_exclude_outcome_leakage_fields():
    assert set(PRICING_NUMERIC_FEATURES).isdisjoint(LEAKAGE_FEATURES)


def test_augment_pricing_features_adds_scenario_safe_interactions():
    frame = pd.DataFrame(
        [
            {
                "avg_unit_price": 8.0,
                "price_lag_1": 10.0,
                "discount_percentage": 0.2,
                "is_display": 1,
                "is_mailer": 0,
                "lag_quantity_1": 12,
                "rolling_quantity_mean_4": 6,
                "rolling_quantity_mean_8": 4,
            }
        ]
    )

    result = augment_pricing_features(frame)

    assert result["price_ratio_lag_1"].iloc[0] == 0.8
    assert result["discount_x_display"].iloc[0] == 0.2
    assert result["discount_x_mailer"].iloc[0] == 0
    assert result["any_promotion"].iloc[0] == 1
    assert result["lag_velocity_4"].iloc[0] == 2
    assert result["lag_velocity_8"].iloc[0] == 3


def test_prepare_pricing_scenario_features_updates_candidate_price_fields():
    base = {
        "avg_unit_price": 10.0,
        "median_unit_price": 10.0,
        "price_lag_1": 12.0,
        "discount_percentage": 0,
        "is_display": 1,
        "is_mailer": 0,
        "lag_quantity_1": 8,
        "rolling_quantity_mean_4": 4,
        "rolling_quantity_mean_8": 8,
    }

    row = prepare_pricing_scenario_features(base, candidate_price=9.0, discount_rate=0.1)

    assert row["avg_unit_price"].iloc[0] == 9.0
    assert row["median_unit_price"].iloc[0] == 9.0
    assert row["discount_percentage"].iloc[0] == 0.1
    assert row["price_change"].iloc[0] == -3.0
    assert row["price_ratio_lag_1"].iloc[0] == 0.75
    assert row["discount_x_display"].iloc[0] == 0.1


def test_add_serving_history_features_uses_previous_product_store_weeks():
    frame = pd.DataFrame(
        [
            {"product_id": 1, "store_id": 10, "week_no": 1, "num_baskets": 5, "num_households": 4},
            {"product_id": 1, "store_id": 10, "week_no": 2, "num_baskets": 7, "num_households": 6},
            {"product_id": 2, "store_id": 10, "week_no": 1, "num_baskets": 100, "num_households": 90},
        ]
    )

    result = add_serving_history_features(frame)

    first_product_second_week = result[(result["product_id"] == 1) & (result["week_no"] == 2)].iloc[0]
    second_product_first_week = result[(result["product_id"] == 2) & (result["week_no"] == 1)].iloc[0]
    assert first_product_second_week["lag_num_baskets_1"] == 5
    assert first_product_second_week["rolling_num_households_mean_4"] == 4
    assert second_product_first_week["lag_num_baskets_1"] == 0
