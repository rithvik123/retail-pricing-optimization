import pandas as pd

from src.features.filter_modeling_ready import build_modeling_ready_features


def test_modeling_ready_filter_removes_fuel_coupon_and_extreme_rows():
    features = pd.DataFrame(
        {
            "department": ["GROCERY", "KIOSK-GAS", "MISC SALES TRAN", "GROCERY"],
            "commodity_desc": ["BREAD", "COUPON/MISC ITEMS", "COUPON/MISC ITEMS", "SOFT DRINKS"],
            "avg_unit_price": [2.5, 0.002, 0.002, 3.0],
            "discount_percentage": [0.1, 0.0, 0.0, 0.2],
            "quantity_sold": [2, 200000, 25000, 3],
            "sales_value": [5.0, 400.0, 50.0, 9.0],
        }
    )

    filtered, summary = build_modeling_ready_features(features)

    assert len(filtered) == 2
    assert set(filtered["commodity_desc"]) == {"BREAD", "SOFT DRINKS"}
    assert int(summary.loc[summary["metric"] == "excluded_rows", "value"].iloc[0]) == 2

