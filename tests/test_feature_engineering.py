import pandas as pd

from src.features.build_features import build_modeling_table


def test_build_modeling_table_adds_promo_and_lag_features():
    transactions = pd.DataFrame(
        {
            "household_key": [1, 2, 1, 2],
            "basket_id": [10, 11, 12, 13],
            "day": [1, 8, 15, 22],
            "product_id": [100, 100, 100, 100],
            "quantity": [2, 3, 4, 5],
            "sales_value": [4.0, 5.7, 7.2, 8.5],
            "store_id": [20, 20, 20, 20],
            "retail_disc": [0.0, -0.3, -0.8, 0.0],
            "coupon_disc": [0.0, 0.0, -0.5, 0.0],
            "coupon_match_disc": [0.0, 0.0, 0.0, 0.0],
            "trans_time": [100, 100, 100, 100],
            "week_no": [1, 2, 3, 4],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": [100],
            "department": ["GROCERY"],
            "commodity_desc": ["BREAD"],
            "sub_commodity_desc": ["BREAD"],
            "brand": ["Private"],
            "curr_size_of_product": ["12 OZ"],
        }
    )
    causal = pd.DataFrame(
        {
            "product_id": [100],
            "store_id": [20],
            "week_no": [3],
            "display": ["A"],
            "mailer": ["0"],
        }
    )

    result = build_modeling_table(transactions, products=products, causal_data=causal)

    assert list(result["quantity_sold"]) == [2, 3, 4, 5]
    assert result.loc[result["week_no"] == 3, "is_display"].iloc[0] == 1
    assert result.loc[result["week_no"] == 4, "lag_quantity_1"].iloc[0] == 4
    assert result.loc[result["week_no"] == 4, "rolling_quantity_mean_4"].iloc[0] == 3
    assert result.loc[result["week_no"] == 1, "department"].iloc[0] == "GROCERY"

