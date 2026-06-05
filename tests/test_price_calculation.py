import pandas as pd
import pytest

from src.features.build_features import build_transaction_price_features


def test_dunnhumby_price_formulas_handle_negative_discounts():
    frame = pd.DataFrame(
        {
            "sales_value": [1.39],
            "retail_disc": [-0.60],
            "coupon_disc": [0.0],
            "coupon_match_disc": [0.0],
            "quantity": [1],
        }
    )

    result = build_transaction_price_features(frame).iloc[0]

    assert result["effective_unit_price"] == 1.39
    assert result["loyalty_card_price"] == pytest.approx(1.99)
    assert result["non_loyalty_card_price"] == 1.39
    assert result["shelf_price_estimate"] == pytest.approx(1.99)
    assert round(result["discount_percentage"], 4) == round(0.60 / 1.99, 4)
    assert result["has_retail_discount"] == 1
    assert result["has_coupon_discount"] == 0
