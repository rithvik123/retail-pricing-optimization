from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_recommend_price_endpoint():
    response = client.post(
        "/recommend-price",
        json={
            "product_id": 1017772,
            "store_id": 327,
            "current_price": 2.50,
            "unit_cost": 1.50,
            "candidate_discounts": [0, 5, 10, 15, 20],
            "inventory_limit": 500,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "recommended_price" in payload
    assert "business_reason" in payload
    assert payload["simulations"]

