import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

# Train model before tests if it doesn't exist
MODEL_PATH = os.environ.get("MODEL_PATH", "models/trained_model.pkl")
if not os.path.exists(MODEL_PATH):
    from train import train
    train()

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_accuracy" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data
    assert "target_names" in data
    assert "metrics" in data


def test_predict_setosa():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "class_name" in data
    assert "probabilities" in data
    assert data["class_name"] == "setosa"


def test_predict_virginica():
    response = client.post("/predict", json={"features": [6.7, 3.0, 5.2, 2.3]})
    assert response.status_code == 200
    data = response.json()
    assert data["class_name"] == "virginica"


def test_predict_invalid_features():
    # Too few features
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 422


def test_batch_predict():
    payload = {
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.8, 2.7, 4.1, 1.0],
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3