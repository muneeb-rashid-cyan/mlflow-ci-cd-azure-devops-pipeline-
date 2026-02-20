import sys
import os
import pickle
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

MODEL_PATH = os.environ.get("MODEL_PATH", "models/trained_model.pkl")


@pytest.fixture(scope="module")
def artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_artifact_keys(artifact):
    required = {"model", "scaler", "feature_names", "target_names", "metrics", "run_id"}
    assert required.issubset(artifact.keys())


def test_accuracy_threshold(artifact):
    assert artifact["metrics"]["accuracy"] >= 0.85, (
        f"Accuracy {artifact['metrics']['accuracy']} below 0.85"
    )


def test_feature_names(artifact):
    assert len(artifact["feature_names"]) == 4


def test_target_names(artifact):
    assert artifact["target_names"] == ["setosa", "versicolor", "virginica"]


def test_prediction_shape(artifact):
    model = artifact["model"]
    scaler = artifact["scaler"]
    X = np.array([[5.1, 3.5, 1.4, 0.2]])
    X_sc = scaler.transform(X)
    pred = model.predict(X_sc)
    assert pred.shape == (1,)


def test_prediction_valid_class(artifact):
    model = artifact["model"]
    scaler = artifact["scaler"]
    X = np.array([[5.1, 3.5, 1.4, 0.2]])
    X_sc = scaler.transform(X)
    pred = int(model.predict(X_sc)[0])
    assert pred in [0, 1, 2]


def test_probabilities_sum_to_one(artifact):
    model = artifact["model"]
    scaler = artifact["scaler"]
    X = np.array([[5.1, 3.5, 1.4, 0.2]])
    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[0]
    assert abs(proba.sum() - 1.0) < 1e-6


def test_batch_prediction(artifact):
    model = artifact["model"]
    scaler = artifact["scaler"]
    X = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [5.8, 2.7, 4.1, 1.0]])
    X_sc = scaler.transform(X)
    preds = model.predict(X_sc)
    assert len(preds) == 3