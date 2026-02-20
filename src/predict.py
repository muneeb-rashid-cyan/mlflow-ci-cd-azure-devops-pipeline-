import pickle
import numpy as np
from typing import List

MODEL_PATH = "models/trained_model.pkl"
_artifact = None


def load_model(path: str = MODEL_PATH):
    global _artifact
    if _artifact is None:
        with open(path, "rb") as f:
            _artifact = pickle.load(f)
    return _artifact


def predict_single(features: List[float]) -> dict:
    artifact = load_model()
    model = artifact["model"]
    scaler = artifact["scaler"]
    target_names = artifact["target_names"]

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred_idx = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0].tolist()

    return {
        "prediction": pred_idx,
        "class_name": target_names[pred_idx],
        "probabilities": {name: round(p, 4) for name, p in zip(target_names, proba)},
    }


def predict_batch(features_list: List[List[float]]) -> List[dict]:
    return [predict_single(f) for f in features_list]


def get_model_info() -> dict:
    artifact = load_model()
    return {
        "feature_names": artifact["feature_names"],
        "target_names": artifact["target_names"],
        "metrics": artifact["metrics"],
        "run_id": artifact["run_id"],
    }