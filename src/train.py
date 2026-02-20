import os
import json
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.environ.get("MODEL_PATH", "models/trained_model.pkl")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlflow/mlruns")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))

# MLflow is optional — gracefully degrade if not installed
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed — skipping experiment tracking. Run: uv add mlflow")


class _DummyRun:
    """Fallback when MLflow is not available."""
    class _Info:
        run_id = "local-no-mlflow"
    info = _Info()
    def __enter__(self): return self
    def __exit__(self, *_): pass


def train():
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("iris-classifier")

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2,
        "random_state": 42,
    }

    run_ctx = (mlflow.start_run() if MLFLOW_AVAILABLE else _DummyRun())

    with run_ctx as run:
        if MLFLOW_AVAILABLE:
            mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall":    recall_score(y_test, y_pred, average="weighted"),
            "f1_score":  f1_score(y_test, y_pred, average="weighted"),
        }

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

        print(f"Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Quality gate
        if metrics["accuracy"] < ACCURACY_THRESHOLD:
            raise ValueError(
                f"Accuracy {metrics['accuracy']:.4f} below threshold {ACCURACY_THRESHOLD}"
            )

        # Persist model + metadata
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        artifact = {
            "model":        model,
            "scaler":       scaler,
            "feature_names": list(iris.feature_names),
            "target_names": iris.target_names.tolist(),
            "metrics":      metrics,
            "run_id":       run.info.run_id,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)

        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✅ Model saved → {MODEL_PATH}")
        print(f"✅ Quality gate passed: accuracy={metrics['accuracy']:.4f}")
        return metrics


if __name__ == "__main__":
    train()