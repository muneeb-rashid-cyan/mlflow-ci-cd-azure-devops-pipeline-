import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uvicorn

from predict import predict_single, predict_batch, get_model_info

app = FastAPI(
    title="Iris Classifier API",
    description="MLflow-trained RandomForest classifier served via FastAPI",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=4,
        max_items=4,
        example=[5.1, 3.5, 1.4, 0.2],
        description="[sepal_length, sepal_width, petal_length, petal_width]",
    )


class BatchPredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        example=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]],
    )


@app.get("/health")
def health():
    try:
        info = get_model_info()
        return {
            "status": "healthy",
            "model_accuracy": info["metrics"].get("accuracy"),
            "run_id": info["run_id"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.get("/model/info")
def model_info():
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_single(request.features)
        return {"input": request.features, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchPredictRequest):
    try:
        results = predict_batch(request.features)
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)