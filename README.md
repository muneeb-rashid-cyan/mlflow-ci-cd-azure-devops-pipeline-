# Mlops-Fastapi-Docker-Azure-Devops-pipeline  🌸
### MLflow + FastAPI + Docker + Azure Pipelines — End-to-End MLOps Pipeline

![Pipeline](https://img.shields.io/badge/Azure%20Pipelines-4%20Stages-blue)
![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?logo=mlflow)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)

---

## 🎯 What This Project Does

A production-grade MLOps pipeline that automates the entire ML lifecycle:

```
git push → Azure Pipeline triggers →
Train model → Log to MLflow →
Run tests → Build Docker image →
Push to Docker Hub → Ready to deploy anywhere
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  Developer pushes code to Azure Repos               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  AZURE PIPELINE (4 Automated Stages)                │
│                                                     │
│  Stage 1 🧠 TRAIN                                   │
│  ├── Install dependencies                           │
│  ├── Train RandomForest with MLflow tracking        │
│  ├── Log metrics (accuracy, precision, recall, F1)  │
│  ├── Quality gate: accuracy >= 85%                  │
│  └── Save model as pipeline artifact                │
│                                                     │
│  Stage 2 🧪 TEST                                    │
│  ├── Download trained model artifact                │
│  ├── Run 15 unit tests (model + API)                │
│  ├── Code coverage >= 80%                           │
│  └── Publish test results                           │
│                                                     │
│  Stage 3 🐳 BUILD                                   │
│  ├── Build Docker image                             │
│  ├── Run container health check                     │
│  └── Tag image with build ID + accuracy             │
│                                                     │
│  Stage 4 📦 PUBLISH                                 │
│  ├── Push to Docker Hub (3 tags)                    │
│  └── Image ready to deploy anywhere                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  DOCKER HUB                                         │
│  yourusername/iris-classifier:latest                │
│  yourusername/iris-classifier:build-123             │
│  yourusername/iris-classifier:acc-0.9333            │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
IrisOps/
├── src/
│   ├── train.py          # MLflow experiment tracking + model training
│   ├── app.py            # FastAPI REST API (4 endpoints)
│   └── predict.py        # Prediction logic (single + batch)
├── tests/
│   ├── test_model.py     # 9 model quality tests
│   └── test_api.py       # 6 API endpoint tests
├── models/               # Saved model artifacts (git-ignored)
├── mlflow/               # MLflow tracking store (git-ignored)
├── Dockerfile            # Container definition
├── azure-pipelines.yml   # 4-stage CI/CD pipeline
├── pyproject.toml        # uv project config
└── requirements.txt      # Python dependencies
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Desktop

### Setup

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/IrisOps.git
cd IrisOps

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
uv pip install -r requirements.txt

# Train model
python src/train.py

# Start API
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=src

# View MLflow UI
mlflow ui --backend-store-uri mlflow/mlruns
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model accuracy |
| GET | `/model/info` | Feature names, metrics, MLflow run ID |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Auto-generated Swagger UI |

### Example Request

```python
import httpx

# Single prediction
response = httpx.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())
# {
#   "prediction": 0,
#   "class_name": "setosa",
#   "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
# }
```

---

## 🐳 Docker

```bash
# Build image
docker build -t iris-classifier .

# Run container
docker run -p 8000:8000 iris-classifier

# Test it
curl http://localhost:8000/health
```

---

## ☁️ Azure Pipeline Setup

### Prerequisites
1. Azure DevOps account
2. Docker Hub account

### Steps

```bash
# Push code to Azure Repos
git remote add origin https://dev.azure.com/YOUR_ORG/YOUR_PROJECT/_git/IrisOps
git checkout -b dev
git push -u origin dev
```

Then in Azure DevOps:
1. **Create Docker Hub service connection** — name it `DockerHubServiceConnection`
2. **Create pipeline** — point to `azure-pipelines.yml`
3. **Add variable** — `DOCKER_HUB_USERNAME` = your Docker Hub username
4. **Run** — watch all 4 stages go green

---

## Quality Gates

| Gate | Threshold | Stage |
|------|-----------|-------|
| Model accuracy | >= 85% | Train |
| Test coverage | >= 80% | Test |
| Container health | HTTP 200 | Build |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| scikit-learn | RandomForest model training |
| MLflow | Experiment tracking + model registry |
| FastAPI | REST API serving |
| Docker | Containerization |
| Azure Pipelines | CI/CD automation |
| Docker Hub | Container registry |
| pytest | Testing + coverage |
| uv | Python package management |

---

## 👨‍💻 Author

**Muneeb Rashid** — ML Engineer
