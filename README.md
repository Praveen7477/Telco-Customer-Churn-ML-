# Telco-Customer-Churn-End-to-End-ML



> Predict customer churn before it happens — served via REST API and a live web UI, deployed on AWS ECS Fargate with full CI/CD.

---

## 🧭 Overview

This project builds and ships a production-grade machine learning solution for predicting customer churn in a telecom setting — from raw data to a containerized inference service running in the cloud.

**Key outcomes:**
- **Faster decisions** — Predict which customers are likely to leave so retention teams can act proactively
- **Operationalized ML** — Model is live via a REST API and Gradio UI; no notebooks required to test it
- **Repeatable delivery** — CI/CD + Docker mean every change is rebuilt, tested, and redeployed consistently
- **Traceable experiments** — MLflow logs every run, metric, and artifact for full reproducibility

---

## 🏗️ Architecture

```
GitHub Push → GitHub Actions (build & push image)
                    ↓
              Docker Hub
                    ↓
         ECS Fargate (force redeploy)
                    ↓
    ALB (HTTP:80) → Target Group (HTTP:8000)
                    ↓
         FastAPI app  +  Gradio UI (/ui)
```

---

## 🔧 What's Inside

| Layer | Technology | Details |
|---|---|---|
| **Data & Modeling** | XGBoost + feature engineering | Experiments logged to MLflow |
| **Model Tracking** | MLflow | Named experiment; runs, metrics & serialized model stored |
| **Inference Service** | FastAPI | `POST /predict` endpoint + `GET /` health check |
| **Web UI** | Gradio | Mounted at `/ui` for manual, shareable testing |
| **Containerization** | Docker | Uvicorn entrypoint on port `8000`; `PYTHONPATH=/app/src` |
| **CI/CD** | GitHub Actions | Builds image → pushes to Docker Hub → optionally triggers ECS redeploy |
| **Orchestration** | AWS ECS Fargate | Serverless container execution |
| **Networking** | AWS ALB | HTTP:80 listener → target group on HTTP:8000 (IP targets) |
| **Security** | AWS Security Groups | ALB: inbound 80 from `0.0.0.0/0`; Task: inbound 8000 from ALB SG only |
| **Observability** | AWS CloudWatch | Container stdout/stderr + ECS service events |

---

## 🚀 Deployment Flow

```
1. Push to main
        ↓
2. GitHub Actions builds Docker image → pushes to Docker Hub
        ↓
3. ECS service updated (manual or via workflow) → force new deployment
        ↓
4. ALB health checks hit GET / on port 8000
        ↓
5. Once healthy → traffic routed to new task
        ↓
6. Users call POST /predict or open Gradio at /ui via ALB DNS
```

---

## 📡 API Reference

### Health Check
```http
GET /
```
Returns `200 OK` when the service is running.

### Predict Churn
```http
POST /predict
Content-Type: application/json

{
  "tenure": 12,
  "monthly_charges": 65.5,
  "total_charges": 786.0,
  ...
}
```

**Response:**
```json
{
  "churn_probability": 0.73,
  "prediction": "churn"
}
```

---

## 🧪 Running Locally

### 1. Train the model
```bash
python src/train.py
```
Logs run to MLflow under the configured experiment name. Model is saved to `./mlruns/.../artifacts/model`.

### 2. Start the API
```bash
uvicorn src.app.main:app --reload --port 8000
```

### 3. Open the UI
Navigate to [http://localhost:8000/ui](http://localhost:8000/ui)

### 4. Run via Docker
```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

---

## ⚙️ Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PYTHONPATH` | Must include `/app/src` | Set in Dockerfile |
| `MLFLOW_EXPERIMENT_NAME` | MLflow experiment to load model from | `telco_churn` |
| `MODEL_PATH` | Override model path (local dev) | Auto-resolved from MLflow |

---

## 🧱 Project Structure

```
.
├── src/
│   ├── app/
│   │   ├── main.py          # FastAPI app + Gradio mount
│   │   └── predict.py       # Inference logic
│   ├── train.py             # Training pipeline
│   └── features.py          # Feature engineering
├── mlruns/                  # MLflow tracking (local)
├── Dockerfile
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD pipeline
└── requirements.txt
```

---

## 🔥 Roadblocks & How They Were Solved

<details>
<summary><strong>Unhealthy targets behind the ALB</strong></summary>

**Cause:** App didn't respond at the health-check path; listener/target port mismatches.

**Fix:**
- Added `GET /` health endpoint to FastAPI
- Confirmed ALB listener on port `80` forwards to target group on port `8000`
- Set TG health check path to `/`
</details>

<details>
<summary><strong>ModuleNotFoundError in container (<code>serving</code>)</strong></summary>

**Cause:** Python path inside the image didn't include `src/`.

**Fix:**
- Set `PYTHONPATH=/app/src` in the Dockerfile
- Corrected uvicorn app path to `src.app.main:app`
</details>

<details>
<summary><strong>ALB DNS timing out</strong></summary>

**Cause:** Security group rules not aligned with the traffic flow.

**Fix:**
- ALB SG: inbound port `80` from `0.0.0.0/0`
- Task SG: inbound port `8000` from **ALB SG only**
- Both SGs: outbound open
</details>

<details>
<summary><strong>ECS redeploy not picking up the new image</strong></summary>

**Cause:** Service still running the previous task definition revision.

**Fix:**
- Force new deployment after pushing: `aws ecs update-service --force-new-deployment`
- Optional CI step added to the GitHub Actions workflow to trigger this automatically
</details>

<details>
<summary><strong>Gradio UI error — "No runs found in experiment"</strong></summary>

**Cause:** Inference/UI expected an MLflow-logged model but couldn't resolve the run.

**Fix:**
- Standardized the MLflow experiment name across training and inference
- Inference loads the logged model consistently; falls back to a direct local path for dev (`./mlruns/.../artifacts/model`)
</details>

<details>
<summary><strong>Local testing vs. production model paths</strong></summary>

**Cause:** MLflow artifact URIs differ between local runs and the container.

**Fix:**
- Local dev: load directly via `./mlruns/.../artifacts/model`
- Production: container uses the packaged model path baked in at build time
</details>

---

## 📊 Experiment Tracking (MLflow)

All training runs are logged with:
- Feature importance scores
- Accuracy, AUC, F1, precision, recall
- The serialized XGBoost model as an artifact

To view the MLflow UI locally:
```powershell
C:\ProgramData\Anaconda3\python.exe -m mlflow ui --backend-store-uri "file:///C:/Users/Source/Documents/MLops/Telco-Customer-Churn-ML--main/mlruns"
```
Then open [http://localhost:5000](http://localhost:5000). If `python` opens the Microsoft Store stub on Windows, use the full interpreter path above or add your real Python install to `PATH`.

---

## 🛠️ CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) on every push to `main`:

1. Checks out the repo
2. Logs in to Docker Hub
3. Builds and tags the image
4. Pushes to Docker Hub
5. *(Optional)* Forces a new ECS deployment

---

## 📋 Requirements

- Python 3.10+
- Docker
- AWS CLI (for manual ECS deploys)
- An AWS account with ECS, ALB, CloudWatch, and ECR/Docker Hub configured

---

## 📄 License

MIT
