# Keyword Spotting (Speech Commands) — MLOps-Ready Project

[![CI](https://github.com/OmidiAmir/Edge-Keyword-Spotting-YAMNet-TL-with-Full-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/OmidiAmir/Edge-Keyword-Spotting-YAMNet-TL-with-Full-MLOps/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-tracking-success)
![Docker](https://img.shields.io/badge/Docker-supported-informational)

Fast, lightweight keyword spotting on the **Google Speech Commands** dataset (subset of 10 common words).  
Trains quickly (<20 min on RTX 3060) and demonstrates a full **MLOps workflow** step by step:  
data → model → tests → config → MLflow tracking → CI/CD → FastAPI API.

---

## 🚀 Features so far
- **Dataset**: `KSDataset` with MFCC features + class filtering (10 keywords).
- **Training**: baseline CNN (~0.5M params) with YAML config.
- **Tracking**: MLflow (metrics, artifacts, models).
- **Export**: TorchScript (`models/ts_model.pt`).
- **Serving**: FastAPI app with robust audio decoding (ffmpeg + libsndfile).
- **CI**: GitHub Actions for tests (and easy extension to Docker).

---

```bash

keyword-spotting-mlops/
├─ pyproject.toml
├─ .gitignore
├─ README.md
├─ data/                   # dataset files (gitignored)
│   └─ SpeechCommands
│       └─ speech_commands_v0.02
│            └─ ...
├─ models/                 # exported TorchScript models
├─ scripts/                # CLI entry points
│    ├─ download_data.py   # downloads Speech Commands
│    ├─ train_model.py     # training with MLflow
│    ├─ evaluate_model.py  # evaluation for specific run
│    ├─ export_model.py    # export TorchScript model
│    └─ infer_wav.py       # inference on single WAV file
├─ src/
│   ├─ kws/
│   │   ├─ __init__.py
│   │   ├─ data/           # dataset utils
│   │   │  └─ dataset.py
│   │   ├─ models/         # CNN model
│   │   └─ training/       # training loop
├─ tests/                  # pytest unit + smoke tests
│   └─ test_data.py
└─ .github/workflows/
    └─ ci.yml              # GitHub Actions workflow

```

---
## 🛠 Requirements
- Python **3.11+**
- PyTorch, Torchaudio, NumPy, tqdm, **MLflow**, **FastAPI**, **Uvicorn**
---

## 🛠 Installation (dev)
```bash
python -m venv .venv
# Activate venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

pip install -e .
```
---

## 📂 Dataset
- **Source:** [Google Speech Commands](https://arxiv.org/abs/1804.03209) (v0.02)  
- **Selected classes:** yes, no, up, down, left, right, on, off, stop, go

- Sampling rate: **16 kHz**  
- Format: WAV, 1-second duration

```bash
python scripts/download_data.py --out data/SpeechCommands

```
---

## 🧠 Baseline Model

- Features: MFCC (n_mfcc=40) + standardization
- Architecture: 3 convolutional blocks → Global Average Pool → Fully Connected
- Parameters: ~0.5M
- Training time: < 20 minutes on RTX 3060

--- 

## 🧪 Train → Evaluate → Export
```bash
# Train (logs to MLflow)
python scripts/train_model.py --config configs/train_model.yaml

# Evaluate a specific run
python scripts/evaluate_model.py --run_id <RUN_ID>

# Export TorchScript
python scripts/export_model.py --run_id <RUN_ID> --out models/ts_model.pt

# Local single-file inference (sanity)
python scripts/infer_wav.py --wav data/SpeechCommands/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav

```

---

## 📈 MLflow Tracking
```bash
mlflow ui
```
Open http://127.0.0.1:5000 to view experiments, metrics, and checkpoints.

---

## 🌐 Serving with FastAPI

Once you’ve exported a TorchScript model to `models/ts_model.pt`, you can serve it via FastAPI.

### Run locally (no Docker)
```bash
uvicorn kws.serve.app:app --host 0.0.0.0 --port 8000 --reload

```
Endpoints:
- GET /health → service status
- GET /selftest → sanity check (generates + reads back a dummy tone)
- POST /predict → keyword prediction for uploaded .wav file
- POST /predict-bytes → keyword prediction from raw audio bytes

Example request:
```bash
# Linux/macOS
curl -F "file=@data/SpeechCommands/.../yes/0a7c2a8d_nohash_0.wav" http://localhost:8000/predict

# Windows PowerShell
$f="C:\path\to\some.wav"
curl.exe -F "file=@$f" http://localhost:8000/predict
```
---
## 📦 Docker Support
Build & run containerized API:
```bash
# Build
docker build -t kws-api .

# Run
docker run --rm -p 8000:8000 \
  -v "${PWD}/models:/app/models:ro" \
  -e MODEL_PATH=/app/models/ts_model.pt \
  kws-api
```
test
```bash
curl http://localhost:8000/health
curl -F "file=@data/SpeechCommands/.../yes/0a7c2a8d_nohash_0.wav" http://localhost:8000/predict
```

---

## 🔄 CI/CD with GitHub Actions

This repo includes a GitHub Actions workflow (.github/workflows/ci.yml) that runs automatically on every push or pull request:
- ✅ Unit tests (dataset + preprocessing)
- ✅ Smoke tests (training & inference run)
- 🔜 Extend with Docker build/push

This ensures code stays stable and reproducible as new MLOps components are added.

---
