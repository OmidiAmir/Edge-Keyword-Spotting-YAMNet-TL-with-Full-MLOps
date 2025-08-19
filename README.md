# Keyword Spotting (Speech Commands) — MLOps-Ready Project

Fast, lightweight keyword spotting on the **Google Speech Commands** dataset (subset of 10 common words).  
Trains quickly (<20 min on RTX 3060) and demonstrates a full **MLOps workflow** step by step:  
data → model → tests → config → MLflow tracking → CI/CD → FastAPI API.

---

## 🚀 Features so far
- **Dataset** wrapper (`KSDataset`) with MFCC features + class filtering.
- **Training** script with YAML config support.
- **Pytest** smoke tests for dataset + training pipeline.
- **Experiment tracking** with MLflow (logs metrics, artifacts, models).
- **Model export** to TorchScript (`models/ts_model.pt`).
- **FastAPI** inference service (real-time keyword spotting API).

---

## 📂 Repo Layout (current)

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
- Python 3.9+
- Torch, Torchaudio, NumPy, tqdm, MLflow, FastAPI, Uvicorn
- (Exact versions pinned in pyproject.toml)
---

## 🛠 Installation
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

---

## 🧠 Baseline Model

- Features: MFCC (n_mfcc=40) + standardization
- Architecture: 3 convolutional blocks → Global Average Pool → Fully Connected
- Parameters: ~0.5M
- Training time: < 20 minutes on RTX 3060

--- 

## 🧪 Training
Run training (logs to MLflow automatically):
```bash
python scripts/train_model.py --data_root data --epochs 10 --batch_size 128

```

---

## 🔄 CI/CD with GitHub Actions

This repo includes a GitHub Actions workflow (.github/workflows/ci.yml) that runs automatically on every push or pull request:
- ✅ Unit tests (dataset + preprocessing)
- ✅ Smoke tests (training & inference run)

This ensures code stays stable and reproducible as new MLOps components are added.

--- 
## 📈 MLflow Tracking
```bash
mlflow ui
```
Then open http://127.0.0.1:5000 to compare experiments and download checkpoints.

---

## 📦 Export TorchScript Model
Pick a run from MLflow, export it:
```bash
Pick a run from MLflow, export it:
```
---

```bash
python scripts/infer_wav.py --wav data/SpeechCommands/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav

```

## 🌐 Serving with FastAPI
- TorchScript model is exported to models/ts_model.pt
- FastAPI app serves endpoints:
  - `GET /health` → service status
  - `POST /predict` → keyword prediction for uploaded .wav
Run locally:
```bash
uvicorn src.kws.api:app --reload --port 8008

```
---

## 🔄 CI/CD with GitHub Actions
This repo includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs:
- Unit tests on every push and pull request
- Smoke tests to validate training and inference scripts

This ensures the codebase remains stable as new MLOps components are added.

---
## 📈 Experiment Tracking
- Integrated with MLflow
- Tracks hyperparameters, metrics, and models
- Run selection based on best validation accuracy

---

