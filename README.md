# Keyword Spotting — Minimal Baseline (MFCC + Tiny CNN)

A fast, lightweight **keyword-spotting** project using the Google Speech Commands dataset (subset of 10 common words).  
Designed to **train in under 1 hour** and serve as a clean foundation for adding full **MLOps** capabilities (MLflow, CI/CD, DVC, FastAPI, etc.).

---

## 🎯 Goal
1. Build a **reproducible baseline**:  
   - Download dataset  
   - Extract MFCC features  
   - Train a small CNN classifier  
2. Extend step-by-step with modern MLOps practices.

---

## 📂 Dataset
- **Source:** [Google Speech Commands](https://arxiv.org/abs/1804.03209) (v0.02)  
- **Selected classes:** yes, no, up, down, left, right, on, off, stop, go





























keyword-spotting-mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py
│   └── training/
│       ├── __init__.py
│       └── train.py
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_train_smoke.py
