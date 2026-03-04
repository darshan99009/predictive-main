# ⚙️ Turbofan Engine RUL Prediction — IST_27 Capstone Project

> **Predicting Remaining Useful Life (RUL) of turbofan engines using a Hybrid XGBoost–LSTM Ensemble on the NASA C-MAPSS dataset**

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-ff4b4b?logo=streamlit)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)](https://xgboost.readthedocs.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Methodology](#4-methodology)
5. [Model Architecture](#5-model-architecture)
6. [Feature Engineering](#6-feature-engineering)
7. [Ensemble Strategy](#7-ensemble-strategy)
8. [Results](#8-results)
9. [Dashboard](#9-dashboard)
10. [Installation & Usage](#10-installation--usage)
11. [Model Files](#11-model-files)
12. [Tech Stack](#12-tech-stack)
13. [Team](#13-team)

---

## 1. Project Overview

This project addresses **Predictive Maintenance** for aircraft turbofan engines. The objective is to predict how many operational cycles remain before an engine requires maintenance — the **Remaining Useful Life (RUL)**.

A late prediction (over-predicting RUL) risks catastrophic engine failure. An early prediction (under-predicting) wastes maintenance budget. The goal is a model that is both accurate and conservatively safe.

**Core Approach:**
A two-branch hybrid model is trained per dataset sub-configuration:
- **LSTM with Attention** — captures sequential degradation patterns from sensor time-series
- **XGBoost** — learns from tabular statistical features engineered per engine
- **Weighted Ensemble** — combines both predictions optimally using a learned blending weight α

The system is deployed as an interactive **Streamlit dashboard** on Streamlit Cloud with real-time inference on uploaded test files.

---

## 2. Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

The dataset simulates run-to-failure degradation of turbofan engines across 4 sub-configurations:

| Sub-dataset | Operating Conditions | Fault Modes | Engines (Train/Test) | Difficulty |
|-------------|---------------------|-------------|----------------------|------------|
| FD001 | 1 | 1 (HPC degradation) | 100 / 100 | Easiest |
| FD002 | 6 | 1 (HPC degradation) | 260 / 259 | Moderate |
| FD003 | 1 | 2 (HPC + Fan) | 100 / 100 | Moderate |
| FD004 | 6 | 2 (HPC + Fan) | 248 / 248 | Hardest |

**Sensors (26 raw channels → 14 informative):**

| Sensor | Physical Measurement | Role in Degradation |
|--------|---------------------|---------------------|
| s2 | LPC Outlet Temperature | Rises with HPC wear |
| s3 | HPC Outlet Temperature | Primary degradation indicator |
| s4 | LPT Outlet Temperature | Key thermal signature |
| s7 | HPC Outlet Static Pressure | Drops with degradation |
| s8 | Fuel/Air Ratio | Changes with combustion shift |
| s9 | LPC Outlet Pressure | Compressor wear proxy |
| s11 | HPC Outlet Pressure | Directly affected by HPC fault |
| s12 | Ratio of Fuel Flow | Efficiency indicator |
| s13 | Corrected Fan Speed | Fan health proxy |
| s14 | LPT Coolant Bleed | Secondary degradation signal |
| s15 | Bypass Ratio | Fan degradation for FD003/FD004 |
| s17 | Bleed Enthalpy | Thermal stress indicator |
| s20 | High Pressure Turbine Coolant Bleed | HPT wear |
| s21 | Low Pressure Turbine Coolant Bleed | LPT wear |

**Removed (zero/near-zero variance, non-informative):** s1, s5, s6, s10, s16, s18, s19

> **Note:** Dataset files are included in the repository. No external download required.

---

## 3. Project Structure

```
IST_27-Capstone_Project/
│
├── app.py                          # Main Streamlit dashboard
├── model_def.py                    # LSTM model class definition (PyTorch)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit theme config
│
├── models/
│   ├── lstm_FD001.pt               # Trained LSTM weights
│   ├── lstm_FD002.pt
│   ├── lstm_FD003.pt
│   ├── lstm_FD004.pt
│   ├── xgb_FD001.json              # Trained XGBoost model
│   ├── xgb_FD002.json
│   ├── xgb_FD003.json
│   ├── xgb_FD004.json
│   ├── scaler_FD001.pkl            # MinMaxScaler fitted on training data
│   ├── scaler_FD002.pkl
│   ├── scaler_FD003.pkl
│   ├── scaler_FD004.pkl
│   ├── features_FD001.npy          # Feature name list
│   ├── features_FD002.npy
│   ├── features_FD003.npy
│   ├── features_FD004.npy
│   ├── alpha_FD001.npy             # Learned ensemble blend weight α
│   ├── alpha_FD002.npy
│   ├── alpha_FD003.npy
│   └── alpha_FD004.npy
│
├── data/
│   ├── train_FD001.txt             # Training sequences
│   ├── train_FD002.txt
│   ├── train_FD003.txt
│   ├── train_FD004.txt
│   ├── test_FD001.txt              # Test sequences
│   ├── test_FD002.txt
│   ├── test_FD003.txt
│   ├── test_FD004.txt
│   ├── RUL_FD001.txt               # Ground-truth RUL for test set
│   ├── RUL_FD002.txt
│   ├── RUL_FD003.txt
│   └── RUL_FD004.txt
│
└── README.md
```

---

## 4. Methodology

### 4.1 Preprocessing Pipeline

```
Raw Test File (.txt)
        │
        ▼
1. Column Assignment
   26 columns: engine_id, cycle, op1–op3, s1–s21
        │
        ▼
2. Sensor Filtering
   Drop 7 non-informative sensors: s1,s5,s6,s10,s16,s18,s19
   Retain 14 sensors + 3 operational settings
        │
        ▼
3. Operating Condition Normalization (FD002/FD004)
   KMeans clustering on [op1, op2, op3]
   Z-score normalize within each cluster
        │
        ▼
4. Feature Engineering
   Per engine, per sensor — 8 statistical features
        │
        ▼
5. Sequence Construction (LSTM)
   Window length = 50 cycles
   Zero-pad engines with fewer than 50 cycles
        │
        ▼
6. Tabular Feature Extraction (XGBoost)
   Last-window feature vector for each engine
        │
        ▼
7. Dual Inference
   LSTM(sequence) → pred_lstm
   XGBoost(tabular) → pred_xgb
        │
        ▼
8. Ensemble Blending
   pred_hybrid = α × pred_xgb + (1−α) × pred_lstm
   Clipped to [0, 125]
```

### 4.2 RUL Labeling (Training)

The RUL target during training uses a **piecewise linear capping** strategy:

```
RUL_label = min(true_RUL, RUL_CAP)   where RUL_CAP = 125
```

Engines far from failure are all labeled as 125 — the model focuses on the degradation phase rather than the healthy early-life phase.

### 4.3 Loss Function — Huber Loss (δ = 10)

Training uses **Huber Loss** instead of MSE or MAE:

```
L(y, ŷ) = { ½(y − ŷ)²             if |y − ŷ| ≤ δ
           { δ·|y − ŷ| − ½δ²      if |y − ŷ| > δ
```

- **Below δ = 10**: behaves like MSE — precise corrections for normal errors
- **Above δ = 10**: behaves like MAE — robust against outlier engines with unusual degradation

This prevents a few abnormal engines from dominating gradient updates during training.

---

## 5. Model Architecture

### 5.1 LSTM with Attention

```
Input:  [batch, seq_len=50, n_features]
         │
    LSTM Layer 1  →  hidden=128, dropout=0.2
         │
    LSTM Layer 2  →  hidden=128
         │
    Attention     →  Softmax weights over time steps
         │
    FC 128        →  BatchNorm + ReLU + Dropout(0.2)
         │
    FC 64         →  ReLU
         │
    FC 1          →  Output: RUL prediction
```

**Training config:** Adam (lr=1e-3), ReduceLROnPlateau, batch=256, max epochs=100, early stopping patience=15

### 5.2 XGBoost Regressor

| Hyperparameter | Value |
|---------------|-------|
| n_estimators | 1500 |
| learning_rate (η) | 0.03 |
| max_depth | 3 |
| reg_alpha (L1) | 1.0 |
| reg_lambda (L2) | 5.0 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | reg:squarederror |

---

## 6. Feature Engineering

For each engine, **8 statistical features** are computed per sensor (14 sensors × 8 = 112 features), calculated over a rolling 30-cycle window:

| Feature | Description |
|---------|-------------|
| Rolling Mean | Smoothed sensor level |
| Rolling Std | Variability / noise level |
| Slope | Linear trend coefficient |
| EWM Mean | Exponentially weighted mean (α=0.1) |
| IQR | Interquartile range |
| Range | Max − Min |
| P10 | 10th percentile |
| P90 | 90th percentile |

**Top contributing features by XGBoost importance:**

| Rank | Feature | Importance | Sensor |
|------|---------|-----------|--------|
| 1 | s4_rmean | 14.2% | LPT Outlet Temperature |
| 2 | s4_slope | 11.8% | LPT Outlet Temperature |
| 3 | s9_rmean | 9.8% | LPC Outlet Pressure |
| 4 | s14_ewm | 8.7% | LPT Coolant Bleed |
| 5 | s11_slope | 7.3% | HPC Outlet Pressure |

---

## 7. Ensemble Strategy

```
ŷ_hybrid = α × ŷ_XGBoost + (1 − α) × ŷ_LSTM
```

α is optimized per dataset by minimizing validation RMSE via 1D grid search over α ∈ [0, 1].

| Dataset | α* | XGBoost Weight | LSTM Weight |
|---------|----|----------------|-------------|
| FD001 | ~0.55 | 55% | 45% |
| FD002 | ~0.45 | 45% | 55% |
| FD003 | ~0.50 | 50% | 50% |
| FD004 | ~0.48 | 48% | 52% |

XGBoost excels at static pattern recognition; LSTM captures temporal dynamics. Their errors are partially uncorrelated, so blending reduces overall variance.

---

## 8. Results

### Hybrid Model Performance (Test Set)

| Dataset | LSTM RMSE | XGBoost RMSE | **Hybrid RMSE** | R² |
|---------|-----------|-------------|-----------------|-----|
| FD001 | 15.8 | 14.1 | **13.02** | 0.84 |
| FD002 | 24.3 | 22.6 | **20.11** | 0.77 |
| FD003 | 17.2 | 15.9 | **14.23** | 0.82 |
| FD004 | 28.1 | 26.4 | **24.18** | 0.71 |

### Maintenance Alert Thresholds

| Status | RUL Range | Action |
|--------|-----------|--------|
| 🔴 CRITICAL | 0 – 20 cycles | Immediate maintenance required |
| 🟡 WARNING | 21 – 50 cycles | Schedule maintenance within 1 week |
| 🟢 HEALTHY | 51 – 125 cycles | Monitor normally |

---

## 9. Dashboard

The Streamlit dashboard provides a **7-tab interactive interface**:

| Tab | Name | Description |
|-----|------|-------------|
| 1 | 🔍 Engine Inspector | Per-engine RUL gauge, status badge, model comparison bars |
| 2 | 📊 RUL Distribution | Histogram, box plots, CDF, summary statistics |
| 3 | 🗺️ Fleet Heatmap | Color-coded engine health grid and donut chart |
| 4 | ⚠️ Maintenance Alerts | Priority-sorted Critical / Warning / Healthy engine lists |
| 5 | 📈 Sensor Degradation | 14-sensor trend lines and full-fleet sensor heatmap |
| 6 | 🔬 Feature Importance | XGBoost feature rankings and sensor group aggregation |
| 7 | ℹ️ About | Project summary, usage guide, reference tables |

---

## 10. Installation & Usage

### Local Setup

```bash
# Clone the repository
git clone https://github.com/darshan99009/IST_27-Capstone_Project
cd IST_27-Capstone_Project

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

### Running Predictions

1. Open the dashboard
2. Select a dataset in the sidebar (FD001–FD004)
3. Upload the corresponding `test_FDxxx.txt` file
4. *(Optional)* Upload `RUL_FDxxx.txt` to evaluate against ground truth
5. Explore the 7 tabs for full analysis

### Input File Format

Space-delimited `.txt`, 26 columns, no header:

```
engine_id  cycle  op1  op2  op3  s1  s2  s3  s4  s5  s6  s7  s8  s9
s10  s11  s12  s13  s14  s15  s16  s17  s18  s19  s20  s21
```

---

## 11. Model Files

Each dataset ships with 5 artefacts in `models/`:

| File | Description |
|------|-------------|
| `lstm_FDxxx.pt` | PyTorch LSTM state dict |
| `xgb_FDxxx.json` | XGBoost booster (JSON format) |
| `scaler_FDxxx.pkl` | Fitted MinMaxScaler (joblib) |
| `features_FDxxx.npy` | Ordered feature column names |
| `alpha_FDxxx.npy` | Learned ensemble weight α |

---

## 12. Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Deep Learning | PyTorch 2.x |
| Gradient Boosting | XGBoost 1.7 |
| Data Processing | NumPy, Pandas |
| Feature Scaling | scikit-learn (MinMaxScaler) |
| Clustering | scikit-learn (KMeans) |
| Visualization | Plotly, Plotly Express |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## 13. Team

**IST_27 — Capstone Project**

**Repository:** [github.com/darshan99009/IST_27-Capstone_Project](https://github.com/darshan99009/IST_27-Capstone_Project)

---

## License

This project is licensed under the MIT License.

---

*Built with ⚙️ for the NASA C-MAPSS Predictive Maintenance Challenge*
