# ⚙️ Turbofan Engine RUL Prediction Dashboard

**Hybrid XGBoost–LSTM model for Remaining Useful Life (RUL) prediction**
NASA C-MAPSS · FD001 / FD002 / FD003 / FD004

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-main.streamlit.app)

---

## 🚀 Live App
👉 **[predictive-main.streamlit.app]([https://predictive-main.streamlit.app](https://predictive-main-6zqfnrdbtguflkfgf3vym3.streamlit.app/))**

---

## 📁 Repository Structure

```
predictive-main/
├── app.py                     ← Streamlit dashboard (main entry point)
├── model_def.py               ← LSTM architecture definition
├── requirements.txt           ← Python dependencies
├── .streamlit/
│   └── config.toml            ← Streamlit dark theme config
│
├── alpha_FD001.npy            ┐
├── alpha_FD002.npy            │
├── alpha_FD003.npy            │
├── alpha_FD004.npy            │
├── features_FD001.npy         │
├── features_FD002.npy         │  Model files (trained in Colab)
├── features_FD003.npy         │
├── features_FD004.npy         │
├── lstm_FD001.pt              │
├── lstm_FD002.pt              │
├── lstm_FD003.pt              │
├── lstm_FD004.pt              │
├── scaler_FD001.pkl           │
├── scaler_FD002.pkl           │
├── scaler_FD003.pkl           │
├── scaler_FD004.pkl           │
├── xgb_FD001.pkl              │
├── xgb_FD002.pkl              │
├── xgb_FD003.pkl              │
└── xgb_FD004.pkl              ┘
```

---

## 🧠 Model Architecture

### LSTM (Temporal Learning)
| Parameter | Value |
|-----------|-------|
| Layers | 2 LSTM layers |
| Hidden size | 128 |
| Attention | Multi-step self-attention |
| Dropout | 0.2 (LSTM) + 0.3 (FC) |
| Loss | Huber (δ=10) |
| Optimizer | Adam + Cosine LR decay |
| Epochs | 100 (FD001/FD003) · 120 (FD002/FD004) |

### XGBoost (Feature Learning)
| Parameter | Value |
|-----------|-------|
| Max depth | 3 |
| Trees | 1500–2000 (early stopping) |
| Learning rate | 0.03 |
| Regularization | L1=1.0, L2=5.0 |
| Features | Last values · mean · std · slope · Q1/Q3 · IQR · range |

### Hybrid Ensemble
```
RUL_final = α × RUL_XGBoost + (1−α) × RUL_LSTM
```
α is tuned on a **held-out set** (separate from validation) via bounded optimization.

---

## 📊 Results

| Dataset | Conditions | Faults | Hybrid RMSE | Hybrid R² |
|---------|-----------|--------|-------------|-----------|
| FD001   | 1         | 1      | ~13         | ~0.91     |
| FD002   | 6         | 1      | ~20         | ~0.76     |
| FD003   | 1         | 2      | ~14         | ~0.90     |
| FD004   | 6         | 2      | ~24         | ~0.70     |

---

## 🖥️ Dashboard Features

| Tab | Description |
|-----|-------------|
| 🔍 Engine Inspector | Per-engine RUL gauge, status badge, model comparison bar chart |
| 📊 Model Comparison | RMSE/MAE/R² bars, scatter plots, residual distributions |
| 🗺️ Fleet Heatmap | Colour-coded health grid for all test engines + donut summary |
| 📋 Predictions Table | Filterable/sortable table with true vs predicted RUL |
| ⬇️ Download | Export predictions as CSV or JSON |

---

## 🛠️ Deploy to Streamlit Cloud

### Step 1 — Fork / clone this repo
```bash
git clone https://github.com/darshan99009/predictive-main.git
cd predictive-main
```

### Step 2 — Connect to Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **New app**
3. Select repository: `darshan99009/predictive-main`
4. Branch: `main`
5. Main file: `app.py`
6. Click **Deploy** ✅

> Streamlit Cloud auto-installs `requirements.txt` — no extra setup needed.

---

## 💻 Run Locally

```bash
git clone https://github.com/darshan99009/predictive-main.git
cd predictive-main
pip install -r requirements.txt
streamlit run app.py
```

---

## 📖 How to Use the Dashboard

1. Open the live app
2. **Select dataset** (FD001–FD004) in the sidebar
3. **Upload** your `test_FDxxx.txt` file
4. Optionally upload `RUL_FDxxx.txt` to compute RMSE/MAE/R²
5. Predictions appear instantly — explore all tabs




---

## 📦 Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- 21 sensor readings per cycle
- 3 operational settings
- Run-to-failure trajectories
- RUL capped at 125 cycles

Dataset Uploded 

---

## 🔧 Tech Stack

- **Training:** PyTorch · XGBoost · Google Colab T4 GPU
- **Dashboard:** Streamlit · Plotly
- **Deployment:** Streamlit Cloud (free tier)
- **Data:** NASA C-MAPSS FD001–FD004
