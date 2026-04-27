# Retail Store Daily Sales Forecasting
**TEAM-4 | AI66B | Time Series Analysis — Final Project**

---

## Team Members
| Full Name | Student ID |
|---|---|
| Chử Vũ Thảo Hiền | 11247287 |
| Phan Thị Anh Quỳnh | 11247347 |

---

## Problem Overview
Accurate daily sales forecasting is critical for retail chains managing hundreds of stores and thousands of product families. Poor forecasts lead to either stockouts (lost revenue, damaged reputation) or overstock (wasted storage costs). This project builds and compares multiple forecasting models on real-world retail data, incorporating external factors such as promotions, oil prices, and public holidays.

---

## Dataset
- **Source:** [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) (Kaggle)
- **Scope:** Store 1 | Product family: GROCERY I
- **Period:** January 2013 – August 2017 (~1,700 daily observations)
- **Raw files:** `train.csv`, `oil.csv`, `holidays_events.csv`

---

## Pipeline

### Stage 1 — Data Preparation

| File | Description |
|---|---|
| `PREP_TRAIN.ipynb` | Load `train.csv`, EDA (sales by store, rolling mean), export `train_cleaned.csv` |
| `PREP_OIL.ipynb` | Load `oil.csv`, forward-fill missing prices, create lag features (`lag_1`, `lag_7`), export `oil_cleaned.csv` |
| `PREP_HOLIDAY.ipynb` | Load `holidays_events.csv`, filter national holidays, create `is_holiday` flag, export `holidays_cleaned.csv` |
| `MERGE_DATASET.ipynb` | Left-join train + oil + holidays on `date`, fill missing values, export `full_data_merged.csv` |

### Stage 2 — Modeling

| File | Description |
|---|---|
| `STATIONARY.ipynb` | ADF & KPSS stationarity tests, ACF/PACF, baseline models AR/MA/ARMA, STL decomposition |
| `COMPARE.ipynb` | Side-by-side comparison: AR(1), MA(1), ARMA(1,1), ARIMA(1,1,1), SARIMA(1,1,1)(1,1,1,7) |
| `SARIMA.ipynb` | SARIMA grid search (fixed d=1), residual diagnostics, Ljung-Box test |
| `SARIMAX.ipynb` | SARIMAX with `is_holiday` exogenous variable, best config: (1,1,2)(0,1,1,7), 30-day forecast |
| `FINAL STAGE.ipynb` | Feature engineering → XGBoost, LightGBM (walk-forward CV) → RNN, LSTM, GRU, CNN-LSTM → Hybrid models |

---

## Results Summary

| Model | MAE | RMSE |
|---|---|---|
| AR / MA / ARMA | ~839 | ~971 |
| ARIMA(1,1,1) | ~698 | ~868 |
| SARIMA(1,1,1)(1,1,1,7) | 371 | 589 |
| SARIMAX(1,1,2)(0,1,1,7) | 645 | 873 |
| XGBoost | ~128 | ~175 |
| **LightGBM** | **~125** | **~160** |
| Hybrid B (SARIMA + DL) | ~304 | ~491 |

> ML models (XGBoost, LightGBM) significantly outperform classical time series models, benefiting from rich lag and calendar features. SARIMA remains a strong classical baseline.

---

## Project Structure
```
├── PREP_TRAIN.ipynb          # Train data loading & EDA
├── PREP_OIL.ipynb            # Oil price preprocessing
├── PREP_HOLIDAY.ipynb        # Holiday feature engineering
├── MERGE_DATASET.ipynb       # Merge all sources into one dataset
│
├── STATIONARY.ipynb          # Stationarity tests, baseline models, STL
├── COMPARE.ipynb             # Classical model comparison
├── SARIMA.ipynb              # SARIMA modeling
├── SARIMAX.ipynb             # SARIMAX with exogenous variables
├── FINAL STAGE.ipynb         # ML, DL, and Hybrid models
│
└── full_data_merged.csv      # Final preprocessed dataset
```

---

## Requirements
```
pandas numpy matplotlib statsmodels scikit-learn xgboost lightgbm tensorflow
```
