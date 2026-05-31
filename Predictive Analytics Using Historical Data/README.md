# Predictive Analytics Using Historical Data
**Internship Project | Due: 22 May 2026**

## Project Overview
Build a predictive model to forecast future trends using historical monthly sales data.

---

## Files
| File | Purpose |
|------|---------|
| `predictive_model.py` | Full ML pipeline — data prep, training, evaluation, forecast |
| `dashboard.html` | Interactive visualization dashboard (open in browser) |
| `results.json` | Model outputs used by the dashboard |

---

## How to Run

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### 2. Run the model
```bash
python predictive_model.py
```

### 3. View the dashboard
Open `dashboard.html` in any web browser (no server needed).

---

## What the Model Does

### Step 1 — Data Loading
- 48 months of monthly sales data (Jan 2021 – Dec 2024)
- Features: time index, cyclic month encoding (sin/cos)

### Step 2 — Preprocessing
- Missing value check
- IQR-based outlier detection
- Cyclic feature engineering for seasonality

### Step 3 — Train/Test Split
- 80/20 chronological split (no shuffling — prevents data leakage)
- Train: 38 samples | Test: 10 samples

### Step 4 — Models Trained
| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | **0.5994** | 6.09 | 7.88 |
| Polynomial Regression (deg 2) | 0.2599 | 8.95 | 10.71 |

**Winner: Linear Regression** (better generalization on test set)

### Step 5 — 12-Month Forecast
| Month | Predicted Sales |
|-------|----------------|
| Jan 2025 | 286.1 |
| Feb 2025 | 299.6 |
| Mar 2025 | 312.4 |
| Apr 2025 | 321.9 |
| May 2025 | 326.8 |
| Jun 2025 | 327.5 |
| Jul 2025 | 325.4 |
| Aug 2025 | 322.6 |
| Sep 2025 | 320.8 |
| Oct 2025 | 321.4 |
| Nov 2025 | 325.5 |
| Dec 2025 | 333.6 |

---

## Key Learnings
- **Regression models** can effectively capture trend + seasonality
- **Polynomial features** don't always improve performance (risk of overfitting)
- **Cyclic encoding** (sin/cos) is better than raw month integers for seasonality
- **R² score** measures goodness of fit; closer to 1.0 = better
- **MAE/RMSE** measure average error in original units

---

## Expected Outcome (from brief)
✅ Predictive modeling  
✅ Trend analysis  
✅ Data-driven forecasting  
✅ Model accuracy evaluation  
✅ Prediction visualization  
