"""
Predictive Analytics Using Historical Data
Internship Project - Forecasting Future Trends
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE / LOAD HISTORICAL DATASET
# ─────────────────────────────────────────────
np.random.seed(42)
months = pd.date_range(start="2021-01-01", periods=48, freq="MS")

# Simulated monthly sales data with trend + seasonality + noise
trend     = np.linspace(100, 280, 48)
seasonal  = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
noise     = np.random.normal(0, 8, 48)
sales     = trend + seasonal + noise

df = pd.DataFrame({
    "date":  months,
    "month": np.arange(48),          # numeric time index
    "sales": sales
})

print("=" * 55)
print("  PREDICTIVE ANALYTICS — HISTORICAL SALES DATA")
print("=" * 55)
print(f"\nDataset shape  : {df.shape}")
print(f"Date range     : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"\nDescriptive Statistics:\n{df['sales'].describe().round(2)}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n--- Preprocessing ---")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Outliers (IQR) : ", end="")

Q1, Q3 = df['sales'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = ((df['sales'] < Q1 - 1.5*IQR) | (df['sales'] > Q3 + 1.5*IQR)).sum()
print(outliers)

# Feature engineering
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df[['month', 'month_sin', 'month_cos']].values
y = df['sales'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nTrain samples  : {len(X_train)}")
print(f"Test  samples  : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. MODEL A — LINEAR REGRESSION
# ─────────────────────────────────────────────
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ─────────────────────────────────────────────
# 5. MODEL B — POLYNOMIAL REGRESSION (degree 2)
# ─────────────────────────────────────────────
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p  = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_p, y_train)
y_pred_pr = pr.predict(X_test_p)

# ─────────────────────────────────────────────
# 6. EVALUATE BOTH MODELS
# ─────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.2f}")
    print(f"    RMSE : {rmse:.2f}")
    print(f"    R²   : {r2:.4f}")
    return r2

print("\n--- Model Evaluation ---")
r2_lr = evaluate("Linear Regression",     y_test, y_pred_lr)
r2_pr = evaluate("Polynomial Regression", y_test, y_pred_pr)

best_model = "Polynomial" if r2_pr > r2_lr else "Linear"
print(f"\n  ✔ Best model: {best_model} Regression")

# ─────────────────────────────────────────────
# 7. FORECAST NEXT 12 MONTHS
# ─────────────────────────────────────────────
future_months = pd.date_range(start="2025-01-01", periods=12, freq="MS")
future_idx    = np.arange(48, 60)
future_sin    = np.sin(2 * np.pi * future_months.month / 12)
future_cos    = np.cos(2 * np.pi * future_months.month / 12)
X_future      = np.column_stack([future_idx, future_sin, future_cos])
X_future_p    = poly.transform(X_future)

forecast = pr.predict(X_future_p)

print("\n--- 12-Month Forecast (Polynomial Model) ---")
forecast_df = pd.DataFrame({
    "Month":    future_months.strftime("%b %Y"),
    "Forecast": np.round(forecast, 1)
})
print(forecast_df.to_string(index=False))

# Save results for dashboard
results = {
    "historical_months": [d.strftime("%b %Y") for d in df['date']],
    "historical_sales":  [round(v, 2) for v in df['sales'].tolist()],
    "test_start_idx":    len(X_train),
    "lr_preds":          [round(v, 2) for v in lr.predict(X).tolist()],
    "poly_preds":        [round(v, 2) for v in pr.predict(poly.transform(X)).tolist()],
    "forecast_months":   forecast_df["Month"].tolist(),
    "forecast_vals":     forecast_df["Forecast"].tolist(),
    "metrics": {
        "lr_r2":   round(r2_lr, 4),
        "poly_r2": round(r2_pr, 4),
        "poly_mae": round(mean_absolute_error(y_test, y_pred_pr), 2),
        "poly_rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_pr)), 2),
    }
}

import json
with open("/home/claude/predictive_analytics/results.json", "w") as f:
    json.dump(results, f)

print("\n✅ results.json saved — open dashboard.html to visualise")
print("=" * 55)
