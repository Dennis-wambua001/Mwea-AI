"""
train_and_forecast.py
---------------------
AI-Based Rice Growth & Yield Forecasting for Mwea Irrigation Scheme

✅ Reads yield data (CSV)
✅ Creates lag & time features
✅ Trains SARIMAX and LightGBM
✅ Forecasts yield through Dec 2030
✅ Saves forecast CSV + model metrics
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import lightgbm as lgb
import rasterio

# ========== Utility Functions ==========

def safe_rmse(y_true, y_pred):
    """Compute RMSE safely."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_df(y_true, y_pred):
    """Return MAE, RMSE, MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = safe_rmse(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}

# ========== Raster Helper ==========

def raster_mean_value(path):
    """Compute mean pixel value of a raster."""
    try:
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
            data = arr.compressed()
            return float(np.nanmean(data))
    except Exception as e:
        print(f"⚠️ Could not read raster {path}: {e}")
        return np.nan

# ========== Data Preprocessing ==========

def preprocess_data(csv_path):
    """Load and clean yield CSV data."""
    df = pd.read_csv(csv_path)

    # Handle date or year
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Year" in df.columns:
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
    else:
        raise ValueError("No 'date' or 'Year' column found in CSV!")

    # Sort and reset
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure target column
    if "Rice_Yield_tonnes" not in df.columns:
        raise ValueError(f"'Rice_Yield_tonnes' column not found! Found: {df.columns.tolist()}")

    # Add time features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["t"] = np.arange(len(df))

    return df

def create_lag_features(df, target="Rice_Yield_tonnes", lags=[1, 2, 3], rolls=[3, 6]):
    """Create lag and rolling mean features."""
    df = df.copy()
    for l in lags:
        df[f"{target}_lag_{l}"] = df[target].shift(l)
    for r in rolls:
        df[f"{target}_roll_{r}"] = df[target].shift(1).rolling(r, min_periods=1).mean()
    return df

# ========== Models ==========

def persistence_forecast(train, val_index, target):
    """Naive baseline: last known value."""
    last_val = train[target].iloc[-1]
    return np.full(len(val_index), last_val)

def sarimax_forecast(train, val, target, seasonal_period=12):
    """Train SARIMAX and forecast validation period."""
    y_train = train[target].astype(float)
    model = sm.tsa.SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    preds = res.get_forecast(steps=len(val)).predicted_mean
    return res, preds

def train_lightgbm(train, val, features, target):
    """Train LightGBM regressor."""
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 200,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train[features], train[target],
              eval_set=[(val[features], val[target])],
              eval_metric="rmse",
              verbose=False)
    return model

def iterative_lightgbm_forecast(model, history, features, target, future_dates):
    """Generate multi-step forecasts iteratively."""
    df_hist = history.copy()
    preds = []

    for date in future_dates:
        row = {
            "month": date.month,
            "year": date.year,
            "t": df_hist["t"].iloc[-1] + 1
        }
        for l in [1, 2, 3]:
            row[f"{target}_lag_{l}"] = df_hist[target].iloc[-l]
        for r in [3, 6]:
            row[f"{target}_roll_{r}"] = df_hist[target].iloc[-r:].mean()

        X = pd.DataFrame([row])[features]
        y_pred = model.predict(X)[0]
        preds.append(y_pred)

        row[target] = y_pred
        df_hist = pd.concat([df_hist, pd.DataFrame([row])], ignore_index=True)

    return np.array(preds)

# ========== Pipeline ==========

def run_pipeline():
    """Main forecasting pipeline."""
    print("Starting Rice Yield Forecasting Pipeline...")

    # Paths
    base_dir = r"D:\Documents\GIS practicals\GIS Prac\MWEA\AI yield system"
    csv_path = os.path.join(base_dir, "CSV", "Mwea data.csv")
    raster_dir = os.path.join(base_dir, "Raster")
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    df = preprocess_data(csv_path)
    df = create_lag_features(df, target="Rice_Yield_tonnes")
    df = df.dropna()

    # Raster features
    raster_features = {}
    if os.path.exists(raster_dir):
        for file in os.listdir(raster_dir):
            if file.lower().endswith(".tif"):
                mean_val = raster_mean_value(os.path.join(raster_dir, file))
                feature_name = os.path.splitext(file)[0].replace(" ", "_")
                raster_features[feature_name] = mean_val
                df[feature_name] = mean_val
        print(f"🛰️ Extracted {len(raster_features)} raster features")

    # Split data
    val_months = min(12, len(df)//5)
    train, val = df.iloc[:-val_months], df.iloc[-val_months:]
    target = "Rice_Yield_tonnes"

    # ========== Baseline (Persistence) ==========
    pers_preds = persistence_forecast(train, val.index, target)
    pers_metrics = metrics_df(val[target], pers_preds)

    # ========== SARIMAX ==========
    sarimax_model, sarimax_preds = sarimax_forecast(train, val, target)
    sarimax_metrics = metrics_df(val[target], sarimax_preds)
    joblib.dump(sarimax_model, os.path.join(out_dir, "sarimax_model.pkl"))

    # ========== LightGBM ==========
    features = ["month", "year", "t"] + \
               [f"{target}_lag_{i}" for i in [1, 2, 3]] + \
               [f"{target}_roll_{i}" for i in [3, 6]] + \
               list(raster_features.keys())

    model_lgb = train_lightgbm(train, val, features, target)
    joblib.dump(model_lgb, os.path.join(out_dir, "lightgbm_model.pkl"))

    lgb_preds = model_lgb.predict(val[features])
    lgb_metrics = metrics_df(val[target], lgb_preds)

    # ========== Forecast to 2030 ==========
    last_date = df["date"].max()
    forecast_index = pd.date_range(last_date + pd.offsets.MonthBegin(), "2030-12-01", freq="MS")
    future_preds = iterative_lightgbm_forecast(model_lgb, df, features, target, forecast_index)

    forecast_df = pd.DataFrame({
        "forecast_date": forecast_index,
        "forecast_tonnes": future_preds
    })
    forecast_df.to_csv(os.path.join(out_dir, "forecast_to_2030.csv"), index=False)

    # ========== Save metrics ==========
    metrics_all = pd.DataFrame({
        "Model": ["Persistence", "SARIMAX", "LightGBM"],
        "MAE": [pers_metrics["MAE"], sarimax_metrics["MAE"], lgb_metrics["MAE"]],
        "RMSE": [pers_metrics["RMSE"], sarimax_metrics["RMSE"], lgb_metrics["RMSE"]],
        "MAPE(%)": [pers_metrics["MAPE(%)"], sarimax_metrics["MAPE(%)"], lgb_metrics["MAPE(%)"]],
    })
    metrics_all.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False)

    print("\n✅ Forecasting complete!")
    print("Results saved in:", out_dir)
    print(metrics_all)
    print("\nForecast CSV → outputs/forecast_to_2030.csv")

# Run main
if __name__ == "__main__":
    run_pipeline()
