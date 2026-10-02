"""
AgroSphere AI - Demand Forecasting Model
Trains and executes XGBoost / Gradient Boosting regression models to forecast product demand surges.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from typing import Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, PRODUCT_CATALOG, REGIONAL_CLUSTERS

class DemandForecastModel:
    def __init__(self, model_path: Path = None):
        self.model_path = model_path or (MODELS_DIR / "demand_forecast_model.joblib")
        self.model: XGBRegressor = None
        self.feature_cols = []

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list) -> Dict[str, float]:
        """
        Trains the XGBoost demand regression model and evaluates on test set.
        """
        self.feature_cols = feature_cols
        
        X_train = train_df[self.feature_cols]
        y_train = train_df["total_quantity_ordered"]
        
        X_test = test_df[self.feature_cols]
        y_test = test_df["total_quantity_ordered"]
        
        self.model = XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        preds = np.clip(preds, a_min=0, a_max=None)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        
        # Calculate MAPE avoiding zero division
        non_zero_mask = y_test > 0
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_test[non_zero_mask] - preds[non_zero_mask]) / y_test[non_zero_mask])) * 100)
        else:
            mape = 0.0
            
        metrics = {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        }
        
        # Save model and metadata
        joblib.dump({
            "model": self.model,
            "feature_cols": self.feature_cols,
            "metrics": metrics
        }, self.model_path)
        
        return metrics

    def load(self):
        """Loads trained model checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train model first.")
        checkpoint = joblib.load(self.model_path)
        self.model = checkpoint["model"]
        self.feature_cols = checkpoint["feature_cols"]

    def forecast_upcoming_demand(
        self, 
        feature_matrix_df: pd.DataFrame, 
        weather_forecast_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generates 14-day forward forecast demand per cluster and SKU.
        """
        if self.model is None:
            self.load()
            
        # Get latest historical rows to seed lag calculations
        hist_df = feature_matrix_df[feature_matrix_df["is_forecast"] == False]
        latest_date = hist_df["date"].max()
        
        forecast_rows = []
        
        # For each cluster and SKU, project forward across upcoming weather dates
        clusters = list(REGIONAL_CLUSTERS.keys())
        skus = list(PRODUCT_CATALOG.keys())
        
        for cluster_id in clusters:
            cluster_weather = weather_forecast_df[
                (weather_forecast_df["cluster_id"] == cluster_id) & 
                (weather_forecast_df["date"] > latest_date)
            ].sort_values("date")
            
            for sku_id in skus:
                prod = PRODUCT_CATALOG[sku_id]
                
                # Get baseline recent 7-day average from history
                sub_hist = hist_df[(hist_df["cluster_id"] == cluster_id) & (hist_df["sku_id"] == sku_id)]
                recent_7d_mean = sub_hist["total_quantity_ordered"].tail(7).mean() if len(sub_hist) > 0 else 50.0
                recent_14d_mean = sub_hist["total_quantity_ordered"].tail(14).mean() if len(sub_hist) > 0 else 50.0
                
                # Code categoricals
                sku_code = feature_matrix_df["sku_id"].astype("category").cat.categories.get_loc(sku_id) if sku_id in feature_matrix_df["sku_id"].values else 0
                cluster_code = feature_matrix_df["cluster_id"].astype("category").cat.categories.get_loc(cluster_id) if cluster_id in feature_matrix_df["cluster_id"].values else 0
                
                running_lag_1 = recent_7d_mean
                
                for _, w_row in cluster_weather.iterrows():
                    w_date = pd.to_datetime(w_row["date"])
                    temp = w_row["temperature_c"]
                    humidity = w_row["relative_humidity_pct"]
                    rain = w_row["precipitation_mm"]
                    pest_threat = w_row["pest_threat_index"]
                    water_stress = w_row["soil_water_stress_index"]
                    
                    feat_dict = {
                        "temperature_c": temp,
                        "relative_humidity_pct": humidity,
                        "precipitation_mm": rain,
                        "pest_threat_index": pest_threat,
                        "soil_water_stress_index": water_stress,
                        "day_of_week": w_date.dayofweek,
                        "day_of_month": w_date.day,
                        "month": w_date.month,
                        "sales_lag_1": running_lag_1,
                        "sales_lag_7": recent_7d_mean,
                        "sales_roll_mean_7": recent_7d_mean,
                        "sales_roll_sum_7": recent_7d_mean * 7,
                        "sales_roll_mean_14": recent_14d_mean,
                        "bio_pest_interaction": pest_threat * recent_7d_mean,
                        "hydro_stress_interaction": water_stress * recent_7d_mean,
                        "rain_humidity_factor": (rain / 50.0) * (humidity / 100.0),
                        "sku_code_num": sku_code,
                        "cluster_code_num": cluster_code,
                    }
                    
                    feat_vector = pd.DataFrame([feat_dict])[self.feature_cols]
                    pred_val = float(self.model.predict(feat_vector)[0])
                    pred_val = max(5.0, pred_val) # non-negative demand
                    
                    forecast_rows.append({
                        "date": w_date.strftime("%Y-%m-%d"),
                        "cluster_id": cluster_id,
                        "district": REGIONAL_CLUSTERS[cluster_id]["district"],
                        "sku_id": sku_id,
                        "product_name": prod["name"],
                        "category": prod["category"],
                        "unit": prod["unit"],
                        "predicted_demand_qty": round(pred_val, 1),
                        "baseline_historical_qty": round(recent_7d_mean, 1),
                        "temperature_c": temp,
                        "relative_humidity_pct": humidity,
                        "precipitation_mm": rain,
                        "pest_threat_index": pest_threat,
                        "soil_water_stress_index": water_stress,
                    })
                    
                    running_lag_1 = pred_val
                    
        return pd.DataFrame(forecast_rows)
