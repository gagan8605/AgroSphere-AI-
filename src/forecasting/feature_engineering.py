"""
AgroSphere AI - Feature Engineering & Signal Store
Constructs time-series lags, agro-climatic interaction terms, and tabular features for ML modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import PRODUCT_CATALOG

class FeatureEngineer:
    def __init__(self):
        pass

    def build_feature_matrix(
        self, 
        sales_agg_df: pd.DataFrame, 
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges daily sales aggregates with agro-climatic records and constructs lag + interaction features.
        """
        # Ensure date formats match
        sales = sales_agg_df.copy()
        weather = weather_df.copy()
        sales["date"] = pd.to_datetime(sales["date"])
        weather["date"] = pd.to_datetime(weather["date"])
        
        # Merge on (date, cluster_id)
        merged = pd.merge(
            sales,
            weather[["date", "cluster_id", "temperature_c", "relative_humidity_pct", 
                     "precipitation_mm", "pest_threat_index", "soil_water_stress_index", "is_forecast"]],
            on=["date", "cluster_id"],
            how="inner"
        )
        
        # Sort values chronologically per group
        merged = merged.sort_values(["cluster_id", "sku_id", "date"]).reset_index(drop=True)
        
        # Temporal Date Features
        merged["day_of_week"] = merged["date"].dt.dayofweek
        merged["day_of_month"] = merged["date"].dt.day
        merged["month"] = merged["date"].dt.month
        
        # Lagged Sales Features (7-day, 14-day rolling demand)
        grouped = merged.groupby(["cluster_id", "sku_id"])["total_quantity_ordered"]
        merged["sales_lag_1"] = grouped.shift(1).fillna(0)
        merged["sales_lag_7"] = grouped.shift(7).fillna(0)
        merged["sales_roll_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).fillna(0)
        merged["sales_roll_sum_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0)
        merged["sales_roll_mean_14"] = grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean()).fillna(0)
        
        # Domain-Specific Agro-Climatic Feature Interactions
        # 1. Biopesticide Threat Multiplier
        merged["bio_pest_interaction"] = merged["pest_threat_index"] * merged["sales_roll_mean_7"]
        
        # 2. Hydrogel Drought Stress Multiplier
        merged["hydro_stress_interaction"] = merged["soil_water_stress_index"] * merged["sales_roll_mean_7"]
        
        # 3. Moisture / Rain Saturation Factor
        merged["rain_humidity_factor"] = (merged["precipitation_mm"] / 50.0) * (merged["relative_humidity_pct"] / 100.0)
        
        # One-Hot / Categorical Mapping
        merged["sku_code_num"] = merged["sku_id"].astype("category").cat.codes
        merged["cluster_code_num"] = merged["cluster_id"].astype("category").cat.codes
        
        return merged

    def get_train_test_split(
        self, 
        feature_df: pd.DataFrame, 
        test_days: int = 14
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Splits feature matrix chronologically into training and validation sets.
        """
        # Filter to historical data only (exclude forward forecasts)
        hist_df = feature_df[feature_df["is_forecast"] == False].copy()
        max_date = hist_df["date"].max()
        split_date = max_date - pd.Timedelta(days=test_days)
        
        train_df = hist_df[hist_df["date"] <= split_date].copy()
        test_df = hist_df[hist_df["date"] > split_date].copy()
        
        feature_cols = [
            "temperature_c", "relative_humidity_pct", "precipitation_mm",
            "pest_threat_index", "soil_water_stress_index",
            "day_of_week", "day_of_month", "month",
            "sales_lag_1", "sales_lag_7", "sales_roll_mean_7", 
            "sales_roll_sum_7", "sales_roll_mean_14",
            "bio_pest_interaction", "hydro_stress_interaction", "rain_humidity_factor",
            "sku_code_num", "cluster_code_num"
        ]
        
        return train_df, test_df, feature_cols
