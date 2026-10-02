"""
AgroSphere AI - Risk Classifier & Stockout Anomaly Engine
Evaluates demand surges, identifies high-risk dealer stockouts, and computes optimal reorder sizes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DEFAULT_SURGE_THRESHOLD_PCT, DEFAULT_SAFETY_STOCK_DAYS, PRODUCT_CATALOG

class RiskClassifier:
    def __init__(self, surge_threshold_pct: float = DEFAULT_SURGE_THRESHOLD_PCT):
        self.surge_threshold_pct = surge_threshold_pct

    def evaluate_regional_surges(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates forward 14-day forecasts to identify regional SKU surges and risk triggers.
        """
        grouped = forecast_df.groupby(
            ["cluster_id", "district", "sku_id", "product_name", "category", "unit"], 
            as_index=False
        ).agg(
            total_predicted_14d_demand=("predicted_demand_qty", "sum"),
            baseline_14d_demand=("baseline_historical_qty", "sum"),
            avg_temp=("temperature_c", "mean"),
            avg_humidity=("relative_humidity_pct", "mean"),
            total_rain=("precipitation_mm", "sum"),
            max_pest_threat=("pest_threat_index", "max"),
            avg_water_stress=("soil_water_stress_index", "mean")
        )
        
        # Calculate surge percentage
        grouped["surge_pct"] = (
            (grouped["total_predicted_14d_demand"] - grouped["baseline_14d_demand"]) / 
            grouped["baseline_14d_demand"].replace(0, 1)
        ) * 100.0
        grouped["surge_pct"] = grouped["surge_pct"].round(1)
        
        # Classify Surge Level
        def classify_surge(row):
            if row["surge_pct"] >= self.surge_threshold_pct:
                if row["max_pest_threat"] >= 0.7 or row["avg_water_stress"] >= 0.7:
                    return "CRITICAL_SURGE"
                return "HIGH_SURGE"
            elif row["surge_pct"] >= 10.0:
                return "MODERATE_SURGE"
            elif row["surge_pct"] <= -15.0:
                return "DEMAND_DROP"
            return "NORMAL"
            
        grouped["surge_status"] = grouped.apply(classify_surge, axis=1)
        grouped["is_alert_triggered"] = grouped["surge_status"].isin(["CRITICAL_SURGE", "HIGH_SURGE"])
        
        return grouped

    def evaluate_dealer_stockout_risks(
        self, 
        regional_surges_df: pd.DataFrame, 
        dealer_snapshot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Cross-references regional surges against localized dealer stock levels to prioritize alerts.
        """
        merged = pd.merge(
            dealer_snapshot_df,
            regional_surges_df[[
                "cluster_id", "sku_id", "surge_pct", "surge_status", 
                "max_pest_threat", "avg_humidity", "total_rain", "is_alert_triggered"
            ]],
            on=["cluster_id", "sku_id"],
            how="inner"
        )
        
        # Estimate 14-day dealer demand based on regional surge
        daily_baseline = merged["recent_order_qty"] / 7.0
        merged["estimated_14d_dealer_demand"] = (daily_baseline * 14.0 * (1.0 + (merged["surge_pct"] / 100.0))).round(0)
        
        # Calculate stock deficit
        # Safety stock buffer = 7 days of surge demand
        safety_stock = (daily_baseline * DEFAULT_SAFETY_STOCK_DAYS * (1.0 + (merged["surge_pct"] / 100.0))).round(0)
        merged["recommended_reorder_qty"] = np.maximum(
            0, 
            (merged["estimated_14d_dealer_demand"] + safety_stock - merged["current_stock"]).round(0)
        ).astype(int)
        
        # Classify Dealer Stockout Risk
        def classify_dealer_risk(row):
            if row["current_stock"] < (row["estimated_14d_dealer_demand"] * 0.4) and row["is_alert_triggered"]:
                return "SEVERE_STOCKOUT_RISK"
            elif row["current_stock"] < row["estimated_14d_dealer_demand"] and row["is_alert_triggered"]:
                return "MODERATE_STOCKOUT_RISK"
            elif row["current_stock"] < (row["estimated_14d_dealer_demand"] * 0.5):
                return "LOW_STOCK_WARNING"
            return "ADEQUATELY_STOCKED"
            
        merged["risk_level"] = merged.apply(classify_dealer_risk, axis=1)
        
        # Filter to only actionable dealers needing replenishment
        actionable = merged[merged["risk_level"].isin(["SEVERE_STOCKOUT_RISK", "MODERATE_STOCKOUT_RISK"])].copy()
        actionable = actionable.sort_values(by=["risk_level", "surge_pct"], ascending=[True, False]).reset_index(drop=True)
        
        return actionable
