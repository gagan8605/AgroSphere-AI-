"""
AgroSphere AI - Unit Tests for Forecasting & Feature Engineering Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.ingestion.sales_loader import SalesDataLoader
from src.ingestion.weather_service import WeatherService
from src.forecasting.feature_engineering import FeatureEngineer
from src.forecasting.risk_classifier import RiskClassifier

def test_sales_loader():
    loader = SalesDataLoader()
    df_sales = loader.load_raw_sales()
    assert not df_sales.empty
    assert "quantity_ordered" in df_sales.columns
    assert "dealer_id" in df_sales.columns
    
    agg = loader.get_aggregated_daily_demand()
    assert not agg.empty
    assert "total_quantity_ordered" in agg.columns

def test_weather_service():
    service = WeatherService()
    weather_df = service.load_weather_feed()
    assert not weather_df.empty
    assert "pest_threat_index" in weather_df.columns
    assert "soil_water_stress_index" in weather_df.columns
    
    summary = service.get_regional_agro_climatic_summary()
    assert not summary.empty
    assert "max_pest_threat" in summary.columns

def test_feature_engineering():
    loader = SalesDataLoader()
    service = WeatherService()
    engineer = FeatureEngineer()
    
    sales_agg = loader.get_aggregated_daily_demand()
    weather_df = service.load_weather_feed()
    
    feat_matrix = engineer.build_feature_matrix(sales_agg, weather_df)
    assert not feat_matrix.empty
    assert "bio_pest_interaction" in feat_matrix.columns
    assert "hydro_stress_interaction" in feat_matrix.columns
    
    train_df, test_df, feature_cols = engineer.get_train_test_split(feat_matrix)
    assert len(train_df) > 0
    assert len(feature_cols) > 10

def test_risk_classifier():
    classifier = RiskClassifier()
    # Mock forecast data
    mock_forecast = pd.DataFrame([
        {
            "cluster_id": "GJ-SAURASHTRA-01",
            "district": "Rajkot",
            "sku_id": "UB-BIO-01",
            "product_name": "AgroSphere Bio-Shield",
            "category": "Green Crop Protection",
            "unit": "Liters",
            "predicted_demand_qty": 100.0,
            "baseline_historical_qty": 50.0,
            "temperature_c": 28.0,
            "relative_humidity_pct": 85.0,
            "precipitation_mm": 40.0,
            "pest_threat_index": 0.85,
            "soil_water_stress_index": 0.1
        }
    ])
    
    surges = classifier.evaluate_regional_surges(mock_forecast)
    assert not surges.empty
    assert surges["surge_status"].iloc[0] == "CRITICAL_SURGE"
    assert surges["is_alert_triggered"].iloc[0] == True
