"""
AgroSphere AI - Unit Tests for Advisory Generation & Prompt Engineering
"""

import pytest
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.genai_advisory.advisory_generator import AdvisoryGenerator
from src.automation.alert_dispatcher import AlertDispatcher

def test_advisory_generation_english():
    generator = AdvisoryGenerator()
    sample_dealer_risk = {
        "dealer_id": "DLR-REG-1001",
        "dealer_name": "Jay Sardar Krushi Kendra",
        "contact_person": "Pravin Patel",
        "district": "Rajkot",
        "sku_id": "UB-BIO-01",
        "surge_pct": 42.0,
        "current_stock": 25,
        "recommended_reorder_qty": 200,
        "total_rain": 65.0,
        "avg_humidity": 85.0
    }
    
    advisory = generator.generate_advisory_text(sample_dealer_risk, language="English")
    assert "Jay Sardar Krushi Kendra" in advisory
    assert "AgroSphere" in advisory or "Bio-Shield" in advisory
    assert "200" in advisory

def test_advisory_generation_gujarati():
    generator = AdvisoryGenerator()
    sample_dealer_risk = {
        "dealer_id": "DLR-REG-1002",
        "dealer_name": "Kisan Sahay Kendra",
        "contact_person": "Hasmukh Patel",
        "district": "Junagadh",
        "sku_id": "UB-BIO-01",
        "surge_pct": 35.0,
        "current_stock": 10,
        "recommended_reorder_qty": 150,
        "total_rain": 40.0,
        "avg_humidity": 80.0
    }
    
    advisory = generator.generate_advisory_text(sample_dealer_risk, language="Gujarati")
    assert "Kisan Sahay Kendra" in advisory or "કૃષિ" in advisory or "એગ્રોસ્ફીયર" in advisory
    assert "૧૫૦" in advisory or "150" in advisory

def test_alert_dispatcher():
    import pandas as pd
    dispatcher = AlertDispatcher()
    
    df_dealers = pd.DataFrame([{
        "dealer_id": "DLR-REG-1001",
        "dealer_name": "Test Krushi Kendra",
        "contact_person": "Test Contact",
        "phone": "+91 98765 43210",
        "email": "test@agrosphere.ai",
        "district": "Central Hub",
        "sku_id": "UB-BIO-01",
        "product_name": "AgroSphere Bio-Shield",
        "current_stock": 15,
        "recommended_reorder_qty": 100,
        "surge_pct": 30.0,
        "risk_level": "SEVERE_STOCKOUT_RISK",
        "preferred_language": "English",
        "total_rain": 25.0,
        "avg_humidity": 75.0
    }])
    
    dispatched = dispatcher.dispatch_dealer_alerts(df_dealers, max_dispatches=1)
    assert len(dispatched) == 1
    assert dispatched[0]["status"] == "DELIVERED_200_OK"
    assert "dispatch_id" in dispatched[0]
