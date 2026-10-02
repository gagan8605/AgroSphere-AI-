"""
AgroSphere AI - Core Configuration & Domain Product Metadata
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Organization & Facility Info
COMPANY_NAME = os.getenv("COMPANY_NAME", "AgroSphere AI")
COMPANY_HQ = os.getenv("COMPANY_HQ_LOCATION", "AgroSphere Central Operations Hub")
MANUFACTURING_PLANT = os.getenv("MANUFACTURING_HUB", "AgroSphere Central Logistics & Manufacturing Hub")

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# OpenWeather API (Optional)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Operational Thresholds
DEFAULT_SURGE_THRESHOLD_PCT = float(os.getenv("DEFAULT_SURGE_THRESHOLD_PCT", "25.0"))
DEFAULT_SAFETY_STOCK_DAYS = 7
FORECAST_HORIZON_DAYS = 14

# AgroSphere Product Catalog (5 Core Commercial Divisions)
PRODUCT_CATALOG = {
    "UB-BIO-01": {
        "name": "AgroSphere Bio-Shield",
        "category": "Green Crop Protection",
        "active_agent": "Trichoderma viride 1.5% WP (Bio-Fungicide)",
        "unit": "Liters",
        "pack_size": "1L / 5L",
        "shelf_life_months": 12,
        "target_threats": ["Root Rot", "Wilt", "Damping-off", "Fungal Blight"],
        "target_crops": ["Groundnut", "Cotton", "Vegetables", "Pulses"],
        "weather_triggers": {"min_humidity": 75, "min_rain_mm": 20, "temp_range": (24, 34)},
    },
    "UB-BIO-02": {
        "name": "AgroSphere Larvi-Kill",
        "category": "Green Crop Protection",
        "active_agent": "Bacillus thuringiensis (Bt) Bio-Larvicide",
        "unit": "Liters",
        "pack_size": "500ml / 1L",
        "shelf_life_months": 18,
        "target_threats": ["Spodoptera", "Bollworm", "Leaf Miner", "Pod Borer"],
        "target_crops": ["Cotton", "Castor", "Chilli", "Soybean"],
        "weather_triggers": {"min_humidity": 65, "min_rain_mm": 5, "temp_range": (26, 36)},
    },
    "UB-NUTR-01": {
        "name": "AgroSphere Aqua-Gel Polymer",
        "category": "Crop Nutrition & Water Management",
        "active_agent": "Potassium Polyacrylate Hydrogel (Soil Moisture Retainer)",
        "unit": "Kg",
        "pack_size": "5Kg / 25Kg Bag",
        "shelf_life_months": 36,
        "target_threats": ["Delayed Monsoon", "Dry Spell", "Water Stress", "Sandy Soil Leaching"],
        "target_crops": ["Groundnut", "Cotton", "Sugarcane", "Horticulture"],
        "weather_triggers": {"max_rain_mm": 10, "min_temp": 33, "consecutive_dry_days": 10},
    },
    "UB-NUTR-02": {
        "name": "AgroSphere Solu-NPK 19:19:19",
        "category": "Crop Nutrition & Water Management",
        "active_agent": "100% Water Soluble Crystalline Macro-Nutrient",
        "unit": "Kg",
        "pack_size": "1Kg / 25Kg Bag",
        "shelf_life_months": 24,
        "target_threats": ["Nutrient Deficiencies", "Early Vegetative Stunting"],
        "target_crops": ["Banana", "Cotton", "Sugarcane", "Wheat", "Vegetables"],
        "weather_triggers": {"sowing_window_active": True},
    },
    "UB-STIM-01": {
        "name": "AgroSphere Bio-Zyme Gold",
        "category": "Bio-Organic Formulations",
        "active_agent": "Ascophyllum Nodosum (Seaweed Extract) + Fulvic Bio-Stimulant",
        "unit": "Liters",
        "pack_size": "500ml / 1L / 5L",
        "shelf_life_months": 18,
        "target_threats": ["Flowering Drop", "Abiotic Stress", "Low Tillering"],
        "target_crops": ["Cotton", "Groundnut", "Mango", "Cumin", "Wheat"],
        "weather_triggers": {"temp_range": (20, 38)},
    },
    "UB-BULK-01": {
        "name": "AgroSphere Tech-Grade Amino 80%",
        "category": "Bulk Raw Material Manufacturing",
        "active_agent": "Enzymatically Hydrolyzed Plant Origin Amino Powder",
        "unit": "Kg",
        "pack_size": "25Kg Drum",
        "shelf_life_months": 24,
        "target_threats": ["B2B Formulator Input Demand"],
        "target_crops": ["All Commercial Crops"],
        "weather_triggers": {"bulk_manufacturing": True},
    },
    "UB-URBN-01": {
        "name": "AgroSphere Plant Revival Tonic",
        "category": "Urban Horticulture (Gardenica)",
        "active_agent": "Organic Bio-Nutrient Spray for Houseplants",
        "unit": "Bottles (500ml)",
        "pack_size": "500ml Spray",
        "shelf_life_months": 24,
        "target_threats": ["Indoor Plant Yellowing", "Urban Balcony Heat Stress"],
        "target_crops": ["Urban Gardening & Ornamental"],
        "weather_triggers": {"urban_weekend_demand": True},
    },
}

# Regional Distribution Clusters
REGIONAL_CLUSTERS = {
    "GJ-SAURASHTRA-01": {
        "district": "Rajkot",
        "sub_regions": ["Gondal", "Jetpur", "Jasdan", "Dhoraji"],
        "primary_crops": ["Groundnut", "Cotton", "Sesame", "Cumin"],
        "lat": 22.3039,
        "lon": 70.8022,
        "dealer_count": 280,
    },
    "GJ-SAURASHTRA-02": {
        "district": "Junagadh",
        "sub_regions": ["Keshod", "Manavadar", "Visavadar", "Una"],
        "primary_crops": ["Groundnut", "Mango (Kesar)", "Cotton", "Wheat"],
        "lat": 21.5222,
        "lon": 70.4579,
        "dealer_count": 210,
    },
    "GJ-CENTRAL-01": {
        "district": "Vadodara",
        "sub_regions": ["Padra", "Karjan", "Savli", "Dabhoi", "Waghodia"],
        "primary_crops": ["Cotton", "Tur (Pigeon Pea)", "Banana", "Vegetables"],
        "lat": 22.3072,
        "lon": 73.1812,
        "dealer_count": 240,
    },
    "GJ-CENTRAL-02": {
        "district": "Anand",
        "sub_regions": ["Petlad", "Borsad", "Khambhat", "Umreth", "Tarapur"],
        "primary_crops": ["Tobacco", "Banana", "Paddy", "Vegetables"],
        "lat": 22.5645,
        "lon": 72.9289,
        "dealer_count": 195,
    },
    "GJ-SOUTH-01": {
        "district": "Surat",
        "sub_regions": ["Bardoli", "Mandvi", "Olpad", "Mahuva"],
        "primary_crops": ["Sugarcane", "Paddy", "Banana", "Vegetables"],
        "lat": 21.1702,
        "lon": 72.8311,
        "dealer_count": 225,
    },
    "GJ-SOUTH-02": {
        "district": "Bharuch",
        "sub_regions": ["Ankleshwar", "Jambusar", "Amod", "Hansot"],
        "primary_crops": ["Cotton", "Sugarcane", "Wheat", "Banana"],
        "lat": 21.7051,
        "lon": 72.9959,
        "dealer_count": 160,
    },
    "GJ-NORTH-01": {
        "district": "Mehsana",
        "sub_regions": ["Kadi", "Visnagar", "Unjha", "Vijapur"],
        "primary_crops": ["Mustard", "Castor", "Fennel", "Cumin", "Potato"],
        "lat": 23.5880,
        "lon": 72.3693,
        "dealer_count": 190,
    },
}
