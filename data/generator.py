"""
AgroSphere AI - Synthetic Data Generator
Generates realistic historical sales, dealer registry (1500+ dealers), and agro-climatic weather logs.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure src is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import PRODUCT_CATALOG, REGIONAL_CLUSTERS, DATA_DIR

REGIONAL_SURNAME_LIST = [
    "Patel", "Sharma", "Gajera", "Savaliya", "Donga", "Verma", 
    "Chauhan", "Desai", "Shah", "Pandya", "Gohil", "Panchal", "Zala", "Parmar"
]

DEALER_PREFIX_LIST = [
    "Kisan Sahay", "Agro Care", "Shree Ram", "Navjivan", "Patel Agro", 
    "Krushi Vikas", "Green Earth", "Kisan Seva", "Bharat Krushi", "Vrundavan"
]

def generate_dealer_directory(num_dealers: int = 1500) -> pd.DataFrame:
    """Generates a realistic registry of 1,500+ regional agrochemical dealers."""
    random.seed(42)
    dealers = []
    
    cluster_keys = list(REGIONAL_CLUSTERS.keys())
    
    for i in range(1, num_dealers + 1):
        cluster_id = random.choice(cluster_keys)
        cluster_info = REGIONAL_CLUSTERS[cluster_id]
        district = cluster_info["district"]
        sub_region = random.choice(cluster_info["sub_regions"])
        
        prefix = random.choice(DEALER_PREFIX_LIST)
        surname = random.choice(REGIONAL_SURNAME_LIST)
        contact_name = f"{random.choice(['Pravin', 'Hasmukh', 'Ramesh', 'Mukesh', 'Kishor', 'Bharat', 'Haresh', 'Dinesh'])} {surname}"
        
        dealer_name = f"{prefix} Krushi Kendra ({sub_region})"
        dealer_id = f"DLR-REG-{1000 + i}"
        phone = f"+91 {random.randint(97000, 99999)} {random.randint(10000, 99999)}"
        email = f"dealer_{1000 + i}@{random.choice(['gmail.com', 'kisanagro.in', 'agrosphere.ai'])}"
        credit_limit_inr = random.choice([200000, 350000, 500000, 750000, 1000000])
        preferred_lang = random.choice(["Gujarati", "Gujarati", "Gujarati", "English", "Hindi"])
        
        dealers.append({
            "dealer_id": dealer_id,
            "dealer_name": dealer_name,
            "contact_person": contact_name,
            "phone": phone,
            "email": email,
            "district": district,
            "sub_region": sub_region,
            "cluster_id": cluster_id,
            "credit_limit_inr": credit_limit_inr,
            "preferred_language": preferred_lang,
            "status": "Active",
        })
        
    df = pd.DataFrame(dealers)
    filepath = DATA_DIR / "regional_dealer_directory.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} dealers saved to {filepath}")
    return df


def generate_weather_feeds(days_back: int = 120, days_forward: int = 14) -> pd.DataFrame:
    """Generates historical and 14-day forecast agro-climatic logs for regional clusters."""
    random.seed(42)
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=days_back)
    total_days = days_back + days_forward
    records = []
    
    for cluster_id, cluster_info in REGIONAL_CLUSTERS.items():
        district = cluster_info["district"]
        base_temp = 28.0 + (1.5 if "SOUTH" in cluster_id else 0.0)
        
        for d in range(total_days):
            current_date = start_date + timedelta(days=d)
            is_forecast = d >= days_back
            
            # Simulate seasonal weather patterns
            day_of_year = current_date.timetuple().tm_yday
            is_monsoon = 160 <= day_of_year <= 270
            
            if is_monsoon:
                rain_mm = round(max(0.0, float(np.random.exponential(scale=18.0)) if random.random() > 0.4 else 0.0), 1)
                humidity = round(float(np.clip(np.random.normal(loc=78, scale=10), 45, 98)), 1)
                temp = round(float(np.clip(np.random.normal(loc=base_temp + 2, scale=3), 24, 38)), 1)
            else:
                rain_mm = round(max(0.0, float(np.random.exponential(scale=2.0)) if random.random() > 0.9 else 0.0), 1)
                humidity = round(float(np.clip(np.random.normal(loc=48, scale=12), 25, 75)), 1)
                temp = round(float(np.clip(np.random.normal(loc=base_temp + 5, scale=4), 20, 42)), 1)
            
            # Specific simulated surge event for upcoming 14-day forecast demo
            if is_forecast and cluster_id in ["GJ-SAURASHTRA-01", "GJ-SAURASHTRA-02"]:
                rain_mm = round(float(np.random.uniform(25.0, 65.0)), 1)
                humidity = round(float(np.random.uniform(80.0, 95.0)), 1)
                temp = round(float(np.random.uniform(26.0, 31.0)), 1)
                
            # Calculate Pest/Pathogen Infestation Threat Index (0.0 to 1.0)
            pest_threat = 0.1
            if humidity > 75 and 24 <= temp <= 32 and rain_mm > 10:
                pest_threat = round(min(0.95, 0.5 + (humidity - 75)/50.0 + (rain_mm / 100.0)), 2)
            elif humidity > 70 and temp > 28:
                pest_threat = round(min(0.75, 0.3 + (humidity - 70)/60.0), 2)
                
            # Calculate Soil Water Stress Index (0.0 to 1.0)
            water_stress = round(float(np.clip((temp - 30)/15.0 + (1.0 - humidity/100.0) - (rain_mm/40.0), 0.0, 1.0)), 2)
            
            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "cluster_id": cluster_id,
                "district": district,
                "temperature_c": temp,
                "relative_humidity_pct": humidity,
                "precipitation_mm": rain_mm,
                "pest_threat_index": pest_threat,
                "soil_water_stress_index": water_stress,
                "is_forecast": is_forecast,
            })
            
    df = pd.DataFrame(records)
    filepath = DATA_DIR / "weather_feed_sample.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} weather logs saved to {filepath}")
    return df


def generate_sales_history(dealer_df: pd.DataFrame, days_back: int = 120) -> pd.DataFrame:
    """Generates 120 days of historical dispatch & restocking transactions across dealers and SKUs."""
    random.seed(42)
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=days_back)
    sales = []
    
    dealers = dealer_df.to_dict(orient="records")
    product_keys = list(PRODUCT_CATALOG.keys())
    
    for d in range(days_back):
        current_date = start_date + timedelta(days=d)
        date_str = current_date.strftime("%Y-%m-%d")
        
        active_dealers = random.sample(dealers, k=int(len(dealers) * random.uniform(0.12, 0.22)))
        
        for dealer in active_dealers:
            selected_skus = random.sample(product_keys, k=random.randint(1, 3))
            
            for sku in selected_skus:
                prod = PRODUCT_CATALOG[sku]
                base_demand = random.randint(10, 80)
                
                # Biopesticides higher during humid periods
                if "BIO" in sku and "SAURASHTRA" in dealer["cluster_id"]:
                    base_demand = int(base_demand * random.uniform(1.3, 1.9))
                # Hydrogel higher in dry spells
                if "NUTR-01" in sku:
                    base_demand = int(base_demand * random.uniform(1.1, 1.6))
                    
                stock_on_hand = random.randint(5, int(base_demand * 1.8))
                
                sales.append({
                    "order_id": f"ORD-REG-{len(sales) + 100001}",
                    "date": date_str,
                    "dealer_id": dealer["dealer_id"],
                    "dealer_name": dealer["dealer_name"],
                    "district": dealer["district"],
                    "cluster_id": dealer["cluster_id"],
                    "sku_id": sku,
                    "product_name": prod["name"],
                    "category": prod["category"],
                    "quantity_ordered": base_demand,
                    "unit": prod["unit"],
                    "dealer_stock_on_hand": stock_on_hand,
                    "dispatch_hub": "AgroSphere Central Logistics Hub",
                })
                
    df = pd.DataFrame(sales)
    filepath = DATA_DIR / "raw_sales_sample.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated {len(df)} sales transactions saved to {filepath}")
    return df


if __name__ == "__main__":
    print("[INIT] Initializing AgroSphere AI Synthetic Data Generator...")
    dealers = generate_dealer_directory(num_dealers=1500)
    weather = generate_weather_feeds(days_back=120, days_forward=14)
    sales = generate_sales_history(dealers, days_back=120)
    print("[SUCCESS] All datasets generated successfully in data/ directory!")
