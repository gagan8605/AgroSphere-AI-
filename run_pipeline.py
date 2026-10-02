"""
AgroSphere AI - Master Command-Line Interface & Pipeline Launcher
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.automation.scheduler import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="🌾 AgroSphere AI: Agrochemical Supply Chain Demand Forecasting & Restocking Alerts"
    )
    parser.add_argument(
        "--retrain", 
        action="store_true", 
        default=False, 
        help="Retrain the XGBoost demand forecasting model on latest data"
    )
    parser.add_argument(
        "--dispatches", 
        type=int, 
        default=25, 
        help="Number of automated dealer alerts to dispatch (default: 25)"
    )
    parser.add_argument(
        "--dashboard", 
        action="store_true", 
        help="Launch the interactive Streamlit command center after pipeline execution"
    )
    
    args = parser.parse_args()
    
    # Execute core pipeline
    results = run_full_pipeline(retrain_model=args.retrain, max_dispatches=args.dispatches)
    
    if args.dashboard:
        print("\n🚀 Launching Streamlit Operations Dashboard...")
        dashboard_path = BASE_DIR / "app" / "dashboard.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main()
